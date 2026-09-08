"""Utility functions for inspecting and extracting information from Python modules."""

from typing import Any, List, Tuple, Optional, Dict
import ast
import re


def get_all_members(module: Any) -> List[Tuple[str, Any]]:
    """Safely extracts all members from a module, bypassing __getattr__ obfuscation.

    Many modern ML frameworks use lazy-loading modules that define __getattr__
    and __all__ but do not correctly implement __dir__. This function combines
    dir() and __all__ to forcefully resolve hidden members.

    Args:
        module: The module to inspect.

    Returns:
        A list of (name, object) tuples.

    """
    members = {}

    # 1. Standard dir() lookup
    for name in dir(module):
        try:
            members[name] = getattr(module, name)
        except Exception:  # pragma: no cover
            pass

    # 2. Aggressive __all__ resolution for lazy loaded modules
    if hasattr(module, "__all__"):
        for name in module.__all__:
            if name not in members:
                try:
                    members[name] = getattr(module, name)
                except Exception:
                    pass

    return list(members.items())


class CExtensionSignature(List[Tuple[str, str, Optional[str], Optional[str]]]):
    """Extracted signature parameter list for C-extensions with overloads support."""

    def __init__(
        self,
        params: List[Tuple[str, str, Optional[str], Optional[str]]],
        returns_type: Optional[str] = None,
        overloads: Optional[List["CExtensionSignature"]] = None,
    ) -> None:
        """Initialize signature with parameters, return type, and optional overloads.

        Args:
            params: List of parameter tuples: (name, kind, default, annotation).
            returns_type: Optional return type annotation string.
            overloads: Optional list of secondary/tertiary overload signatures.
        """
        super().__init__(params)
        self.returns_type: Optional[str] = returns_type
        self.overloads: List["CExtensionSignature"] = overloads or []


def _normalize_c_sig_args(args_str: str) -> str:
    """Normalize C++ / PyBind / ATen parameter syntax to Python syntax for AST parsing.

    Converts signatures like 'Tensor self, bool inplace=False' into 'self: "Tensor" = False'.

    Args:
        args_str: Raw parameter string from C-extension docstring.

    Returns:
        Normalized Python parameter string.
    """
    tokens: List[str] = []
    depth = 0
    cur: List[str] = []
    for char in args_str:
        if char in "<[(":
            depth += 1
            cur.append(char)
        elif char in ">])":
            depth -= 1
            cur.append(char)
        elif char == "," and depth == 0:
            tokens.append("".join(cur).strip())
            cur = []
        else:
            cur.append(char)
    if cur:
        tokens.append("".join(cur).strip())

    transformed: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        if tok in ("*", "/"):
            transformed.append(tok)
            continue
        if tok.startswith("*"):
            transformed.append(tok)
            continue
        if ":" in tok:
            transformed.append(tok)
            continue
        # C++ / PyBind style: Type [&*] name [= default]
        m = re.match(
            r"^(?:const\s+)?([a-zA-Z_][\w:]*(?:<[^>]+>)?(?:\s*[*&]+)?)\s+([a-zA-Z_]\w*)(?:\s*=\s*(.+))?$",
            tok,
        )
        if m:
            typ, name, default = m.group(1).strip(), m.group(2).strip(), m.group(3)
            clean_typ = typ.replace("&", "").replace("*", "").strip()
            if default is not None:
                transformed.append(f'{name}: "{clean_typ}" = {default.strip()}')
            else:
                transformed.append(f'{name}: "{clean_typ}"')
        else:
            transformed.append(tok)
    return ", ".join(transformed)


def _parse_c_extension_sig_str(
    sig_line: str,
) -> Optional[
    Tuple[
        str,
        List[Tuple[str, str, Optional[str], Optional[str]]],
        Optional[str],
    ]
]:
    """Parse a single signature line into function name, params, and return type.

    Args:
        sig_line: A single signature line (e.g. `add(input, other) -> Tensor`).

    Returns:
        A tuple of (func_name, extracted_params, returns_type), or None if parsing fails.
    """
    clean_line = sig_line.strip()
    clean_line = re.sub(r"^(?:\d+[\.\)]\s*)?(?:aten::|c10::|torch\.)", "", clean_line)
    pattern = r"^(?:\d+[\.\)]\s*)?([a-zA-Z0-9_]+)\((.*)\)(?:\s*->\s*(.*))?$"
    match = re.match(pattern, clean_line)
    if not match:
        return None

    func_name = match.group(1)
    args_str = match.group(2)
    ret_type_str = match.group(3).strip() if match.group(3) else None

    dummy_code = f"def dummy_func({args_str}): pass"
    try:
        tree = ast.parse(dummy_code)
        func_def = tree.body[0]
        args = func_def.args  # type: ignore
    except SyntaxError:
        norm_args = _normalize_c_sig_args(args_str)
        try:
            tree = ast.parse(f"def dummy_func({norm_args}): pass")
            func_def = tree.body[0]
            args = func_def.args  # type: ignore
        except SyntaxError:
            return None

    extracted_params: List[Tuple[str, str, Optional[str], Optional[str]]] = []

    def unparse_anno(node: Any) -> Optional[str]:
        """Convert an AST annotation node back to a string.

        Args:
            node: AST node.

        Returns:
            String representation or None.
        """
        if node is None:
            return None
        return ast.unparse(node)

    def unparse_default(node: Any) -> Optional[str]:
        """Convert an AST default value node to a string representation.

        Args:
            node: AST node.

        Returns:
            String representation or None.
        """
        try:
            val = ast.literal_eval(node)
            if isinstance(val, str):
                return repr(val)
            return str(val)
        except ValueError:
            return ast.unparse(node)

    num_defaults = len(args.defaults)
    posonly_count = len(getattr(args, "posonlyargs", []))
    if hasattr(args, "posonlyargs"):
        combined_args = getattr(args, "posonlyargs", []) + args.args
    else:  # pragma: no cover
        combined_args = args.args

    num_combined = len(combined_args)
    default_offset = num_combined - num_defaults

    for i, arg in enumerate(combined_args):
        p_name = arg.arg
        if p_name == "self":
            continue
        p_kind = "POSITIONAL_ONLY" if i < posonly_count else "POSITIONAL_OR_KEYWORD"
        p_anno = unparse_anno(arg.annotation)
        p_def = (
            unparse_default(args.defaults[i - default_offset])
            if i >= default_offset
            else None
        )
        extracted_params.append((p_name, p_kind, p_def, p_anno))

    if args.vararg:
        p_name = args.vararg.arg
        p_kind = "VAR_POSITIONAL"
        p_anno = unparse_anno(args.vararg.annotation)
        extracted_params.append((p_name, p_kind, None, p_anno))

    for i, arg in enumerate(args.kwonlyargs):
        p_name = arg.arg
        p_kind = "KEYWORD_ONLY"
        p_anno = unparse_anno(arg.annotation)
        default_node = args.kw_defaults[i]
        p_def = unparse_default(default_node) if default_node else None
        extracted_params.append((p_name, p_kind, p_def, p_anno))

    if args.kwarg:
        p_name = args.kwarg.arg
        p_kind = "VAR_KEYWORD"
        p_anno = unparse_anno(args.kwarg.annotation)
        extracted_params.append((p_name, p_kind, None, p_anno))

    return func_name, extracted_params, ret_type_str


def extract_c_extension_signature(
    target: Any, target_name: str
) -> Optional[CExtensionSignature]:
    """Extract signature parameters from C-extension / PyBind11 docstrings.

    Many compiled ML frameworks (like Torch, PyBind11 modules) do not support
    inspect.signature. Their signatures are often embedded in the first lines of
    the docstring (e.g., `relu(input: Tensor) -> Tensor`), potentially with
    multiple consecutive or numbered overloads.

    Args:
        target: The callable object to inspect.
        target_name: The name of the callable.

    Returns:
        A CExtensionSignature list of parameter tuples: (name, kind, default_str, annotation_str)
        with .returns_type and .overloads populated, or None if parsing fails.
    """
    docstring = getattr(target, "__doc__", None)
    if not docstring or not isinstance(docstring, str):
        return None

    lines = docstring.strip().split("\n")
    if not lines:  # pragma: no cover
        return None

    parsed_sigs: List[
        Tuple[
            str,
            List[Tuple[str, str, Optional[str], Optional[str]]],
            Optional[str],
        ]
    ] = []

    has_numbered = any(re.match(r"^\s*\d+[\.\)]", line) for line in lines)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if parsed_sigs and not has_numbered:
                break
        elif (parsed := _parse_c_extension_sig_str(stripped)) is not None:
            parsed_sigs.append(parsed)
        elif parsed_sigs and not has_numbered:
            if "overloaded function" in stripped.lower():
                has_numbered = True
            else:
                break
        elif "overloaded function" in stripped.lower():
            has_numbered = True

    if not parsed_sigs:
        # Fallback: scan lines for target_name(...) calls or docstring usage patterns
        for line in lines:
            stripped = line.strip()
            if "(" in stripped and ")" in stripped:
                clean_cand = re.sub(r"^(?:>>>|\.\.\.)\s*", "", stripped).strip()
                if (
                    clean_cand.startswith(f"{target_name}(")
                    or f".{target_name}(" in clean_cand
                    or target_name in clean_cand
                ):
                    parsed = _parse_c_extension_sig_str(clean_cand)
                    if parsed is not None:
                        parsed_sigs.append(parsed)
                        break

    if not parsed_sigs:
        return None

    if (
        len(parsed_sigs) > 1
        and has_numbered
        and len(parsed_sigs[0][1]) == 2
        and parsed_sigs[0][1][0][1] == "VAR_POSITIONAL"
        and parsed_sigs[0][1][1][1] == "VAR_KEYWORD"
    ):
        primary = parsed_sigs[1]
        secondary = parsed_sigs[2:]
    else:
        primary = parsed_sigs[0]
        secondary = parsed_sigs[1:]

    overload_objs = [
        CExtensionSignature(params=ov[1], returns_type=ov[2], overloads=[])
        for ov in secondary
    ]

    return CExtensionSignature(
        params=primary[1],
        returns_type=primary[2],
        overloads=overload_objs,
    )


def get_framework_docstring_parser(
    framework_name: str, docstring: Optional[str] = None
) -> str:
    """Determine the most appropriate docstring parser for a framework.

    Sphinx/reST is used by default for Sphinx-heavy frameworks (PyTorch, TensorFlow,
    Scikit-Learn). NumPy convention is used for NumPy/SciPy. Google convention
    is used for JAX/Keras/Transformers. If a docstring is provided, structural heuristics
    are evaluated to detect or dynamically fall back to the observed style.

    Args:
        framework_name: Name of the framework or top-level module (e.g. 'torch', 'numpy').
        docstring: Optional docstring text to inspect for format heuristics.

    Returns:
        The docstring parser style identifier ('rest', 'numpy', 'google', or 'auto').
    """
    if docstring and isinstance(docstring, str):
        # Heuristics on docstring content
        if any(
            marker in docstring
            for marker in (":param ", ":type ", ":return:", ":rtype:", ":raises:")
        ):
            return "rest"
        if re.search(r"\n(Parameters|Returns|Raises)\n\s*[-=]{3,}", docstring):
            return "numpy"
        if re.search(r"\n(Args|Arguments|Returns|Raises):\s*\n", docstring):
            return "google"

    # Framework-level heuristics
    name_clean = framework_name.lower().replace("-", "_")
    if name_clean in (
        "torch",
        "pytorch",
        "tensorflow",
        "tf",
        "sklearn",
        "scikit_learn",
        "scipy",
    ):
        return "rest"
    if name_clean in ("numpy", "np"):
        return "numpy"
    if name_clean in ("jax", "flax", "keras", "transformers", "huggingface"):
        return "google"

    return "rest"


def resolve_griffe_parser(parser_name: str) -> Any:
    """Resolve a parser identifier into a valid Griffe Parser instance.

    Ensures that 'rest' maps to griffe.Parser.rest if available, or gracefully
    falls back to griffe.Parser.sphinx (reST/Sphinx parser).

    Args:
        parser_name: The parser name ('rest', 'sphinx', 'google', 'numpy', 'auto').

    Returns:
        A Griffe Parser instance or None.
    """
    try:
        import griffe

        if not hasattr(griffe.Parser, "rest"):  # pragma: no branch
            try:
                # Provide transparent fallback hook for Parser('rest')
                setattr(
                    griffe.Parser,
                    "_missing_",
                    classmethod(lambda cls, val: cls.sphinx if val == "rest" else None),
                )
            except Exception:  # pragma: no cover
                pass

        if parser_name == "rest":
            return getattr(griffe.Parser, "rest", griffe.Parser.sphinx)

        return griffe.Parser(parser_name)
    except Exception:
        return None


def parse_docstring_with_griffe(
    docstring: str, parser_name: Optional[str] = None
) -> List[Any]:
    """Parse a docstring using Griffe with dynamic fallback across conventions.

    Tries the requested or detected parser first. If the resulting parse yields
    no structured sections (parameters, returns, or raises), it dynamically falls
    back through the alternative conventions ('rest', 'google', 'numpy').

    Args:
        docstring: The docstring text to parse.
        parser_name: Optional starting parser ('rest', 'google', 'numpy').

    Returns:
        A list of parsed Griffe DocstringSection objects.
    """
    try:
        import griffe
    except ImportError:  # pragma: no cover
        return []

    initial_parser = parser_name or "rest"
    candidates = [initial_parser]
    for alt in ("rest", "google", "numpy"):
        if alt not in candidates:
            candidates.append(alt)

    best_sections: List[Any] = []

    for candidate in candidates:
        resolved = resolve_griffe_parser(candidate)
        if not resolved:
            continue
        try:
            doc_obj = griffe.Docstring(docstring)
            sections = griffe.parse(doc_obj, resolved)
            has_structured = any(
                getattr(s.kind, "value", str(s.kind))
                in ("parameters", "returns", "raises")
                for s in sections
            )
            if has_structured:
                return sections
            if not best_sections and sections:
                best_sections = sections
        except Exception:  # pragma: no cover
            pass

    return best_sections


def strip_sphinx_roles(text: Optional[str]) -> Optional[str]:
    """Strip Sphinx cross-referencing roles from text.

    Transforms roles such as `:class:`~torch.Tensor`` or `:func:`relu``
    into clean representations (e.g. `torch.Tensor` or `relu`).

    Args:
        text: The input text containing potential Sphinx roles.

    Returns:
        Cleaned text with Sphinx roles stripped, or None if input is None.
    """
    if text is None:
        return None
    cleaned = str(text)
    # Strip Sphinx roles :role:`~target` -> target
    cleaned = re.sub(r":[a-zA-Z0-9_:]+:`~?([^`]*)`", r"\1", cleaned)
    return cleaned


def extract_griffe_docstring_metadata(
    docstring: str, parser_name: Optional[str] = None
) -> Dict[str, Any]:
    """Extract structured parameters, return types, and exceptions from a docstring.

    Leverages Griffe's parsed docstring section enumeration to capture parameter
    descriptions, structured types, defaults, return annotations, and raised exceptions.

    Args:
        docstring: The docstring text to parse.
        parser_name: Optional parser name to prioritize.

    Returns:
        A dictionary containing:
            'params': Dict[str, Dict[str, Optional[str]]],
            'returns': Optional[Dict[str, Optional[str]]],
            'raises': List[str]
    """
    sections = parse_docstring_with_griffe(docstring, parser_name=parser_name)

    params: Dict[str, Dict[str, Optional[str]]] = {}
    returns: Optional[Dict[str, Optional[str]]] = None
    raises: List[str] = []

    for section in sections:
        kind_name = getattr(section.kind, "value", str(section.kind))
        if (
            kind_name == "parameters"
            and hasattr(section, "value")
            and isinstance(section.value, list)
        ):
            for item in section.value:
                p_name = getattr(item, "name", None)
                if p_name:
                    annotation = getattr(item, "annotation", None)
                    value = getattr(item, "value", None)
                    desc = getattr(item, "description", None)
                    cleaned_typ = (
                        strip_sphinx_roles(str(annotation))
                        if annotation is not None
                        else None
                    )
                    params[p_name] = {
                        "doc": str(desc) if desc else "",
                        "typ": cleaned_typ,
                        "default": str(value) if value is not None else None,
                    }
        elif (
            kind_name == "returns"
            and hasattr(section, "value")
            and isinstance(section.value, list)
        ):
            for item in section.value:
                annotation = getattr(item, "annotation", None)
                desc = getattr(item, "description", None)
                cleaned_typ = (
                    strip_sphinx_roles(str(annotation))
                    if annotation is not None
                    else None
                )
                returns = {
                    "doc": str(desc) if desc else "",
                    "typ": cleaned_typ,
                }
        elif (
            kind_name == "raises"
            and hasattr(section, "value")
            and isinstance(section.value, list)
        ):
            for item in section.value:
                annotation = (
                    getattr(item, "annotation", None)
                    or getattr(item, "name", None)
                    or getattr(item, "value", None)
                )
                if annotation:
                    exc_name = strip_sphinx_roles(str(annotation)) or str(annotation)
                    if exc_name not in raises:
                        raises.append(exc_name)

    # Sphinx field lists: :raises <type>: or :raise <type>: or :except <type>:
    # Griffe's parser often treats :raises <type>: as a DocstringParameter in 'parameters'.
    for exc in re.findall(r":(?:raises?|except)\s+([A-Za-z0-9_.]+):", docstring):
        if exc in params:
            del params[exc]
        if exc not in raises:
            raises.append(exc)

    return {"params": params, "returns": returns, "raises": raises}
