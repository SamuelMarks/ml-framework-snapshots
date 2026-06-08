"""Utility functions for inspecting and extracting information from Python modules."""

from typing import Any, List, Tuple, Optional
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


def extract_c_extension_signature(
    target: Any, target_name: str
) -> Optional[List[Tuple[str, str, Optional[str], Optional[str]]]]:
    """Extract signature parameters from C-extension / PyBind11 docstrings.

    Many compiled ML frameworks (like Torch, PyBind11 modules) do not support
    inspect.signature. Their signatures are often embedded in the first line of
    the docstring (e.g., `relu(input: Tensor) -> Tensor`).

    Args:
        target: The callable object to inspect.
        target_name: The name of the callable.

    Returns:
        A list of parameter tuples: (name, kind, default_str, annotation_str),
        or None if parsing fails.

    """
    docstring = getattr(target, "__doc__", None)
    if not docstring or not isinstance(docstring, str):
        return None

    lines = docstring.strip().split("\n")
    if not lines:  # pragma: no cover
        return None

    first_line = lines[0].strip()

    # Try generic pattern: word(args) -> ret
    pattern = r"^([a-zA-Z0-9_]+)\((.*)\)(?:\s*->\s*(.*))?$"
    match = re.match(pattern, first_line)
    if not match:
        return None

    args_str = match.group(2)

    # Use Python's built-in AST parser for robustness against complex default values or annotations
    dummy_code = f"def dummy_func({args_str}): pass"
    try:
        tree = ast.parse(dummy_code)
        func_def = tree.body[0]
        args = func_def.args  # type: ignore
    except SyntaxError:
        return None

    extracted_params = []

    def unparse_anno(node: Any) -> Optional[str]:
        """Convert an AST annotation node back to a string.

        Args:
            node: node

        Returns:
            string or None
        """
        if node is None:
            return None
        return ast.unparse(node)

    def unparse_default(node: Any) -> Optional[str]:
        """Convert an AST default value node to a string representation.

        Args:
            node: node

        Returns:
            string or None
        """
        try:
            val = ast.literal_eval(node)
            if isinstance(val, str):
                return repr(val)
            else:  # pragma: no cover
                pass
            return str(val)
        except ValueError:
            return ast.unparse(node)

    num_defaults = len(args.defaults)

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
        p_kind = "POSITIONAL_OR_KEYWORD"
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

    return extracted_params
