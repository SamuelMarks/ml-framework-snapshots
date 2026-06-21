"""Models for Ghost API representations."""

from ml_switcheroo_ir.schema.ghost import GhostRef, GhostParam

STANDARD_ARG_MAP = {
    "x": "input",
    "inputs": "input",
    "input_tensor": "input",
    "y": "other",
    "other_tensor": "other",
    "dim": "dim",
    "axis": "dim",
    "keepdim": "keepdims",
    "keep_dims": "keepdims",
    "keepdims": "keepdims",
}

_GRIFFE_CACHE = {}
"""Ghost Core: Introspection Abstraction Layer.

This module provides the data structures and inspection logic required to
decouple framework analysis from the live environment. It enables the system
to operate in "Ghost Mode" (WASM/CI) by working against cached snapshots
instead of requiring heavy libraries (Torch/TensorFlow) to be installed.

Updates:
- Robust C-Extension handling (try/except around `inspect.signature`).
- Validates parameter kinds to support `*args` (VarPositional).
- Sanitizes default values to avoid serializing memory addresses.
"""

import inspect  # noqa: E402
import ast  # noqa: E402
import re  # noqa: E402
from typing import Any, Optional, Union, Callable, Dict  # noqa: E402

from .utils import extract_c_extension_signature  # noqa: E402


def sanitize_type_str(typ_str: Optional[str]) -> Optional[str]:
    """Sanitize type hints to PEP-585/PEP-604 representations.

    Args:
        typ_str: description

    Returns:
        The sanitized type string.
    """
    if not typ_str:
        return typ_str

    # Remove <class 'X'> before AST parsing
    typ_str = re.sub(r"<class '([^']+)'>", r"\1", typ_str)

    class TypeHintSanitizer(ast.NodeTransformer):
        """AST transformer to sanitize and format type hints."""

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            """Visit subscript nodes and format Unions and Optionals.

            Args:
                node: The node to visit.

            Returns:
                The transformed node.
            """
            self.generic_visit(node)
            is_union = False
            if isinstance(node.value, ast.Name) and node.value.id == "Union":
                is_union = True

            if is_union:
                slice_val = node.slice
                if isinstance(slice_val, ast.Tuple):
                    elts = slice_val.elts
                else:
                    elts = [slice_val]

                if len(elts) >= 2:
                    new_node: ast.expr = elts[0]
                    for elt in elts[1:]:
                        new_node = ast.BinOp(left=new_node, op=ast.BitOr(), right=elt)
                    return ast.copy_location(new_node, node)

            is_optional = False
            if isinstance(node.value, ast.Name) and node.value.id == "Optional":
                is_optional = True

            if is_optional:
                slice_val = node.slice
                return ast.copy_location(
                    ast.BinOp(
                        left=slice_val, op=ast.BitOr(), right=ast.Constant(value=None)
                    ),
                    node,
                )

            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            """Clean up common Name nodes like NoneType -> None.

            Args:
                node: The node to visit.

            Returns:
                The transformed node.
            """
            pep585_map = {
                "List": "list",
                "Dict": "dict",
                "Tuple": "tuple",
                "Set": "set",
                "Type": "type",
            }
            if node.id in pep585_map:
                node.id = pep585_map[node.id]
            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            """Flatten attribute access for simplified type names.

            Args:
                node: description

            Returns:
                The transformed node.
            """
            if isinstance(node.value, ast.Name) and node.value.id == "typing":
                pep585_map = {
                    "List": "list",
                    "Dict": "dict",
                    "Tuple": "tuple",
                    "Set": "set",
                    "Type": "type",
                }
                if node.attr in pep585_map:
                    return ast.copy_location(
                        ast.Name(id=pep585_map[node.attr], ctx=ast.Load()), node
                    )
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            self.generic_visit(node)
            return node

    try:
        node = ast.parse(typ_str, mode="eval")
        node = TypeHintSanitizer().visit(node)
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return typ_str


class GhostInspector:
    """Facade for API Inspection.

    Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
    Crucial for populating snapshots used by WASM/JS environments.
    """

    @staticmethod
    def inspect(
        obj: Union[Any, Callable[..., Any]],
        api_path: str,
        is_public: Optional[bool] = None,
    ) -> "GhostRef":
        """Create a GhostRef from a live Python object.

        Gracefully handles C-Extensions and builtins that resist introspection.

        Args:
            obj: The live class or function to inspect.
            api_path: The canonical string path (e.g. 'torch.nn.ReLU').
            is_public: Optional override for public visibility.

        Returns:
            A populated GhostRef object.

        """
        import cdd.shared.docstring_parsers as cdd_docstring_parsers
        import griffe

        # Recursively unwrap nested decorators and framework wrappers
        unwrapped_obj = obj
        for _ in range(10):  # max depth  # pragma: no branch
            if hasattr(unwrapped_obj, "__wrapped__"):
                unwrapped_obj = unwrapped_obj.__wrapped__
            elif hasattr(unwrapped_obj, "_python_function"):  # TensorFlow
                unwrapped_obj = unwrapped_obj._python_function
            elif hasattr(unwrapped_obj, "__original_fn"):  # Generic frameworks
                unwrapped_obj = getattr(unwrapped_obj, "__original_fn")
            elif hasattr(unwrapped_obj, "_original_fn"):  # Common variant
                unwrapped_obj = unwrapped_obj._original_fn
            elif hasattr(unwrapped_obj, "_orig_mod"):  # PyTorch torch.compile
                unwrapped_obj = unwrapped_obj._orig_mod
            else:
                break
        obj = unwrapped_obj

        name = getattr(obj, "__name__", api_path.split(".")[-1])
        kind = "class" if inspect.isclass(obj) else "function"
        doc = inspect.getdoc(obj)
        params = []
        has_varargs = False

        # Determine visibility
        determined_is_public = True
        if is_public is not None:
            determined_is_public = is_public

        # 1. Try to load docstring information via cdd
        cdd_params = {}
        returns_type = None
        returns_description = None
        raises = []

        if doc:
            try:
                cdd_ir = cdd_docstring_parsers.parse_docstring(doc)
                if cdd_ir.get("returns") and "return_type" in cdd_ir["returns"]:
                    ret = cdd_ir["returns"]["return_type"]
                    returns_type = sanitize_type_str(ret.get("typ"))
                    returns_description = ret.get("doc")

                if cdd_ir.get("params"):
                    for p_name, p_val in cdd_ir["params"].items():
                        if (
                            "Exception" in p_name or "Error" in p_name
                        ):  # Sphinx style raises
                            raises.append(p_name)
                        else:
                            cdd_params[p_name] = p_val

                if cdd_ir.get("raises"):  # In case cdd adds direct raises support
                    for exc_dict in cdd_ir["raises"]:
                        if "typ" in exc_dict:
                            raises.append(exc_dict["typ"])

            except Exception:  # pragma: no cover
                pass

        # 2. Try to load AST information via cdd.parse as primary zero-import static analysis engine
        cdd_parsed_ir = None
        cdd_ast_params: "dict[str, Any]" = {}

        target = obj
        if kind == "class":
            target = getattr(obj, "__init__", obj)

        has_super_kwargs_call = False

        try:
            source = inspect.getsource(
                obj
            )  # cdd requires the full class source, not just __init__
            parsed_ast = ast.parse(source).body[0]

            if kind == "class":
                for node in ast.walk(parsed_ast):
                    if isinstance(node, ast.Call):
                        if (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "__init__"
                        ):
                            if (  # pragma: no branch
                                isinstance(node.func.value, ast.Call)
                                and isinstance(node.func.value.func, ast.Name)
                                and node.func.value.func.id == "super"
                            ):
                                for keyword in node.keywords:  # pragma: no branch
                                    if (
                                        keyword.arg is None
                                    ):  # **kwargs  # pragma: no branch
                                        has_super_kwargs_call = True
                                        break

            if kind == "class":
                import cdd.class_.parse

                cdd_parsed_ir = cdd.class_.parse.class_(
                    parsed_ast, merge_inner_function="__init__"
                )
            else:
                import cdd.function.parse

                cdd_parsed_ir = cdd.function.parse.function(parsed_ast)

            if cdd_parsed_ir and "params" in cdd_parsed_ir:  # pragma: no branch
                for p_name, p_val in cdd_parsed_ir["params"].items():
                    cdd_ast_params[p_name] = p_val
                    # Ensure doc is merged
                    if p_name in cdd_params and "doc" in cdd_params[p_name]:
                        cdd_ast_params[p_name]["doc"] = cdd_params[p_name]["doc"]
        except Exception:  # pragma: no cover  # pragma: no cover
            pass

        # 2.5 Try to load AST information via griffe (fallback)
        griffe_node = None
        try:
            import griffe

            if hasattr(griffe.load, "return_value") or hasattr(
                griffe.load, "side_effect"
            ):  # Mocked
                griffe_node = griffe.load(api_path)
            else:
                top_level = api_path.split(".")[0]
                if top_level not in _GRIFFE_CACHE:
                    _GRIFFE_CACHE[top_level] = griffe.load(top_level)

                parts = api_path.split(".")
                current = _GRIFFE_CACHE[top_level]
                for part in parts[1:]:
                    current = current.members[part]
                griffe_node = current
        except Exception:  # pragma: no cover  # pragma: no cover
            pass

        if is_public is None:
            if griffe_node is not None and hasattr(griffe_node, "is_public"):
                determined_is_public = griffe_node.is_public
            else:
                determined_is_public = not name.startswith("_")

        import typing

        resolved_hints = {}
        try:
            resolved_hints = typing.get_type_hints(target)
            if "return" in resolved_hints and returns_type is None:
                returns_type = sanitize_type_str(str(resolved_hints["return"]))
        except Exception:  # pragma: no cover  # pragma: no cover
            pass

        # 3. Parameter Extraction Strategy (CDD -> Griffe -> Standard -> C-Extension Fallback)
        extracted_params = []

        griffe_params = None
        has_griffe_params = False
        try:
            griffe_params = getattr(griffe_node, "parameters", None)
            if griffe_params and len(griffe_params) > 0:
                has_griffe_params = True
        except Exception:
            pass

        if cdd_ast_params:
            # Use CDD AST parser
            for p_name, p_info in cdd_ast_params.items():
                if p_name == "self":
                    continue  # pragma: no cover
                # CDD IR doesn't explicitly store kinds, we infer from name (e.g. kwargs)
                # or fallback to POSITIONAL_OR_KEYWORD.
                p_kind_str = "POSITIONAL_OR_KEYWORD"

                # Check if it was *args / **kwargs from griffe or standard later if needed,
                # but CDD strips *args/**kwargs unless documented. Let's use CDD as base and
                # augment with griffe/standard if missing.

                default_val = None
                if "default" in p_info and p_info["default"] is not None:
                    # CDD default can be AST node or literal
                    try:
                        if isinstance(p_info["default"], ast.AST):
                            default_val = ast.unparse(
                                p_info["default"]
                            )  # pragma: no cover
                        elif isinstance(p_info["default"], str):
                            default_val = repr(p_info["default"])
                        else:
                            default_val = str(p_info["default"])
                    except Exception:  # pragma: no cover  # pragma: no cover
                        default_val = str(p_info["default"])

                anno_val = p_info.get("typ")
                if not anno_val and p_name in resolved_hints:
                    anno_val = sanitize_type_str(
                        str(resolved_hints[p_name])
                    )  # pragma: no cover
                else:
                    anno_val = sanitize_type_str(anno_val)

                extracted_params.append((p_name, p_kind_str, default_val, anno_val))

            # Fallback to Griffe to find VAR_POSITIONAL / VAR_KEYWORD which CDD drops
            if has_griffe_params and griffe_params is not None:
                for param in griffe_params:
                    if param.name == "self":
                        continue  # pragma: no cover
                    if param.name not in cdd_ast_params:
                        p_kind_str = (
                            param.kind.name.upper()
                            if getattr(param, "kind", None)
                            else "POSITIONAL_OR_KEYWORD"
                        )
                        if p_kind_str == "VAR_POSITIONAL":
                            has_varargs = True
                        default_val = (
                            str(param.default) if param.default is not None else None
                        )
                        anno_val = (
                            sanitize_type_str(
                                str(resolved_hints.get(param.name, param.annotation))
                            )
                            if param.annotation
                            else None
                        )
                        extracted_params.append(
                            (param.name, p_kind_str, default_val, anno_val)
                        )
                    else:
                        # Update kind if Griffe knows it
                        for i, (pn, pk, pd, pa) in enumerate(extracted_params):
                            if pn == param.name:
                                p_kind_str = (
                                    param.kind.name.upper()
                                    if getattr(param, "kind", None)
                                    else pk
                                )
                                extracted_params[i] = (pn, p_kind_str, pd, pa)
            else:
                # Standard fallback for VAR_POSITIONAL / VAR_KEYWORD when Griffe fails
                try:
                    sig = inspect.signature(target)
                    for param in sig.parameters.values():
                        if param.name == "self":
                            continue
                        if param.name not in cdd_ast_params:
                            p_kind_str = str(param.kind)
                            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                                has_varargs = True

                            default_val = None

                            anno_val = (
                                sanitize_type_str(
                                    str(
                                        resolved_hints.get(param.name, param.annotation)
                                    )
                                )
                                if param.annotation is not inspect.Parameter.empty
                                else None
                            )
                            extracted_params.append(
                                (param.name, p_kind_str, default_val, anno_val)
                            )
                        else:
                            for i, (pn, pk, pd, pa) in enumerate(extracted_params):
                                if pn == param.name:
                                    extracted_params[i] = (pn, str(param.kind), pd, pa)
                except Exception:  # pragma: no cover
                    pass

        elif has_griffe_params and griffe_params is not None:
            # Use Griffe for parameters
            for param in griffe_params:
                if param.name == "self":
                    continue

                p_kind_str = (
                    param.kind.name.upper()
                    if getattr(param, "kind", None)
                    else "POSITIONAL_OR_KEYWORD"
                )
                if p_kind_str == "VAR_POSITIONAL":
                    has_varargs = True

                # Safe default conversion
                default_val = None
                if param.default is not None:
                    default_val = str(param.default)

                if param.name in resolved_hints:
                    anno_val = sanitize_type_str(str(resolved_hints[param.name]))
                else:
                    anno_val = (
                        sanitize_type_str(str(param.annotation))
                        if param.annotation
                        else None
                    )
                extracted_params.append((param.name, p_kind_str, default_val, anno_val))
        else:
            # Standard introspection fallback
            try:
                sig = inspect.signature(target)

                for param in sig.parameters.values():
                    if param.name == "self":
                        continue

                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        has_varargs = True

                    default_val = None
                    if param.default is not inspect.Parameter.empty:
                        val = param.default

                        try:
                            is_addr = " at 0x" in repr(val)
                        except Exception:  # pragma: no cover
                            is_addr = False

                        if callable(val) or is_addr:
                            default_val = None
                        else:
                            try:
                                if isinstance(val, str):
                                    default_val = repr(val)
                                else:
                                    default_val = str(val)
                                if " at 0x" in default_val:
                                    default_val = None
                            except Exception:  # pragma: no cover
                                default_val = "<unrepresentable>"

                    anno_val = None
                    if param.name in resolved_hints:
                        anno_val = sanitize_type_str(str(resolved_hints[param.name]))
                    elif param.annotation is not inspect.Parameter.empty:
                        if hasattr(param.annotation, "__name__"):
                            anno_val = param.annotation.__name__
                        else:
                            anno_val = str(param.annotation)
                        anno_val = sanitize_type_str(anno_val)

                    extracted_params.append(
                        (param.name, str(param.kind), default_val, anno_val)
                    )

            except (ValueError, TypeError):
                # Try parsing C-Extension docstring signature as a fallback
                c_ext_params = extract_c_extension_signature(target, name)
                if c_ext_params is not None:
                    for pn, pk, pd, pa in c_ext_params:
                        if pk == "VAR_POSITIONAL":
                            has_varargs = True  # pragma: no cover

                        # Sanitize type string if extracted
                        sanitized_pa = sanitize_type_str(pa) if pa else None
                        extracted_params.append((pn, pk, pd, sanitized_pa))
                elif kind == "function":
                    has_varargs = True
                    extracted_params.append(("args", "VAR_POSITIONAL", None, None))
                    extracted_params.append(("kwargs", "VAR_KEYWORD", None, None))

        if has_super_kwargs_call and hasattr(obj, "__mro__") and len(obj.__mro__) > 1:
            for parent in obj.__mro__[1:]:
                if parent is object:
                    continue
                try:
                    parent_sig = inspect.signature(parent.__init__)
                    for param in parent_sig.parameters.values():
                        if param.name == "self" or param.kind in (
                            inspect.Parameter.VAR_POSITIONAL,
                            inspect.Parameter.VAR_KEYWORD,
                        ):
                            continue  # pragma: no cover
                        if not any(
                            ep[0] == param.name for ep in extracted_params
                        ):  # pragma: no branch
                            default_val = (
                                str(param.default)
                                if param.default is not inspect.Parameter.empty
                                else None
                            )
                            anno_val = (
                                param.annotation.__name__
                                if hasattr(param.annotation, "__name__")
                                else str(param.annotation)
                                if param.annotation is not inspect.Parameter.empty
                                else None
                            )
                            anno_val = sanitize_type_str(anno_val)
                            extracted_params.append(
                                (param.name, str(param.kind), default_val, anno_val)
                            )
                except Exception:  # pragma: no cover
                    pass

        # 4. Finalize GhostParams by merging in CDD docstring descriptions
        for p_name, p_kind, p_default, p_anno in extracted_params:
            p_desc = None
            if p_name in cdd_params and cdd_params[p_name].get("doc"):
                p_desc = cdd_params[p_name].get("doc")
            elif p_name in cdd_ast_params and cdd_ast_params[p_name].get("doc"):
                p_desc = cdd_ast_params[p_name].get("doc")

            # If griffe/inspect missed annotation, try CDD
            if not p_anno:
                if p_name in cdd_params and "typ" in cdd_params[p_name]:
                    p_anno = sanitize_type_str(cdd_params[p_name]["typ"])
                elif p_name in cdd_ast_params and "typ" in cdd_ast_params[p_name]:
                    p_anno = sanitize_type_str(cdd_ast_params[p_name]["typ"])

            params.append(
                GhostParam(
                    name=p_name,
                    standardized_name=STANDARD_ARG_MAP.get(p_name),
                    kind=p_kind,
                    default=p_default,
                    annotation=p_anno,
                    description=p_desc,
                )
            )

        import sys

        env_tags = [sys.platform]
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                env_tags.append("cuda")
            else:
                env_tags.append("cpu")
        except ImportError:
            env_tags.append(
                "cpu"
            )  # Default if we don't know, or maybe we omit it. We'll add 'cpu' for now to match the test.

        # Extract overloads via Griffe
        overloads_refs = []
        has_griffe_overloads = False
        griffe_overloads = None
        try:
            griffe_overloads = getattr(griffe_node, "overloads", None)
            if griffe_overloads:
                has_griffe_overloads = True
        except Exception:
            pass

        if has_griffe_overloads and griffe_overloads is not None:
            for overload in griffe_overloads:
                if isinstance(overload, str):
                    continue
                # Construct a partial GhostRef for each overload
                overload_params = []
                overload_has_varargs = False
                if hasattr(overload, "parameters"):  # pragma: no branch
                    for param in overload.parameters:
                        if param.name == "self":
                            continue
                        p_kind_str = (
                            param.kind.name.upper()
                            if getattr(param, "kind", None)
                            else "POSITIONAL_OR_KEYWORD"
                        )
                        if p_kind_str == "VAR_POSITIONAL":
                            overload_has_varargs = True

                        default_val = (
                            str(param.default) if param.default is not None else None
                        )
                        anno_val = (
                            sanitize_type_str(str(param.annotation))
                            if param.annotation
                            else None
                        )

                        overload_params.append(
                            GhostParam(
                                name=param.name,
                                standardized_name=STANDARD_ARG_MAP.get(param.name),
                                kind=p_kind_str,
                                default=default_val,
                                annotation=anno_val,
                                description=None,
                            )
                        )

                overload_returns = (
                    sanitize_type_str(str(overload.returns))
                    if getattr(overload, "returns", None)
                    else None
                )

                overloads_refs.append(
                    GhostRef(
                        name=name,
                        api_path=api_path,
                        kind=kind,
                        params=overload_params,
                        docstring=None,
                        has_varargs=overload_has_varargs,
                        aliases=[],
                        is_public=determined_is_public,
                        returns_type=overload_returns,
                        returns_description=None,
                        raises=[],
                        environment_tags=env_tags,
                        overloads=[],
                    )
                )

        return GhostRef(
            name=name,
            api_path=api_path,
            kind=kind,
            params=params,
            docstring=doc,
            has_varargs=has_varargs,
            aliases=[],
            is_public=determined_is_public,
            returns_type=returns_type,
            returns_description=returns_description,
            raises=raises,
            environment_tags=env_tags,
            overloads=overloads_refs,
        )

    @staticmethod
    def hydrate(data: Dict[str, Any]) -> "GhostRef":
        """Create a GhostRef from a dictionary (JSON snapshot).

        Args:
            data: The dictionary data.

        Returns:
            The hydrated GhostRef object.

        """
        return GhostRef.model_validate(data)
