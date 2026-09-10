"""Models for Ghost API representations.

Ghost Core: Introspection Abstraction Layer.

This module provides the data structures and inspection logic required to
decouple framework analysis from the live environment. It enables the system
to operate in "Ghost Mode" (WASM/CI) by working against cached snapshots
instead of requiring heavy libraries (Torch/TensorFlow) to be installed.

Updates:
- Robust C-Extension handling (try/except around `inspect.signature`).
- Validates parameter kinds to support `*args` (VarPositional).
- Sanitizes default values to avoid serializing memory addresses.
"""

import ast
import contextlib
from enum import Enum
import inspect
import io
import logging
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict
from ml_switcheroo_ir.schema.ghost import GhostParam as GhostParam, GhostRef as GhostRef

from .utils import (
    extract_c_extension_signature,
    get_framework_docstring_parser,
    resolve_griffe_parser,
    extract_griffe_docstring_metadata,
    strip_sphinx_roles,
)

STANDARD_ARG_MAP: Dict[str, str] = {
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

FRAMEWORK_CAPABILITIES: Dict[str, List[str]] = {
    "torch": ["cpu", "cuda", "rocm", "metal"],
    "tensorflow": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "tf": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "jax": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "mlx": ["cpu", "metal"],
    "triton": ["cuda", "rocm"],
    "cupy": ["cuda", "rocm"],
    "numpy": ["cpu"],
    "sklearn": ["cpu"],
    "scikit_learn": ["cpu"],
    "scipy": ["cpu"],
    "deepspeed": ["cpu", "cuda", "rocm"],
    "onnxruntime": ["cpu", "cuda", "rocm", "metal"],
    "nvidia_sass": ["cuda"],
    "amd_rdna": ["rocm"],
    "mlir": ["cpu", "cuda", "rocm", "tpu"],
    "stablehlo": ["cpu", "cuda", "rocm", "tpu"],
}

STANDARD_ENUM_MAP: Dict[str, List[str]] = {
    "reduction": ["none", "mean", "sum"],
    "mode": [
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
        "trilinear",
        "area",
        "nearest-exact",
    ],
    "padding": [
        "valid",
        "same",
        "zeros",
        "reflect",
        "replicate",
        "circular",
    ],
    "layout": ["NCHW", "NHWC", "NCDHW", "NDHWC"],
}


class OperandDirection(str, Enum):
    """Structured operand directionality for assembly and low-level instructions."""

    READ = "READ"
    WRITE = "WRITE"
    READ_WRITE = "READ_WRITE"
    PREDICATE = "PREDICATE"


class IRParameterRole(str, Enum):
    """Distinguishes parameter roles for compiler intermediate representations."""

    OPERAND = "OPERAND"
    ATTRIBUTE = "ATTRIBUTE"
    RESULT = "RESULT"
    SUCCESSOR = "SUCCESSOR"
    REGION = "REGION"


class GhostResult(BaseModel):
    """Structured SSA return or result for compiler IR operations."""

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = Field(
        default=None, description="Result SSA name or output identifier."
    )
    type: Optional[str] = Field(
        default=None, description="Result type (e.g. tensor<?x?xf32>)."
    )
    description: Optional[str] = Field(
        default=None, description="Description of the result."
    )


class ExtendedGhostParam(GhostParam):
    """Extended GhostParam supporting operand directionality, IR roles, dtypes, rank, and factory defaults."""

    model_config = ConfigDict(extra="allow")

    default: Optional[Union[str, Any]] = Field(
        default=None,
        description="Default value representation.",
    )
    direction: Optional[OperandDirection] = Field(
        default=None,
        description="Operand directionality (READ, WRITE, READ_WRITE, PREDICATE).",
    )
    role: Optional[IRParameterRole] = Field(
        default=None,
        description="IR parameter role (OPERAND, ATTRIBUTE, RESULT, etc.).",
    )
    dtypes: Optional[List[str]] = Field(
        default=None,
        description="Allowed tensor dtypes (e.g. ['float32', 'bfloat16', 'float16']).",
    )
    allowed_dtypes: Optional[List[str]] = Field(
        default=None,
        description="Canonical allowed tensor dtypes (e.g. ['float32', 'bfloat16']).",
    )
    allowed_values: Optional[List[str]] = Field(
        default=None,
        description="Allowed enum or literal string values (e.g. ['none', 'mean', 'sum']).",
    )
    rank: Optional[Union[int, str]] = Field(
        default=None,
        description="Allowed tensor rank (e.g. 0 for scalar, 1, 2, 'N-D').",
    )
    rank_constraint: Optional[str] = Field(
        default=None,
        description="Allowed tensor rank constraint (e.g. '==2', '>=2', 'scalar').",
    )
    is_contracting_dim: Optional[bool] = Field(
        default=None,
        description="Whether this parameter represents a contracting tensor dimension.",
    )
    default_factory: Optional[str] = Field(
        default=None,
        description="Name or representation of factory function producing default value.",
    )
    is_mandatory: Optional[bool] = Field(
        default=None,
        description="Whether parameter is mandatory (no default value).",
    )


def sanitize_param_default(
    val: Any,
) -> Tuple[Optional[str], Optional[str], bool]:
    """Sanitize parameter default value while preserving semantic literals and scrubbing memory addresses.

    Args:
        val: Raw default value from inspect.Parameter or AST.

    Returns:
        Tuple of (default_value_string, default_factory_string, is_mandatory_bool).
    """
    if val is inspect.Parameter.empty:
        return (None, None, True)
    if val is Ellipsis:
        return ("...", None, False)
    if val is None:
        return ("None", None, False)
    if isinstance(val, bool):
        return ("True" if val else "False", None, False)
    if isinstance(val, (int, float)):
        return (str(val), None, False)
    if isinstance(val, str):
        return (repr(val), None, False)

    try:
        if callable(val):
            val_repr = repr(val)
            if val_repr.startswith("<class '") and val_repr.endswith("'>"):
                cls_name = val_repr[8:-2]
                return (cls_name, None, False)
            func_name = getattr(val, "__name__", None)
            factory_str = (
                f"<function {func_name}>" if func_name else "<factory_default>"
            )
            return ("<factory_default>", factory_str, False)

        val_repr = repr(val)
        if re.match(r"^<dtype:\s*'([^']+)'\s*>$", val_repr):
            m_dt = re.match(r"^<dtype:\s*'([^']+)'\s*>$", val_repr)
            return (f"tf.{m_dt.group(1)}" if m_dt else val_repr, None, False)

        if "device(type=" in val_repr:
            m_dev = re.search(r"type='([^']+)'", val_repr)
            dev_type = m_dev.group(1) if m_dev else "cpu"
            return (repr(dev_type), val_repr, False)

        if " at 0x" in val_repr or re.search(r"\b0x[0-9a-fA-F]{4,}\b", val_repr):
            scrubbed = re.sub(r"\s*at\s*0x[0-9a-fA-F]+", "", val_repr)
            scrubbed = re.sub(r"\b0x[0-9a-fA-F]{4,}\b", "<addr>", scrubbed)
            if scrubbed.startswith("<") and scrubbed.endswith(">"):
                return ("<factory_default>", scrubbed, False)
            return (scrubbed, None, False)

        val_str = str(val)
        if " at 0x" in val_str or re.search(r"\b0x[0-9a-fA-F]{4,}\b", val_str):
            return ("<factory_default>", "<factory_default>", False)
        return (val_str, None, False)
    except Exception:
        return ("<unrepresentable>", None, False)


class ExtendedGhostRef(GhostRef):
    """Extended GhostRef with support for domain metadata, multiple SSA returns, and IR operands."""

    model_config = ConfigDict(extra="allow")

    params: List[Union[ExtendedGhostParam, GhostParam]] = Field(
        default_factory=list,
        description="List of extended parameter specifications.",
    )
    returns: Optional[List[GhostResult]] = Field(
        default=None, description="Multiple SSA returns or results."
    )
    domain_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured domain metadata for ISAs and compilers.",
    )
    signature_completeness: Optional[Literal["exact", "heuristic", "opaque"]] = Field(
        default="exact",
        description="Completeness of signature resolution: exact, heuristic, or opaque.",
    )
    is_c_extension: Optional[bool] = Field(
        default=False,
        description="Whether the symbol originates from a compiled C/C++ extension.",
    )
    accepted_kwargs: Optional[List[str]] = Field(
        default=None,
        description="Explicit list of accepted keyword arguments when **kwargs is present.",
    )


class SnapshotEnvelope(BaseModel):
    """Structured provenance envelope for framework and ISA/IR snapshots."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(default="1.0.0", description="Snapshot schema version.")
    target: str = Field(..., description="Target framework, dialect, or hardware ISA.")
    version: Optional[str] = Field(
        default=None,
        description="Upstream framework version or toolkit release.",
    )
    upstream_version: Optional[str] = Field(
        default=None,
        description="Upstream hardware specification or compiler version.",
    )
    source_type: Optional[str] = Field(
        default=None,
        description="Extraction source (tablegen, binary_disassembly, python_ast).",
    )
    upstream_commit: Optional[str] = Field(
        default=None, description="Upstream git commit hash or release tag."
    )
    supported_microarchitectures: Optional[List[str]] = Field(
        default=None,
        description="Explicit list of supported GPU compute capabilities or target architectures.",
    )
    generated_at: Optional[str] = Field(
        default=None, description="ISO-8601 generation timestamp."
    )
    environment: Optional[Dict[str, Any]] = Field(
        default=None, description="Build host environment metadata."
    )
    categories: Dict[str, List[Any]] = Field(
        default_factory=dict, description="Categorized symbol dictionaries."
    )


class GhostPythonRef(ExtendedGhostRef):
    """GhostRef specialized for high-level Python ML frameworks (PyTorch, JAX, TF, Keras)."""

    model_config = ConfigDict(extra="allow")
    domain_type: Literal["python"] = "python"


class GhostIsaRef(ExtendedGhostRef):
    """GhostRef specialized for GPU assembly ISAs (NVIDIA SASS, AMD RDNA/CDNA)."""

    model_config = ConfigDict(extra="allow")
    domain_type: Literal["isa"] = "isa"
    predicate_guards: Optional[List[str]] = Field(
        default=None,
        description="Allowed predicate guard registers (e.g. ['@P0', '@!P1', '@PT']).",
    )
    register_classes: Optional[Dict[str, str]] = Field(
        default=None,
        description="Register classes for operands (e.g. {'op0': 'VGPR_32', 'op1': 'VReg_64'}).",
    )
    control_codes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Instruction control code and scheduling schema.",
    )
    instruction_modifiers: Optional[List[str]] = Field(
        default=None,
        description="Valid instruction modifiers (e.g. ['.SAT', '.FTZ', 'omod:2']).",
    )
    structured_modifiers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured modifier bitfields (e.g. rounding, cache, saturation).",
    )
    structured_operands: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Detailed operand records with roles, register classes, and immediate constraints.",
    )
    vopd_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="VOPD dual-issue profile and pairing rules for RDNA3/GFX11.",
    )
    supported_architectures: Optional[List[str]] = Field(
        default=None,
        description="Microarchitectures supporting this instruction.",
    )


class GhostMlirRef(ExtendedGhostRef):
    """GhostRef specialized for compiler IR dialects (Core MLIR and StableHLO)."""

    model_config = ConfigDict(extra="allow")
    domain_type: Literal["mlir"] = "mlir"
    traits: Optional[List[str]] = Field(
        default=None,
        description="Dialect verification traits (e.g. ['SameOperandsAndResultType', 'Commutative']).",
    )
    operands: Optional[List[Union[ExtendedGhostParam, GhostParam]]] = Field(
        default=None,
        description="Strictly decoupled SSA value arguments (operands).",
    )
    attributes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured attribute specifications and schemas.",
    )
    regions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Region definitions with block arguments and yield types.",
    )
    successors: Optional[List[str]] = Field(
        default=None,
        description="Successor block identifiers for control flow operations.",
    )
    type_constraints: Optional[Dict[str, str]] = Field(
        default=None,
        description="Type constraints for operands and results (e.g. RankedTensorOf, AnyFloat).",
    )


_GRIFFE_CACHE: Dict[str, Any] = {}

logging.getLogger("griffe").setLevel(logging.CRITICAL)


def preload_griffe_cache(frameworks: Optional[List[str]] = None) -> None:
    """Preload Griffe AST for specified frameworks into _GRIFFE_CACHE.

    Args:
        frameworks: Optional list of package names to preload. If None,
            preloads common ML packages if installed.
    """
    import griffe

    targets = frameworks or ["torch", "jax", "tensorflow", "keras", "mlx", "scipy"]
    for target in targets:
        if target not in _GRIFFE_CACHE:
            try:
                parser = resolve_griffe_parser(get_framework_docstring_parser(target))
                _GRIFFE_CACHE[target] = griffe.load(target, docstring_parser=parser)
            except Exception:
                pass


def sanitize_type_str(typ_str: Optional[str]) -> Optional[str]:
    """Sanitize type hints to PEP-585/PEP-604 representations.

    Args:
        typ_str: description

    Returns:
        The sanitized type string.
    """
    if not typ_str:
        return typ_str

    # Clean Sphinx roles like :class:`~torch.Tensor` -> torch.Tensor
    typ_str = strip_sphinx_roles(typ_str)
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
        environment_tags: Optional[List[str]] = None,
        kind: Optional[str] = None,
    ) -> GhostRef:
        """Create a GhostRef from a live Python object.

        Gracefully handles C-Extensions and builtins that resist introspection.

        Args:
            obj: The live class or function to inspect.
            api_path: The canonical string path (e.g. 'torch.nn.ReLU').
            is_public: Optional override for public visibility.
            environment_tags: Optional explicit execution tags (e.g. ['cpu', 'cuda']).
            kind: Optional explicit kind override (e.g. 'function', 'class', 'method').

        Returns:
            A populated GhostRef object.

        """
        import cdd.docstring.parse as cdd_docstring_parsers
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

        is_griffe_node = hasattr(obj, "is_class") and hasattr(obj, "is_function")
        if is_griffe_node:
            name = getattr(obj, "name", api_path.split(".")[-1])
            determined_kind = (
                kind
                if kind is not None
                else ("class" if getattr(obj, "is_class", False) else "function")
            )
            doc_obj = getattr(obj, "docstring", None)
            doc = (
                str(doc_obj.value)
                if doc_obj and hasattr(doc_obj, "value")
                else (str(doc_obj) if doc_obj else None)
            )
            griffe_node = obj
        else:
            name = getattr(obj, "__name__", api_path.split(".")[-1])
            if name == "<lambda>":
                name = api_path.split(".")[-1]
            determined_kind = (
                kind
                if kind is not None
                else ("class" if inspect.isclass(obj) else "function")
            )
            doc = inspect.getdoc(obj)
            griffe_node = None
        kind = determined_kind
        params = []
        has_varargs = False
        is_c_ext = False
        sig_completeness: Literal["exact", "heuristic", "opaque"] = "exact"

        # Determine visibility
        determined_is_public = True
        if is_public is not None:
            determined_is_public = is_public

        # 1. Try to load docstring information via cdd and griffe reST/Sphinx enumeration
        cdd_params = {}
        returns_type = None
        returns_description = None
        raises = []

        top_level = api_path.split(".")[0] if api_path else ""
        docstring_parser_name = get_framework_docstring_parser(top_level, doc)

        if doc:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    cdd_ir = cdd_docstring_parsers.docstring(doc)
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

            # Supplement with Griffe's structured docstring enumeration
            try:
                griffe_meta = extract_griffe_docstring_metadata(
                    doc, parser_name=docstring_parser_name
                )
                if griffe_meta.get("returns"):
                    ret_meta = griffe_meta["returns"]
                    if ret_meta.get("typ") and not returns_type:
                        returns_type = sanitize_type_str(ret_meta["typ"])
                    if ret_meta.get("doc") and not returns_description:
                        returns_description = ret_meta["doc"]

                for exc in griffe_meta.get("raises", []):
                    if exc not in raises:
                        raises.append(exc)
                    # If CDD misclassified an exception into cdd_params, remove it
                    if exc in cdd_params:
                        del cdd_params[exc]

                for p_name, p_data in griffe_meta.get("params", {}).items():
                    if p_name not in cdd_params:
                        cdd_params[p_name] = p_data
                    else:
                        if p_data.get("doc") and not cdd_params[p_name].get("doc"):
                            cdd_params[p_name]["doc"] = p_data["doc"]
                        if p_data.get("typ") and not cdd_params[p_name].get("typ"):
                            cdd_params[p_name]["typ"] = p_data["typ"]
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

                with contextlib.redirect_stderr(io.StringIO()):
                    cdd_parsed_ir = cdd.class_.parse.class_(
                        parsed_ast, merge_inner_function="__init__"
                    )
            else:
                import cdd.function.parse

                with contextlib.redirect_stderr(io.StringIO()):
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
        if griffe_node is None:
            try:
                import griffe

                resolved_parser = resolve_griffe_parser(docstring_parser_name)
                if hasattr(griffe.load, "return_value") or hasattr(
                    griffe.load, "side_effect"
                ):  # Mocked
                    griffe_node = griffe.load(api_path)
                else:
                    top_level = api_path.split(".")[0]
                    if top_level not in _GRIFFE_CACHE:
                        _GRIFFE_CACHE[top_level] = griffe.load(
                            top_level, docstring_parser=resolved_parser
                        )

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
        if not is_griffe_node:
            try:
                resolved_hints = typing.get_type_hints(target)
                if "return" in resolved_hints and returns_type is None:
                    returns_type = sanitize_type_str(str(resolved_hints["return"]))
            except Exception:  # pragma: no cover  # pragma: no cover
                pass

        if griffe_node is not None and returns_type is None:
            try:
                g_ret = getattr(griffe_node, "returns", None)
                if g_ret:
                    returns_type = sanitize_type_str(str(g_ret))
            except Exception:
                pass

        # 3. Parameter Extraction Strategy (CDD -> Griffe -> Standard -> C-Extension Fallback)
        extracted_params = []
        c_ext_params = None

        griffe_params = None
        has_griffe_params = False
        try:
            if griffe_node is not None:
                griffe_params = getattr(griffe_node, "parameters", None)
                if griffe_params is None and getattr(griffe_node, "is_class", False):
                    members = getattr(griffe_node, "members", {})
                    init_node = members.get("__init__") or members.get("__new__")
                    if init_node:
                        griffe_params = getattr(init_node, "parameters", None)
                if griffe_params and len(griffe_params) > 0:
                    has_griffe_params = True
        except Exception:  # pragma: no cover
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

                    default_val, _factory, _is_mand = sanitize_param_default(
                        param.default
                    )

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

                if (
                    inspect.isbuiltin(obj)
                    or getattr(target, "__module__", "").startswith("_")
                    or "torch._C" in str(target)
                    or type(target).__name__ == "builtin_function_or_method"
                ):
                    is_c_ext = True

            except (ValueError, TypeError):
                is_c_ext = True
                # Try parsing C-Extension docstring signature as a fallback
                c_ext_params = extract_c_extension_signature(target, name)

                # ATen & native function introspection for PyTorch
                is_torch_target = (
                    (name and name.startswith("torch."))
                    or (getattr(target, "__module__", "") or "").startswith("torch")
                    or api_path.startswith("torch.")
                )
                is_exact_aten = False
                if is_torch_target:
                    try:
                        from .frameworks.torch import (
                            extract_aten_c_extension_signature,
                        )

                        aten_sig = extract_aten_c_extension_signature(
                            target,
                            name,
                            is_method=(
                                kind == "method"
                                or (name is not None and ".Tensor." in name)
                            ),
                        )
                        if aten_sig is not None:
                            if c_ext_params is None:
                                c_ext_params = aten_sig
                                is_exact_aten = True
                            elif (
                                not getattr(c_ext_params, "overloads", None)
                                and aten_sig.overloads
                            ):
                                c_ext_params.overloads = aten_sig.overloads
                                is_exact_aten = True
                    except Exception:  # pragma: no cover
                        pass

                if c_ext_params is not None:
                    sig_completeness = "exact" if is_exact_aten else "heuristic"
                    if (returns_type is None or returns_type == "NoneType") and getattr(
                        c_ext_params, "returns_type", None
                    ):
                        returns_type = sanitize_type_str(c_ext_params.returns_type)
                    for pn, pk, pd, pa in c_ext_params:
                        if pk == "VAR_POSITIONAL":
                            has_varargs = True  # pragma: no cover

                        # Sanitize type string if extracted
                        sanitized_pa = sanitize_type_str(pa) if pa else None
                        extracted_params.append((pn, pk, pd, sanitized_pa))
                elif kind == "function":
                    sig_completeness = "opaque"
                    has_varargs = True
                    extracted_params.append(("args", "VAR_POSITIONAL", None, None))
                    extracted_params.append(("kwargs", "VAR_KEYWORD", None, None))
                    if environment_tags is None:
                        environment_tags = []
                    if "inexact_signature" not in environment_tags:
                        environment_tags.append("inexact_signature")
                    if "opaque_c_extension" not in environment_tags:
                        environment_tags.append("opaque_c_extension")

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
                                else (
                                    str(param.annotation)
                                    if param.annotation is not inspect.Parameter.empty
                                    else None
                                )
                            )
                            anno_val = sanitize_type_str(anno_val)
                            extracted_params.append(
                                (param.name, str(param.kind), default_val, anno_val)
                            )
                except Exception:  # pragma: no cover
                    pass

        # 3.5 Promote documented **kwargs into formal KEYWORD_ONLY GhostParams
        has_var_kw = any(ep[1] == "VAR_KEYWORD" for ep in extracted_params)
        if has_var_kw and cdd_params:
            existing_names = {ep[0] for ep in extracted_params}
            var_kw_idx = next(
                i for i, ep in enumerate(extracted_params) if ep[1] == "VAR_KEYWORD"
            )
            promoted_params = []
            for p_name, p_data in cdd_params.items():
                if (
                    p_name not in existing_names
                    and p_name not in ("kwargs", "args", "self")
                    and not p_name.startswith("_")
                ):
                    p_typ = (
                        sanitize_type_str(p_data.get("typ"))
                        if p_data.get("typ")
                        else None
                    )
                    p_def = p_data.get("default")
                    promoted_params.append((p_name, "KEYWORD_ONLY", p_def, p_typ))
                    existing_names.add(p_name)

            if promoted_params:
                extracted_params = (
                    extracted_params[:var_kw_idx]
                    + promoted_params
                    + extracted_params[var_kw_idx:]
                )

        # 4. Finalize GhostParams by merging in CDD docstring descriptions
        is_torch_target = (
            (name and name.startswith("torch."))
            or (getattr(target, "__module__", "") or "").startswith("torch")
            or api_path.startswith("torch.")
        )

        for p_name, p_kind, p_default, p_anno in extracted_params:
            p_desc = None
            if p_name in cdd_params and cdd_params[p_name].get("doc"):
                p_desc = cdd_params[p_name].get("doc")
            elif p_name in cdd_ast_params and cdd_ast_params[p_name].get("doc"):
                p_desc = cdd_ast_params[p_name].get("doc")

            # If griffe/inspect missed annotation or gave Any, try CDD / docstring
            if not p_anno or p_anno == "Any":
                if p_name in cdd_params and cdd_params[p_name].get("typ"):
                    p_anno = sanitize_type_str(cdd_params[p_name]["typ"])
                elif p_name in cdd_ast_params and cdd_ast_params[p_name].get("typ"):
                    p_anno = sanitize_type_str(cdd_ast_params[p_name]["typ"])

            p_dtypes, p_rank = None, None
            if is_torch_target:
                try:
                    from .frameworks.torch import infer_torch_dtype_and_rank

                    p_dtypes, p_rank = infer_torch_dtype_and_rank(
                        name, p_name, p_anno or ""
                    )
                except Exception:  # pragma: no cover
                    pass

            p_factory = None
            if p_default and "<factory_default>" in str(p_default):
                p_factory = "<factory_default>"

            p_allowed_values = None
            if p_anno and "Literal[" in p_anno:
                p_allowed_values = re.findall(r"['\"]([^'\"]+)['\"]", p_anno)
            elif p_name.lower() in STANDARD_ENUM_MAP:
                p_allowed_values = STANDARD_ENUM_MAP[p_name.lower()]

            params.append(
                ExtendedGhostParam(
                    name=p_name,
                    standardized_name=STANDARD_ARG_MAP.get(p_name),
                    kind=p_kind,
                    default=p_default,
                    annotation=p_anno,
                    description=p_desc,
                    dtypes=p_dtypes,
                    rank=p_rank,
                    allowed_values=p_allowed_values,
                    default_factory=p_factory,
                    is_mandatory=p_default is None,
                )
            )

        if environment_tags is not None:
            env_tags = list(environment_tags)
        else:
            top_pkg = api_path.split(".")[0].lower() if api_path else ""
            env_tags = list(FRAMEWORK_CAPABILITIES.get(top_pkg, ["cpu"]))

        # Extract overloads via Griffe
        overloads_refs = []
        has_griffe_overloads = False
        griffe_overloads = None
        try:
            griffe_overloads = getattr(griffe_node, "overloads", None)
            if not griffe_overloads and getattr(griffe_node, "is_class", False):
                init_node = getattr(griffe_node, "members", {}).get("__init__")
                if init_node:
                    griffe_overloads = getattr(init_node, "overloads", None)
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

                        ov_allowed_values = None
                        if anno_val and "Literal[" in anno_val:
                            ov_allowed_values = re.findall(
                                r"['\"]([^'\"]+)['\"]", anno_val
                            )
                        elif param.name.lower() in STANDARD_ENUM_MAP:
                            ov_allowed_values = STANDARD_ENUM_MAP[param.name.lower()]

                        overload_params.append(
                            ExtendedGhostParam(
                                name=param.name,
                                standardized_name=STANDARD_ARG_MAP.get(param.name),
                                kind=p_kind_str,
                                default=default_val,
                                annotation=anno_val,
                                description=None,
                                allowed_values=ov_allowed_values,
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
        elif c_ext_params is not None and getattr(c_ext_params, "overloads", None):
            for ov in c_ext_params.overloads:
                ov_params = []
                ov_has_varargs = False
                for pn, pk, pd, pa in ov:
                    if pk == "VAR_POSITIONAL":
                        ov_has_varargs = True
                    sanitized_pa = sanitize_type_str(pa) if pa else None

                    ov_dtypes, ov_rank = None, None
                    if is_torch_target:
                        try:
                            from .frameworks.torch import infer_torch_dtype_and_rank

                            ov_dtypes, ov_rank = infer_torch_dtype_and_rank(
                                name, pn, sanitized_pa or ""
                            )
                        except Exception:  # pragma: no cover
                            pass

                    ov_factory = None
                    if pd and "<factory_default>" in str(pd):
                        ov_factory = "<factory_default>"

                    ov_allowed_values = None
                    if sanitized_pa and "Literal[" in sanitized_pa:
                        ov_allowed_values = re.findall(
                            r"['\"]([^'\"]+)['\"]", sanitized_pa
                        )
                    elif pn.lower() in STANDARD_ENUM_MAP:
                        ov_allowed_values = STANDARD_ENUM_MAP[pn.lower()]

                    ov_params.append(
                        ExtendedGhostParam(
                            name=pn,
                            standardized_name=STANDARD_ARG_MAP.get(pn),
                            kind=pk,
                            default=pd,
                            annotation=sanitized_pa,
                            description=None,
                            dtypes=ov_dtypes,
                            rank=ov_rank,
                            allowed_values=ov_allowed_values,
                            default_factory=ov_factory,
                            is_mandatory=pd is None,
                        )
                    )
                ov_ret = (
                    sanitize_type_str(ov.returns_type)
                    if getattr(ov, "returns_type", None)
                    else None
                )
                overloads_refs.append(
                    GhostRef(
                        name=name,
                        api_path=api_path,
                        kind=kind,
                        params=ov_params,
                        docstring=None,
                        has_varargs=ov_has_varargs,
                        aliases=[],
                        is_public=determined_is_public,
                        returns_type=ov_ret,
                        returns_description=None,
                        raises=[],
                        environment_tags=env_tags,
                        overloads=[],
                    )
                )

        return GhostPythonRef(
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
            signature_completeness=sig_completeness,
            is_c_extension=is_c_ext,
        )

    @staticmethod
    def hydrate(data: Dict[str, Any]) -> ExtendedGhostRef:
        """Create a GhostRef from a dictionary (JSON snapshot).

        Args:
            data: The dictionary data.

        Returns:
            The hydrated ExtendedGhostRef or specialized polymorphic GhostRef object.
        """
        # Synthesize domain_type and required fields for raw dumps if missing
        if "mnemonic" in data and ("name" not in data or "operands" in data):
            name = str(data.get("name") or data["mnemonic"])
            desc = str(data.get("description") or f"ISA {name} instruction.")
            arch = data.get("architecture")
            valid_archs: List[str] = []
            if isinstance(arch, list):
                valid_archs = [str(a) for a in arch]
            elif isinstance(arch, str):
                valid_archs = [arch]
            modifiers = [str(m) for m in data.get("modifiers", [])]
            operands_list: List[List[str]] = data.get("operands", [])
            max_sig: List[str] = []
            for sig in operands_list:
                if len(sig) > len(max_sig):
                    max_sig = sig
            params: List[ExtendedGhostParam] = []
            for i, raw_op in enumerate(max_sig):
                role = "dst" if i == 0 else f"src{i - 1}"
                direction = OperandDirection.WRITE if i == 0 else OperandDirection.READ
                params.append(
                    ExtendedGhostParam(
                        name=f"op{i}",
                        kind="POSITIONAL_ONLY",
                        annotation=str(raw_op),
                        standardized_name=role,
                        direction=direction,
                        role=IRParameterRole.OPERAND,
                    )
                )
            domain_metadata = dict(data.get("domain_metadata") or {})
            domain_metadata.setdefault("architecture", arch)
            domain_metadata.setdefault("valid_architectures", valid_archs)
            domain_metadata.setdefault("modifiers", modifiers)
            domain_metadata.setdefault("operand_signatures", operands_list)
            isa_overloads: List[ExtendedGhostRef] = []
            for sig_idx, sig in enumerate(operands_list):
                sig_params: List[ExtendedGhostParam] = []
                for idx, op_type in enumerate(sig):
                    role = "dst" if idx == 0 else f"src{idx - 1}"
                    direction = (
                        OperandDirection.WRITE if idx == 0 else OperandDirection.READ
                    )
                    sig_params.append(
                        ExtendedGhostParam(
                            name=f"op{idx}",
                            kind="POSITIONAL_ONLY",
                            annotation=str(op_type),
                            standardized_name=role,
                            direction=direction,
                            role=IRParameterRole.OPERAND,
                        )
                    )
                isa_overloads.append(
                    ExtendedGhostRef(
                        name=name,
                        api_path=str(data.get("api_path") or f"isa.inst.{name}"),
                        kind="function",
                        params=sig_params,
                        docstring=f"Overload {sig_idx} for {name}: {', '.join(str(s) for s in sig)}",
                        environment_tags=valid_archs,
                        domain_metadata=domain_metadata,
                    )
                )
            return GhostIsaRef(
                name=name,
                api_path=str(data.get("api_path") or f"isa.inst.{name}"),
                kind=str(data.get("kind") or "function"),
                params=params,
                docstring=desc,
                environment_tags=valid_archs,
                overloads=isa_overloads if isa_overloads else None,
                domain_metadata=domain_metadata,
            )

        if (
            ("attributes" in data or "results" in data)
            and ("name" not in data or "params" not in data)
            and data.get("domain_type") != "mlir"
        ):
            api_path = str(data.get("api_path") or "mlir.op")
            name = str(data.get("name") or api_path.split(".")[-1])
            params = []
            for op in data.get("operands", []):
                op_name = op if isinstance(op, str) else str(op.get("name", "arg"))
                params.append(
                    ExtendedGhostParam(
                        name=op_name,
                        kind="POSITIONAL_ONLY",
                        role=IRParameterRole.OPERAND,
                    )
                )
            for attr in data.get("attributes", []):
                attr_name = (
                    attr if isinstance(attr, str) else str(attr.get("name", "attr"))
                )
                params.append(
                    ExtendedGhostParam(
                        name=attr_name,
                        kind="KEYWORD_ONLY",
                        role=IRParameterRole.ATTRIBUTE,
                    )
                )
            results = []
            for res in data.get("results", []):
                results.append(
                    GhostResult(
                        name=(
                            res if isinstance(res, str) else str(res.get("name", "res"))
                        ),
                        type=(
                            str(res.get("type", "Value"))
                            if isinstance(res, dict)
                            else "Value"
                        ),
                    )
                )
            raw_traits = data.get("traits")
            raw_regions = data.get("regions")
            raw_successors = data.get("successors")
            raw_attrs = data.get("attributes")
            return GhostMlirRef(
                name=name,
                api_path=api_path,
                kind=str(data.get("kind") or "operation"),
                params=params,
                returns=results,
                docstring=data.get("docstring") or data.get("description"),
                traits=raw_traits if isinstance(raw_traits, list) else None,
                regions=raw_regions if isinstance(raw_regions, dict) else None,
                successors=raw_successors if isinstance(raw_successors, list) else None,
                domain_metadata=data.get("domain_metadata"),
                attributes=raw_attrs if isinstance(raw_attrs, dict) else None,
            )

        domain_type = data.get("domain_type")
        ref_cls: Any = ExtendedGhostRef
        if domain_type == "python":
            ref_cls = GhostPythonRef
        elif domain_type == "isa":
            ref_cls = GhostIsaRef
        elif domain_type == "mlir":
            ref_cls = GhostMlirRef

        res = ref_cls.model_validate(data)
        assert isinstance(res, ExtendedGhostRef)
        return res
