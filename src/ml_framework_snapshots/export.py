"""Export module for generating JSON Schema, OpenAPI specifications, Pydantic classes, and Protobuf definitions."""

from typing import Any, Dict, List, Optional, Set
from collections import OrderedDict

from ml_switcheroo_ir.schema.ghost import GhostRef


def _ghost_to_cdd_ir(ref: GhostRef) -> Dict[str, Any]:
    """Convert a GhostRef to cdd-python Intermediate Representation (IR).

    Args:
        ref: description

    Returns:
        The cdd IR representation.
    """
    ir: "dict[str, Any]" = {
        "name": ref.name,
        "type": "class" if ref.kind == "class" else "static",
        "doc": ref.docstring or "",
        "params": OrderedDict(),
        "returns": None,
    }

    for param in ref.params:
        p_dict = {}
        if param.annotation:
            p_dict["typ"] = param.annotation
        else:  # pragma: no cover
            pass
        if param.description:
            p_dict["doc"] = param.description
        else:  # pragma: no cover
            pass
        if param.default is not None:
            p_dict["default"] = param.default
        else:  # pragma: no cover
            pass
        ir["params"][param.name] = p_dict

    if ref.returns_type or ref.returns_description:
        ir["returns"] = OrderedDict()
        ret_dict = {}
        if ref.returns_type:
            ret_dict["typ"] = ref.returns_type
        if ref.returns_description:
            ret_dict["doc"] = ref.returns_description
        ir["returns"]["return_type"] = ret_dict

    return ir


def to_json_schema(ref: GhostRef) -> Dict[str, Any]:
    """Convert a GhostRef into a JSON Schema.

    Args:
        ref: The GhostRef to convert.

    Returns:
        A dictionary representing the JSON Schema.

    """
    import cdd.json_schema.emit
    from typing import cast

    ir = _ghost_to_cdd_ir(ref)
    return cast(
        Dict[str, Any], cdd.json_schema.emit.json_schema(ir, identifier=ref.api_path)
    )


def to_openapi(refs: List[GhostRef]) -> Dict[str, Any]:
    """Convert a list of GhostRefs into an OpenAPI specification.

    Args:
        refs: A list of GhostRefs.

    Returns:
        A dictionary representing the OpenAPI specification.

    """
    import cdd.compound.openapi.emit
    from cdd.compound.openapi.utils.emit_openapi_utils import NameModelRouteIdCrud

    nm_cruds = []
    for ref in refs:
        ir = _ghost_to_cdd_ir(ref)
        # Use api_path for route to make it unique and descriptive
        route = f"/{ref.api_path.replace('.', '/')}"

        nm_cruds.append(
            NameModelRouteIdCrud(
                name=ref.name, model=ir, route=route, id=None, crud="CRU"
            )
        )

    from typing import cast

    return cast(Dict[str, Any], cdd.compound.openapi.emit.openapi(nm_cruds))


def to_pydantic(ref: GhostRef, validate_varargs: bool = False) -> str:
    """Convert a GhostRef into a Pydantic V2 class definition string.

    Supports overloaded signatures by generating a typing.Union of Pydantic models,
    variable positional arguments as validated Tuple fields, and parameter descriptions
    in Field(description=...) for OpenAPI/schema generation.

    Args:
        ref: The GhostRef to convert.
        validate_varargs: Whether to represent *args as validated Tuple fields instead of skipping.

    Returns:
        A string containing the Python source for the Pydantic model.
    """
    header_lines = [
        "from pydantic import BaseModel, Field",
        "from typing import Any, Optional, Union, List, Dict, Tuple",
        "",
        "",
    ]

    def _render_single_model(
        name: str, doc: Optional[str], params: List[Any]
    ) -> List[str]:
        """Render Python source code lines for a single Pydantic model.

        Args:
            name: Class name for the Pydantic model.
            doc: Optional docstring for the class.
            params: List of GhostParam specifications.

        Returns:
            List of Python source code lines for the class.
        """
        m_lines = [f"class {name}(BaseModel):"]
        if doc:
            m_lines.append(f'    """{doc}"""')
        else:
            m_lines.append('    """Generated Pydantic model."""')

        valid_params = []
        for p in params:
            if p.kind in ("VAR_POSITIONAL", "VAR_KEYWORD") and not validate_varargs:
                continue
            valid_params.append(p)

        if not valid_params:
            m_lines.append("    pass")
            return m_lines

        for param in valid_params:
            typ = param.annotation if param.annotation else "Any"
            desc = param.description.replace('"', "'") if param.description else ""
            default_val = param.default

            if param.kind == "VAR_POSITIONAL":
                typ = f"Tuple[{typ}, ...]" if typ != "Any" else "Tuple[Any, ...]"
                field_def = (
                    f'Field(default=(), description="{desc}")'
                    if desc
                    else "Field(default=())"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")
                continue
            if param.kind == "VAR_KEYWORD":
                typ = f"Dict[str, {typ}]" if typ != "Any" else "Dict[str, Any]"
                field_def = (
                    f'Field(default_factory=dict, description="{desc}")'
                    if desc
                    else "Field(default_factory=dict)"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")
                continue

            if default_val is None:
                field_def = (
                    f'Field(..., description="{desc}")' if desc else "Field(...)"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")
            else:
                field_def = (
                    f'Field(default={default_val}, description="{desc}")'
                    if desc
                    else f"{default_val}"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")

        return m_lines

    all_models: List[str] = []
    if ref.overloads:
        all_variants = [ref] + [ov for ov in ref.overloads if isinstance(ov, GhostRef)]
        variant_names = []
        for i, variant in enumerate(all_variants):
            var_name = f"{ref.name}Variant{i}"
            variant_names.append(var_name)
            all_models.extend(
                _render_single_model(
                    var_name, variant.docstring or ref.docstring, variant.params
                )
            )
            all_models.append("")

        all_models.append(f"{ref.name} = Union[{', '.join(variant_names)}]")
    else:
        all_models.extend(_render_single_model(ref.name, ref.docstring, ref.params))

    return "\n".join(header_lines + all_models) + "\n"


PROTO_TENSOR_DEFINITION = """message TensorProto {
  repeated int64 shape = 1;
  string dtype = 2;
  bytes raw_data = 3;
}"""

PROTO_ENUM_DEFINITIONS: Dict[str, str] = {
    "reduction": """enum ReductionType {
  REDUCTION_UNSPECIFIED = 0;
  REDUCTION_NONE = 1;
  REDUCTION_MEAN = 2;
  REDUCTION_SUM = 3;
}""",
    "padding": """enum PaddingMode {
  PADDING_UNSPECIFIED = 0;
  PADDING_VALID = 1;
  PADDING_SAME = 2;
  PADDING_ZEROS = 3;
  PADDING_REFLECT = 4;
  PADDING_REPLICATE = 5;
  PADDING_CIRCULAR = 6;
}""",
    "layout": """enum LayoutMode {
  LAYOUT_UNSPECIFIED = 0;
  LAYOUT_NCHW = 1;
  LAYOUT_NHWC = 2;
  LAYOUT_NCDHW = 3;
  LAYOUT_NDHWC = 4;
}""",
    "mode": """enum InterpolationMode {
  INTERPOLATION_UNSPECIFIED = 0;
  INTERPOLATION_NEAREST = 1;
  INTERPOLATION_LINEAR = 2;
  INTERPOLATION_BILINEAR = 3;
  INTERPOLATION_BICUBIC = 4;
  INTERPOLATION_TRILINEAR = 5;
  INTERPOLATION_AREA = 6;
}""",
}


PROTO_DEFINITIONS_BY_TYPE: Dict[str, str] = {
    "TensorProto": PROTO_TENSOR_DEFINITION,
    "ReductionType": PROTO_ENUM_DEFINITIONS["reduction"],
    "PaddingMode": PROTO_ENUM_DEFINITIONS["padding"],
    "LayoutMode": PROTO_ENUM_DEFINITIONS["layout"],
    "InterpolationMode": PROTO_ENUM_DEFINITIONS["mode"],
}


def _py_type_to_proto(typ: Optional[str], param_name: str = "") -> str:
    """Map Python type to Protobuf type with structured message and enum mapping.

    Args:
        typ: Python type string.
        param_name: Parameter name for shape and enum heuristic mapping.

    Returns:
        Protobuf type string.
    """
    if not typ:
        return "string"

    clean_t = str(typ).lower().strip()
    p_norm = param_name.lower().strip()

    # 1. Structured Tensor types -> TensorProto
    if clean_t in (
        "tensor",
        "torch.tensor",
        "tensorproto",
        "ndarray",
        "array",
    ) or clean_t.startswith("tensor<"):
        return "TensorProto"

    # 2. Dimension shapes -> repeated int64
    if p_norm in ("shape", "size", "dims", "dimensions") or clean_t in (
        "shape",
        "dimensions",
        "tuple[int, ...]",
        "sequence[int]",
    ):
        return "repeated int64"

    # 3. Preserved Enums
    if p_norm in ("reduction", "reduction_type"):
        return "ReductionType"
    if p_norm in ("padding", "padding_mode"):
        return "PaddingMode"
    if p_norm in ("layout", "layout_mode"):
        return "LayoutMode"
    if p_norm == "mode" and "interp" in clean_t:
        return "InterpolationMode"

    if "list" in clean_t or "tuple" in clean_t:
        return "repeated string"
    elif "dict" in clean_t:
        return "map<string, string>"
    elif "int" in clean_t:
        return "int64"
    elif "float" in clean_t:
        return "double"
    elif "bool" in clean_t:
        return "bool"
    elif "str" in clean_t:
        return "string"
    return "string"


def to_protobuf(ref: GhostRef, package: str = "ml_framework") -> str:
    """Convert a GhostRef into a Protocol Buffers (.proto) message definition.

    Args:
        ref: The GhostRef to convert.
        package: The protobuf package name.

    Returns:
        A string containing the .proto definition.

    """
    preamble_blocks: List[str] = []
    seen_blocks: Set[str] = set()

    field_lines: List[str] = []
    field_num = 1
    for param in ref.params:
        if param.kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue

        proto_type = _py_type_to_proto(param.annotation, param_name=param.name)
        def_block = PROTO_DEFINITIONS_BY_TYPE.get(proto_type)
        if def_block and def_block not in seen_blocks:
            seen_blocks.add(def_block)
            preamble_blocks.append(def_block)

        if (
            param.default is not None
            and not proto_type.startswith("repeated")
            and not proto_type.startswith("map")
        ):
            proto_type = f"optional {proto_type}"

        desc = f" // {param.description}" if param.description else ""
        field_lines.append(f"  {proto_type} {param.name} = {field_num};{desc}")
        field_num += 1

    lines = [
        'syntax = "proto3";',
        "",
        f"package {package};",
        "",
    ]
    if preamble_blocks:
        lines.extend(preamble_blocks)
        lines.append("")

    lines.append(
        f"// {ref.docstring}"
        if ref.docstring
        else f"// Generated message for {ref.api_path}"
    )
    lines.append(f"message {ref.name} {{")
    lines.extend(field_lines)
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_llm_prompt_context(refs: List[GhostRef]) -> str:
    """Export GhostRef list into a compact, typed format optimized for LLM prompting.

    Formats functions, operands, attributes, and constraints into concise markdown
    specifications designed to prevent LLM hallucinations during code generation.

    Args:
        refs: List of GhostRef items to format.

    Returns:
        A Markdown-formatted string with compact signatures and parameter constraints.
    """
    blocks: List[str] = []

    for ref in refs:
        params_str_list: List[str] = []
        param_details: List[str] = []

        for p in ref.params:
            type_annot = p.annotation or "Any"
            default_part = f" = {p.default}" if p.default is not None else ""
            params_str_list.append(f"{p.name}: {type_annot}{default_part}")

            desc = f" - {p.description}" if p.description else ""

            constraints: List[str] = []
            p_dtypes = getattr(p, "dtypes", None)
            if p_dtypes:
                constraints.append(f"dtypes: {p_dtypes}")
            p_rank = getattr(p, "rank", None)
            if p_rank is not None:
                constraints.append(f"rank: {p_rank}")
            constr_str = f" [{', '.join(constraints)}]" if constraints else ""

            param_details.append(
                f"  - `{p.name}` ({p.kind}, type `{type_annot}`{default_part}){constr_str}{desc}"
            )

        sig = f"{ref.api_path}({', '.join(params_str_list)})"
        if ref.returns_type:
            sig += f" -> {ref.returns_type}"

        lines = [
            f"### `{ref.api_path}`",
            f"- **Kind**: `{ref.kind}`",
            f"- **Signature**: `{sig}`",
        ]

        if param_details:
            lines.append("- **Parameters**:")
            lines.extend(param_details)

        if ref.raises:
            lines.append(f"- **Raises**: {', '.join(ref.raises)}")

        if ref.environment_tags:
            lines.append(f"- **Environments**: {', '.join(ref.environment_tags)}")

        if ref.docstring:
            first_line = ref.docstring.strip().splitlines()[0]
            lines.append(f"- **Summary**: {first_line}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def export_sass_prompt_context(refs: List[GhostRef]) -> str:
    """Export SASS instruction specifications as canonical assembly prompt templates.

    Formats GPU instructions into real SASS assembly syntax templates rather than
    synthetic Python signatures, specifying valid register files, modifiers, and SM architectures.

    Args:
        refs: List of SASS GhostRef items to format.

    Returns:
        Markdown string formatted with assembly syntax templates and register constraints.
    """
    blocks: List[str] = []

    for ref in refs:
        meta = getattr(ref, "domain_metadata", None) or {}
        modifiers = meta.get("modifiers") or []
        archs = meta.get("valid_architectures") or ref.environment_tags or []
        op_sigs = meta.get("operand_signatures") or []

        mod_suffix = (
            f"{{.{' | .'.join(m.lstrip('.') for m in modifiers[:4])}}}"
            if modifiers
            else ""
        )
        if op_sigs:
            sample_sig = op_sigs[0]
            op_terms = []
            for i, op_t in enumerate(sample_sig):
                if i == 0:
                    op_terms.append(f"R{i} (dst)")
                else:
                    op_terms.append(f"R{i} ({op_t})")
            syntax_str = f"[@P0] {ref.name}{mod_suffix} {', '.join(op_terms)};"
        else:
            syntax_str = f"[@P0] {ref.name}{mod_suffix};"

        lines = [
            f"### `{ref.name}`",
            f"- **Syntax**: `{syntax_str}`",
        ]

        if archs:
            clean_archs = [a for a in archs if a != "cuda"]
            if clean_archs:
                lines.append(f"- **Architectures**: {', '.join(clean_archs)}")

        if modifiers:
            lines.append(f"- **Modifiers**: {', '.join(modifiers)}")

        if ref.params:
            lines.append("- **Operands**:")
            for p in ref.params:
                dir_str = (
                    f" [{p.direction}]"
                    if hasattr(p, "direction") and p.direction
                    else ""
                )
                lines.append(
                    f"  - `{p.name}` ({p.standardized_name or 'operand'}, type `{p.annotation or 'Register'}`){dir_str}"
                )

        if ref.docstring:
            first_line = ref.docstring.strip().splitlines()[0]
            lines.append(f"- **Description**: {first_line}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def export_mlir_prompt_context(refs: List[GhostRef]) -> str:
    """Export MLIR / StableHLO operations as canonical SSA IR prompt templates.

    Formats compiler operations into true MLIR syntax templates, distinguishing
    runtime SSA operands from compile-time attributes, regions, and multi-result signatures.

    Args:
        refs: List of MLIR or StableHLO GhostRef items to format.

    Returns:
        Markdown string formatted with canonical MLIR operation specifications.
    """
    blocks: List[str] = []

    for ref in refs:
        meta = getattr(ref, "domain_metadata", None) or {}
        traits = meta.get("traits") or []
        regions = meta.get("regions") or []

        operands: List[str] = []
        attributes: List[str] = []

        for p in ref.params:
            role = getattr(p, "role", None)
            if str(role) == "IRParameterRole.REGION" or p.annotation == "Region":
                continue
            if str(role) == "IRParameterRole.ATTRIBUTE" or p.kind == "KEYWORD_ONLY":
                attributes.append(f"{p.name} = ...")
            else:
                operands.append(f"%{p.name}")

        op_str = ", ".join(operands)
        attr_str = f" {{{', '.join(attributes)}}}" if attributes else ""
        ret_str = ref.returns_type or "none"
        syntax_str = f"%res = {ref.api_path}({op_str}){attr_str} : {ret_str}"

        lines = [
            f"### `{ref.api_path}`",
            f"- **Syntax**: `{syntax_str}`",
        ]

        if operands:
            lines.append("- **SSA Operands**:")
            for p in ref.params:
                if (
                    str(getattr(p, "role", None)) != "IRParameterRole.ATTRIBUTE"
                    and p.annotation != "Region"
                ):
                    lines.append(f"  - `%{p.name}` (type `{p.annotation or 'Value'}`)")

        if attributes:
            lines.append("- **Attributes**:")
            for p in ref.params:
                if (
                    str(getattr(p, "role", None)) == "IRParameterRole.ATTRIBUTE"
                    or p.kind == "KEYWORD_ONLY"
                ):
                    lines.append(
                        f"  - `{p.name}` (type `{p.annotation or 'Attribute'}`)"
                    )

        results = getattr(ref, "returns", None)
        if results:
            lines.append("- **Results**:")
            for r in results:
                lines.append(f"  - `%{r.name}` (type `{r.type}`)")

        if traits:
            lines.append(f"- **Traits**: {', '.join(str(t) for t in traits)}")

        if regions:
            lines.append(f"- **Regions**: {', '.join(str(r) for r in regions)}")

        if ref.docstring:
            first_line = ref.docstring.strip().splitlines()[0]
            lines.append(f"- **Description**: {first_line}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


COMMON_HALLUCINATION_GUARDS: Dict[str, List[str]] = {
    "torch": [
        "Do NOT pass 'axis' to 'torch.sum' or reduction operations; use 'dim'.",
        "Do NOT pass 'keepdims' to PyTorch reduction ops; use 'keepdim'.",
        "Do NOT pass integer dtypes (e.g. 'int32') to linear algebra ops like 'torch.linalg.inv' or 'torch.cholesky'; floating-point or complex required.",
    ],
    "nvidia_sass": [
        "Uniform registers (UR0-UR63) require Volta+ (sm_70+).",
        "Align 64-bit register pairs to even indices (e.g. R0:R1, not R1:R2).",
        "Align 128-bit register quads to modulo-4 starting indices (e.g. R0:R3).",
    ],
    "amd_rdna": [
        "Dual-issue instructions (v_dual_*) are strictly restricted to GFX11/RDNA3 and GFX12/RDNA4.",
        "Matrix accumulator instructions (v_mfma_*) are strictly restricted to GFX9/CDNA.",
        "Align 64-bit vector registers to even indices (v[0:1], s[0:1]).",
    ],
    "mlir": [
        "Do NOT pass floating-point types to signless integer ops like 'arith.addi'; use 'arith.addf'.",
        "Ensure SSA operand segment sizes match AttrSizedOperandSegments attributes.",
    ],
    "stablehlo": [
        "Verify DotDimensionNumbers contracting dimensions against input operand ranks.",
        "Broadcast dimensions length in 'stablehlo.broadcast_in_dim' must equal input rank.",
    ],
}


def export_scoped_prompt_context(
    framework: str,
    module_prefix: Optional[str] = None,
    max_symbols: int = 50,
) -> str:
    """Export a scoped prompt context with a hierarchical summary index.

    Prevents context window explosion by providing an indexed summary of available
    operations and detailed parameter constraints for up to max_symbols matching APIs.

    Args:
        framework: Name of the framework (e.g. 'torch', 'stablehlo', 'nvidia_sass').
        module_prefix: Optional namespace filter (e.g. 'torch.nn.functional', 'arith').
        max_symbols: Maximum number of detailed symbols to include (default: 50).

    Returns:
        Structured Markdown string containing an index summary followed by detailed signatures.
    """
    from ml_framework_snapshots.mcp_server import get_framework_snapshot
    from ml_framework_snapshots.models import GhostInspector

    snap = get_framework_snapshot(framework)
    categories = snap.get("categories", {})

    all_raw_items: List[Dict[str, Any]] = []
    for _cat, items in sorted(categories.items()):
        all_raw_items.extend(items)

    matching_items: List[Dict[str, Any]] = []
    for item in all_raw_items:
        if not isinstance(item, dict):
            continue
        path = item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
        if module_prefix:
            if path.startswith(module_prefix) or path.startswith(f"{module_prefix}."):
                matching_items.append(item)
        else:
            matching_items.append(item)

    total_matching = len(matching_items)
    selected_items = matching_items[:max_symbols]

    hydrated_refs: List[GhostRef] = []
    for item in selected_items:
        try:
            hydrated_refs.append(GhostInspector.hydrate(item))
        except Exception:
            pass

    index_lines: List[str] = [
        f"# Framework Grounding Context: `{framework}`",
    ]

    guards = COMMON_HALLUCINATION_GUARDS.get(framework.lower())
    if guards:
        index_lines.append("\n## Common Hallucination Guards & Anti-Patterns")
        for g in guards:
            index_lines.append(f"- {g}")

    index_lines.append(f"\n## Index of Available Operations ({total_matching} total)")

    grouped: Dict[str, List[str]] = {}
    for item in matching_items:
        path = item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
        parts = path.split(".")
        group_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].append(path)

    for group_name, members in sorted(grouped.items()):
        sample = ", ".join(f"`{m.split('.')[-1]}`" for m in members[:8])
        if len(members) > 8:
            sample += f", ... (+{len(members) - 8} more)"
        index_lines.append(f"- **{group_name}** ({len(members)} ops): {sample}")

    index_lines.append("\n## Detailed Operation Signatures")

    if framework in ("nvidia_sass", "amd_rdna"):
        details = export_sass_prompt_context(hydrated_refs)
    elif framework in ("mlir", "stablehlo"):
        details = export_mlir_prompt_context(hydrated_refs)
    else:
        details = export_llm_prompt_context(hydrated_refs)

    result_parts = ["\n".join(index_lines), details]

    if total_matching > max_symbols:
        result_parts.append(
            f"*(Showing {max_symbols} of {total_matching} operations to prevent context explosion. "
            f"Query specific submodules or APIs using 'search_apis' or 'get_api_signature' for complete constraints.)*\n"
        )

    return "\n\n".join(result_parts).strip() + "\n"
