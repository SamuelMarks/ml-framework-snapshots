"""Export module for generating JSON Schema, OpenAPI specifications, Pydantic classes, and Protobuf definitions."""

from typing import Dict, Any, List, Optional
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


def to_pydantic(ref: GhostRef) -> str:
    """Convert a GhostRef into a Pydantic V2 class definition string.

    Args:
        ref: The GhostRef to convert.

    Returns:
        A string containing the Python source for the Pydantic model.

    """
    lines = [
        "from pydantic import BaseModel, Field",
        "from typing import Any, Optional, Union, List, Dict",
        "",
        "",
        f"class {ref.name}(BaseModel):",
    ]

    if ref.docstring:
        lines.append(f'    """{ref.docstring}"""')
    else:
        lines.append('    """Generated Pydantic model."""')

    if not ref.params:
        lines.append("    pass")
    else:
        for param in ref.params:
            typ = param.annotation if param.annotation else "Any"
            desc = param.description.replace('"', "'") if param.description else ""
            default_val = param.default

            if param.kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
                continue  # Skip *args / **kwargs for structured Pydantic

            if default_val is None:
                # Required field
                field_def = (
                    f'Field(..., description="{desc}")' if desc else "Field(...)"
                )
                lines.append(f"    {param.name}: {typ} = {field_def}")
            else:
                # Optional/Default field
                field_def = (
                    f'Field(default={default_val}, description="{desc}")'
                    if desc
                    else f"{default_val}"
                )
                lines.append(f"    {param.name}: {typ} = {field_def}")

    return "\n".join(lines) + "\n"


def _py_type_to_proto(typ: str) -> str:
    """Map Python type to Protobuf type.

    Args:
        typ: python type string.

    Returns:
        proto type string.
    """
    if not typ:
        return "string"  # Fallback

    typ = typ.lower()
    if "list" in typ or "tuple" in typ:
        return "repeated string"  # Simplification for complex generics
    elif "dict" in typ:
        return "map<string, string>"
    elif "int" in typ:
        return "int64"
    elif "float" in typ:
        return "double"
    elif "bool" in typ:
        return "bool"
    elif "str" in typ:
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
    lines = [
        'syntax = "proto3";',
        "",
        f"package {package};",
        "",
        (
            f"// {ref.docstring}"
            if ref.docstring
            else f"// Generated message for {ref.api_path}"
        ),
        f"message {ref.name} {{",
    ]

    field_num = 1
    for param in ref.params:
        if param.kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue

        proto_type = _py_type_to_proto(param.annotation or "Any")
        # Optional semantics in proto3 can be explicit with `optional` or implicit
        if (
            param.default is not None
            and not proto_type.startswith("repeated")
            and not proto_type.startswith("map")
        ):
            proto_type = f"optional {proto_type}"

        desc = f" // {param.description}" if param.description else ""
        lines.append(f"  {proto_type} {param.name} = {field_num};{desc}")
        field_num += 1

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
            param_details.append(
                f"  - `{p.name}` ({p.kind}, type `{type_annot}`{default_part}){desc}"
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
        f"## Index of Available Operations ({total_matching} total)",
    ]

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
