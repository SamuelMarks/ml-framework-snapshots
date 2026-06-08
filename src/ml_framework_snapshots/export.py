"""Export module for generating JSON Schema, OpenAPI specifications, Pydantic classes, and Protobuf definitions."""

from typing import Dict, Any, List
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
    import cdd.emit.json_schema
    from typing import cast

    ir = _ghost_to_cdd_ir(ref)
    return cast(
        Dict[str, Any], cdd.emit.json_schema.json_schema(ir, identifier=ref.api_path)
    )


def to_openapi(refs: List[GhostRef]) -> Dict[str, Any]:
    """Convert a list of GhostRefs into an OpenAPI specification.

    Args:
        refs: A list of GhostRefs.

    Returns:
        A dictionary representing the OpenAPI specification.

    """
    import cdd.emit.openapi
    from cdd.emit.utils.openapi_utils import NameModelRouteIdCrud

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

    return cast(Dict[str, Any], cdd.emit.openapi.openapi(nm_cruds))


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
        f"// {ref.docstring}"
        if ref.docstring
        else f"// Generated message for {ref.api_path}",
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
