"""Ghost Type Stubs Generator.

Generate .pyi stub files from snapshot JSON data.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import ast


def _sanitize_default(default_str: str) -> str:
    """Sanitize complex or un-importable default values for stubs.

    Args:
        default_str: description

    Returns:
        The sanitized string.
    """
    if default_str == "<unrepresentable>":
        return "..."
    try:
        # Check if the default string parses as valid Python
        node = ast.parse(default_str, mode="eval").body
        # We can selectively let things through
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, (ast.List, ast.Dict, ast.Tuple, ast.Set)):
            return default_str
        elif isinstance(node, ast.Name):
            return default_str
        elif isinstance(node, ast.Attribute):
            return default_str
        elif isinstance(node, ast.UnaryOp):
            return default_str
        elif isinstance(node, ast.BinOp):
            return default_str
        return "..."
    except SyntaxError:
        return "..."


def generate_stubs(
    snapshot_data: Dict[str, Any], output_dir: str, include_nonpublic: bool = False
) -> None:
    """Generate .pyi stub files from a snapshot dictionary.

    Args:
        snapshot_data: The snapshot dictionary.
        output_dir: The base directory where stubs should be written.
        include_nonpublic: Whether to generate stubs for non-public components.

    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    modules: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}

    for cat, items in snapshot_data.get("categories", {}).items():
        for item in items:
            if not include_nonpublic and not item.get("is_public", True):
                continue

            api_path = item.get("api_path", "")
            if not api_path:
                continue

            parts = api_path.split(".")
            module_name = ".".join(parts[:-1])
            obj_name = parts[-1]

            if module_name not in modules:
                modules[module_name] = []

            modules[module_name].append((obj_name, item))

    for module_name, items in modules.items():
        if not module_name:
            continue

        module_path = out_path / module_name.replace(".", "/")
        module_path.mkdir(parents=True, exist_ok=True)
        init_file = module_path / "__init__.pyi"

        lines = []
        lines.append(
            "from typing import Any, Optional, Union, Tuple, List, Callable, Dict"
        )
        lines.append("")

        for obj_name, item in items:
            kind = item.get("kind", "function")
            params = item.get("params", [])
            has_varargs = item.get("has_varargs", False)

            param_strs = []
            if kind == "class":
                param_strs.append("self")

            for p in params:
                p_name = p.get("name")
                p_anno = p.get("annotation")
                p_default = p.get("default")
                p_kind = p.get("kind")

                if p_kind == "VAR_POSITIONAL":
                    p_str = f"*{p_name}"
                elif p_kind == "VAR_KEYWORD":
                    p_str = f"**{p_name}"
                else:
                    p_str = p_name
                    if p_anno:
                        p_str += f": {p_anno}"
                    else:
                        p_str += ": Any"

                    if p_default is not None:
                        sanitized = _sanitize_default(str(p_default))
                        p_str += f" = {sanitized}"

                param_strs.append(p_str)

            if has_varargs and not any(
                p.get("kind") == "VAR_POSITIONAL" for p in params
            ):
                param_strs.append("*args: Any")

            sig = ", ".join(param_strs)

            ret_type = item.get("returns_type")
            ret_str = f" -> {ret_type}" if ret_type else " -> Any"

            if kind == "class":
                lines.append(f"class {obj_name}:")
                if param_strs == ["self"]:
                    lines.append(f"    def __init__(self){ret_str}: ...")
                else:
                    lines.append(f"    def __init__({sig}){ret_str}: ...")
                lines.append("")
            else:
                lines.append(f"def {obj_name}({sig}){ret_str}: ...")
                lines.append("")

        with open(init_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
