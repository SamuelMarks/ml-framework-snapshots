"""Ghost Type Stubs Generator.

Generate .pyi stub files from snapshot JSON data.
"""

import os

from typing import Dict, Any, List, Tuple
from pathlib import Path
import ast


def _sanitize_default(default_str: str) -> str:
    """Sanitize complex or un-importable default values for stubs.

    Args:
        default_str: The default value string to sanitize.

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
    snapshot_data: Any, output_dir: str, include_nonpublic: bool = False
) -> None:
    """Generate .pyi stub files from a snapshot dictionary.

    Args:
        snapshot_data: The snapshot dictionary or list.
        output_dir: The base directory where stubs should be written.
        include_nonpublic: Whether to generate stubs for non-public components.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    modules: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}

    if isinstance(snapshot_data, list):
        categories_dict = {"all": snapshot_data}
    elif isinstance(snapshot_data, dict):
        categories_dict = snapshot_data.get("categories", {})
    else:
        categories_dict = {}

    for cat, items in categories_dict.items():
        for item in items:
            if not include_nonpublic and not item.get("is_public", True):
                continue

            api_path = (
                item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
            )
            if not api_path:
                continue

            parts = api_path.split(".")
            if len(parts) == 1:
                module_name = item.get("framework") or item.get("dialect") or ""
                obj_name = parts[0]
            else:
                module_name = ".".join(parts[:-1])
                obj_name = parts[-1]

            if module_name not in modules:
                modules[module_name] = []

            modules[module_name].append((obj_name, item))

    for module_name, items in modules.items():
        if not module_name:
            continue

        module_path = Path(os.path.join(out_path, module_name.replace(".", os.sep)))
        module_path.mkdir(parents=True, exist_ok=True)
        init_file = Path(os.path.join(module_path, "__init__.pyi"))

        lines = []
        lines.append(
            "from typing import Any, Optional, Union, Tuple, List, Callable, Dict, overload"
        )
        lines.append("")

        for obj_name, item in items:
            kind = item.get("kind", "function")
            params = item.get("params", [])
            has_varargs = item.get("has_varargs", False)

            # Generate @overload signatures if defined
            for ov in item.get("overloads", []):
                ov_params = ov.get("params", [])
                ov_param_strs = []
                if kind == "class":
                    ov_param_strs.append("self")
                for p in ov_params:
                    p_name = p.get("name")
                    p_anno = p.get("annotation") or "Any"
                    p_default = p.get("default")
                    p_kind = p.get("kind")
                    if p_kind == "VAR_POSITIONAL":
                        ov_param_strs.append(f"*{p_name}")
                    elif p_kind == "VAR_KEYWORD":
                        ov_param_strs.append(f"**{p_name}")
                    else:
                        p_str = f"{p_name}: {p_anno}"
                        if p_default is not None:
                            p_str += f" = {_sanitize_default(str(p_default))}"
                        ov_param_strs.append(p_str)
                ov_sig = ", ".join(ov_param_strs)
                ov_ret = (
                    f" -> {ov.get('returns_type')}"
                    if ov.get("returns_type")
                    else " -> Any"
                )
                lines.append("@overload")
                if kind == "class":
                    lines.append(f"def __init__({ov_sig}){ov_ret}: ...")
                else:
                    lines.append(f"def {obj_name}({ov_sig}){ov_ret}: ...")

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

        stub_content = "\n".join(lines) + "\n"
        validate_pyi_stub(stub_content)

        with open(init_file, "w", encoding="utf-8") as f:
            f.write(stub_content)


def validate_pyi_stub(stub_content: str) -> bool:
    """Validate that a generated .pyi stub compiles into valid Python AST.

    Args:
        stub_content: The string content of the .pyi stub file.

    Returns:
        True if the stub compiles without syntax errors.

    Raises:
        SyntaxError: If the stub content is syntactically invalid.
    """
    try:
        ast.parse(stub_content)
    except SyntaxError as e:
        raise SyntaxError(str(e)) from e
    return True
