#!/usr/bin/env python3
"""Build StableHLO Ground Truth Snapshot.

Parses the official openxla/stablehlo TableGen ODS definitions (StableHloOps.td)
or specification spec.md to extract operations, SSA operands, attributes, regions,
traits, and return types, outputting a standard GhostRef JSON snapshot.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import urllib.request

SPEC_URL = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
TABLEGEN_URL = "https://raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/dialect/StablehloOps.td"


def strip_tablegen_comments(text: str) -> str:
    """Strip block and line comments from TableGen source cleanly.

    Args:
        text: Raw TableGen source code.

    Returns:
        Source code with comments removed.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def is_attribute(type_str: str) -> bool:
    """Determine if a spec type string represents an attribute rather than an operand.

    Args:
        type_str: The type string from the specification.

    Returns:
        True if the type represents an attribute, False otherwise.
    """
    t = type_str.lower()
    if any(k in t for k in ("tensor", "value", "anytype", "anyshaped")):
        return False
    return (
        "constant" in t
        or "enum" in t
        or "function" in t
        or "region" in t
        or "attribute" in t
        or "floattype" in t
        or "si64" == t.strip("`")
        or "string" in t
        or "aliasing parts" in t
        or "attr" in t
        or "dimensionnumbers" in t
        or "precision" in t
        or "algorithm" in t
        or "config" in t
        or "direction" in t
        or "distribution" in t
        or "comparison" in t
        or "transpose" in t
        or "mode" in t
        or "fft" in t
        or "dense" in t
        or "array" in t
    )


def parse_stablehlo_tablegen(content: str) -> List[Dict[str, Any]]:
    """Parse StableHLO TableGen / ODS definitions into structured operations.

    Extracts:
        - Mnemonic & Op class name
        - Verification constraints and dialect traits (Commutative, SameOperandsAndResultType, etc.)
        - SSA Operands (variadic, optional, tensor)
        - Attributes (enums, array attributes, constants)
        - SSA Results (single return, multi-return tuples)
        - Op Regions and block arguments (reduce, while, if, sort, scatter, gather)

    Args:
        content: Raw TableGen file content.

    Returns:
        List of structured GhostRef operation dictionaries.
    """
    clean_content = strip_tablegen_comments(content)
    ghost_refs: List[Dict[str, Any]] = []

    # 1. Parse base classes for inherited arguments/results
    base_classes: Dict[str, Dict[str, Any]] = {}
    for m in re.finditer(
        r"\bclass\s+([A-Za-z0-9_]+)(?:<[\s\S]*?>)?\s*(?::\s*([^{;]+))?\s*(\{|;)",
        clean_content,
    ):
        cls_name = m.group(1)
        supers_raw = m.group(2) or ""
        term = m.group(3)
        cls_body = ""
        if term == "{":
            start = m.end()
            depth = 1
            pos = start
            while pos < len(clean_content) and depth > 0:
                if clean_content[pos] == "{":
                    depth += 1
                elif clean_content[pos] == "}":
                    depth -= 1
                pos += 1
            cls_body = clean_content[start : pos - 1]

        args_m = re.search(
            r"let\s+arguments\s*=\s*\(\s*ins(.*?)\);", cls_body, re.DOTALL
        ) or re.search(r"Arguments<\s*\(\s*ins(.*?)\)>", supers_raw, re.DOTALL)
        res_m = re.search(
            r"let\s+results\s*=\s*\(\s*outs(.*?)\);", cls_body, re.DOTALL
        ) or re.search(r"Results<\s*\(\s*outs(.*?)\)>", supers_raw, re.DOTALL)
        supers = [
            s.split("<")[0].strip() for s in re.split(r",", supers_raw) if s.strip()
        ]
        base_classes[cls_name] = {
            "args": args_m.group(1).strip() if args_m else None,
            "results": res_m.group(1).strip() if res_m else None,
            "supers": supers,
        }

    def resolve_field(cls_name: str, field_name: str) -> Optional[str]:
        """Recursively resolve inherited TableGen fields from base classes.

        Args:
            cls_name: Name of the class to resolve.
            field_name: Field key to lookup ('args', 'results').

        Returns:
            Resolved field string value or None.
        """
        if cls_name not in base_classes:
            return None
        val = base_classes[cls_name].get(field_name)
        if val is not None:
            return str(val)
        for s in base_classes[cls_name].get("supers", []):
            res = resolve_field(s, field_name)
            if res is not None:
                return str(res)
        return None

    # 2. Extract records using balanced brace parsing
    for m in re.finditer(
        r"\bdef\s+([A-Za-z0-9_]+Op)\s*:\s*([A-Za-z0-9_]+)", clean_content
    ):
        class_name = m.group(1)
        base_cls = m.group(2)
        start = m.end()

        # Scan until opening brace or semicolon, balancing <>, (), and []
        depth_angle = 0
        depth_paren = 0
        depth_bracket = 0
        p = start
        term = None
        while p < len(clean_content):
            c = clean_content[p]
            if c == "<":
                depth_angle += 1
            elif c == ">":
                depth_angle -= 1
            elif c == "(":
                depth_paren += 1
            elif c == ")":
                depth_paren -= 1
            elif c == "[":
                depth_bracket += 1
            elif c == "]":
                depth_bracket -= 1
            elif depth_angle == 0 and depth_paren == 0 and depth_bracket == 0:
                if c in ("{", ";"):
                    term = c
                    break
            p += 1

        targs_raw = clean_content[start:p].strip()
        body = ""
        if term == "{":
            start_body = p + 1
            depth = 1
            b_idx = start_body
            while b_idx < len(clean_content) and depth > 0:
                if clean_content[b_idx] == "{":
                    depth += 1
                elif clean_content[b_idx] == "}":
                    depth -= 1
                b_idx += 1
            body = clean_content[start_body : b_idx - 1]

        # Extract operation mnemonic
        mnem = re.search(r'let\s+mnemonic\s*=\s*"([a-z0-9_]+)"', body)
        if mnem:
            op_name = mnem.group(1)
        elif targs_raw:
            m_targ = re.search(r'"([a-z0-9_]+)"', targs_raw)
            if m_targ:
                op_name = m_targ.group(1)
            else:
                raw_name = class_name.replace("StableHLO_", "").replace("Op", "")
                op_name = re.sub(r"(?<!^)(?=[A-Z])", "_", raw_name).lower()
        else:
            raw_name = class_name.replace("StableHLO_", "").replace("Op", "")
            op_name = re.sub(r"(?<!^)(?=[A-Z])", "_", raw_name).lower()

        clean_class_name = (
            class_name.replace("StableHLO_", "")
            if class_name.startswith("StableHLO_")
            else class_name
        )

        traits: List[str] = []
        traits_match = re.search(r"\[(.*?)\]", targs_raw)
        if traits_match:
            traits.extend(
                [
                    t.strip()
                    for t in traits_match.group(1).split(",")
                    if t.strip() and not t.strip().startswith("//")
                ]
            )
        traits_in_body = re.search(r"let\s+traits\s*=\s*\[(.*?)\];", body, re.DOTALL)
        if traits_in_body:
            traits.extend(
                [
                    t.strip()
                    for t in traits_in_body.group(1).split(",")
                    if t.strip() and not t.strip().startswith("//")
                ]
            )

        args_match = re.search(
            r"let\s+arguments\s*=\s*\(\s*ins(.*?)\);", body, re.DOTALL
        ) or re.search(r"Arguments<\s*\(\s*ins(.*?)\)>", targs_raw, re.DOTALL)
        args_str = (
            args_match.group(1).strip()
            if args_match
            else resolve_field(base_cls, "args")
        )
        if not args_str and ("Binary" in base_cls or "Binary" in class_name):
            args_str = "HLO_Tensor:$lhs, HLO_Tensor:$rhs"
        elif not args_str and ("Unary" in base_cls or "Unary" in class_name):
            args_str = "HLO_Tensor:$operand"
        elif not args_str and ("Cast" in base_cls or "Cast" in class_name):
            args_str = "HLO_Tensor:$operand"

        params: List[Dict[str, Any]] = []
        operands: List[Dict[str, str]] = []
        attributes: List[Dict[str, str]] = []

        if args_str:
            depth_angle = 0
            parts_list: List[str] = []
            curr_chars: List[str] = []
            for c in args_str:
                if c == "<":
                    depth_angle += 1
                elif c == ">":
                    depth_angle -= 1
                elif c == "," and depth_angle == 0:
                    parts_list.append("".join(curr_chars).strip())
                    curr_chars = []
                    continue
                curr_chars.append(c)
            if curr_chars:
                parts_list.append("".join(curr_chars).strip())

            for arg_item in parts_list:
                arg_item = arg_item.strip()
                if not arg_item:
                    continue
                parts = arg_item.split(":$")
                if len(parts) == 2:
                    arg_type, arg_name = parts[0].strip(), parts[1].strip()
                else:
                    arg_parts = arg_item.split(":")
                    arg_type = arg_parts[0].strip()
                    arg_name = (
                        arg_parts[1].lstrip("$").strip()
                        if len(arg_parts) > 1
                        else f"arg{len(params)}"
                    )

                is_attr = is_attribute(arg_type)
                kind = "KEYWORD_ONLY" if is_attr else "POSITIONAL_OR_KEYWORD"
                p_entry = {
                    "name": arg_name,
                    "annotation": arg_type,
                    "default": None,
                    "kind": kind,
                    "description": f"{'Attribute' if is_attr else 'Operand'} of type {arg_type}",
                    "standardized_name": "attribute" if is_attr else "operand",
                }
                params.append(p_entry)
                if is_attr:
                    attributes.append({"name": arg_name, "type": arg_type})
                else:
                    operands.append({"name": arg_name, "type": arg_type})

        # Extract regions
        regions: List[str] = []
        regions_match = re.search(
            r"let\s+regions\s*=\s*\(\s*region(.*?)\);", body, re.DOTALL
        )
        if regions_match:
            reg_str = regions_match.group(1).strip()
            for reg_item in reg_str.split(","):
                reg_item = reg_item.strip()
                if not reg_item:
                    continue
                parts = reg_item.split(":$")
                reg_name = parts[1].strip() if len(parts) == 2 else reg_item
                regions.append(reg_name)
                params.append(
                    {
                        "name": reg_name,
                        "annotation": "Region",
                        "default": None,
                        "kind": "KEYWORD_ONLY",
                        "description": f"Op region block {reg_name}",
                        "standardized_name": "region",
                    }
                )

        # Extract results
        results_match = re.search(
            r"let\s+results\s*=\s*\(\s*outs(.*?)\);", body, re.DOTALL
        ) or re.search(r"Results<\s*\(\s*outs(.*?)\)>", targs_raw, re.DOTALL)
        res_str = (
            results_match.group(1).strip()
            if results_match
            else resolve_field(base_cls, "results")
        )
        if not res_str and (
            "Binary" in base_cls or "Unary" in base_cls or "Cast" in base_cls
        ):
            res_str = "HLO_Tensor:$result"
        returns_type = None
        results: List[Dict[str, str]] = []

        if res_str:
            depth_angle = 0
            res_parts_list: List[str] = []
            res_curr_chars: List[str] = []
            for c in res_str:
                if c == "<":
                    depth_angle += 1
                elif c == ">":
                    depth_angle -= 1
                elif c == "," and depth_angle == 0:
                    res_parts_list.append("".join(res_curr_chars).strip())
                    res_curr_chars = []
                    continue
                res_curr_chars.append(c)
            if res_curr_chars:
                res_parts_list.append("".join(res_curr_chars).strip())

            res_types: List[str] = []
            for res_item in res_parts_list:
                res_item = res_item.strip()
                if not res_item:
                    continue
                parts = res_item.split(":$")
                res_type = parts[0].strip()
                res_name = parts[1].strip() if len(parts) == 2 else "result"
                results.append({"name": res_name, "type": res_type})
                res_types.append(res_type)

            if len(res_types) == 1:
                returns_type = res_types[0]
            elif len(res_types) > 1:
                returns_type = f"tuple[{', '.join(res_types)}]"

        summary_match = re.search(r'let\s+summary\s*=\s*"([^"]*)";', body)
        docstring = summary_match.group(1) if summary_match else None

        ghost_refs.append(
            {
                "api_path": f"stablehlo.{op_name}",
                "name": op_name,
                "class_name": clean_class_name,
                "kind": "function",
                "is_public": True,
                "has_varargs": False,
                "environment_tags": ["cpu", "cuda", "rocm", "tpu"],
                "aliases": [],
                "overloads": [],
                "params": params,
                "operands": operands,
                "attributes": attributes,
                "results": results,
                "returns_type": returns_type,
                "returns_description": None,
                "raises": [],
                "docstring": docstring,
                "traits": traits,
                "regions": regions,
            }
        )

    return ghost_refs


def extract_ops(source_content: Optional[str] = None) -> List[Dict[str, Any]]:
    """Download and parse StableHLO TableGen or spec.md to extract ops.

    Args:
        source_content: Optional raw string content (TableGen or markdown spec).

    Returns:
        List of parsed operations as dictionary objects.
    """
    content = source_content
    if content is None:
        try:
            content = (
                urllib.request.urlopen(TABLEGEN_URL, timeout=10).read().decode("utf-8")
            )
        except Exception:  # pragma: no cover
            content = urllib.request.urlopen(SPEC_URL).read().decode("utf-8")

    assert content is not None

    if "StableHLO_" in content:
        return parse_stablehlo_tablegen(content)

    ops: Dict[str, Dict[str, Any]] = {}
    current_op = None
    current_section = None

    for line in content.split("\n"):
        if line.startswith("### "):
            op_name = line.strip()[4:]
            if op_name.replace("_", "").isalnum():
                current_op = op_name
                ops[current_op] = {"inputs": [], "outputs": [], "semantics": []}
                current_section = None

        elif line.startswith("#### "):
            section_name = line.strip()[5:]
            if section_name.lower() in ("inputs", "outputs", "semantics"):
                current_section = section_name.lower()
            else:
                current_section = None
        elif current_op and current_section == "semantics":
            if line.strip() and not line.startswith("####"):
                ops[current_op]["semantics"].append(line)
        elif (
            line.startswith("|")
            and current_section in ("inputs", "outputs")
            and current_op
        ):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if (
                len(parts) >= 2
                and "---" not in parts[0]
                and "Label" not in parts[0]
                and "Name" not in parts[0]
            ):
                if current_section == "inputs":
                    if len(parts) >= 3 and parts[0].startswith("("):
                        name = parts[1]
                        type_str = parts[2]
                    else:
                        name = parts[0]
                        type_str = parts[1]
                else:
                    name = parts[0]
                    type_str = parts[1] if len(parts) > 1 else ""

                name = name.replace("`", "")
                if name:
                    ops[current_op][current_section].append((name, type_str))

    ghost_refs = []

    for op_name, data in ops.items():
        if not data["inputs"] and not data["outputs"]:
            continue

        params = []
        for name, type_str in data["inputs"]:
            kind = "KEYWORD_ONLY" if is_attribute(type_str) else "POSITIONAL_OR_KEYWORD"
            params.append(
                {
                    "name": name,
                    "annotation": type_str,
                    "default": None,
                    "kind": kind,
                    "description": None,
                    "standardized_name": None,
                }
            )

        returns_type = None
        if data["outputs"]:
            # Join multiple return types into a tuple or single string
            returns_type = ", ".join(t for _, t in data["outputs"] if t)
            if len(data["outputs"]) > 1:
                returns_type = f"tuple[{returns_type}]"

        docstring = "\n".join(data["semantics"]).strip()

        ghost_refs.append(
            {
                "api_path": f"stablehlo.{op_name}",
                "name": op_name,
                "kind": "function",
                "is_public": True,
                "has_varargs": False,
                "environment_tags": [],
                "aliases": [],
                "overloads": [],
                "params": params,
                "returns_type": returns_type,
                "returns_description": None,
                "raises": [],
                "docstring": docstring if docstring else None,
            }
        )

    return ghost_refs


def main() -> None:
    """Main entrypoint for snapshot generation."""
    ghost_refs = extract_ops()

    snapshot_data = {"categories": {"stablehlo_op": ghost_refs}}

    out_dir = os.path.join(os.path.dirname(__file__), "..", "snapshots")
    out_path = os.path.abspath(os.path.join(out_dir, "stablehlo_v1.0.0.json"))

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snapshot_data, f, indent=2, sort_keys=True)
        f.write("\n")

    exhaustive_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "frameworks",
            "stablehlo_exhaustive.json",
        )
    )
    with open(exhaustive_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ghost_refs, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(ghost_refs)} ops to {out_path} and {exhaustive_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
