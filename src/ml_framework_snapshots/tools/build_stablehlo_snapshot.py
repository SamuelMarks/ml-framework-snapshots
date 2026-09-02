#!/usr/bin/env python3
"""Build StableHLO Ground Truth Snapshot.

Parses the official openxla/stablehlo spec.md to extract operations,
operands, attributes, and return types, outputting a standard GhostRef JSON snapshot.
"""

import json
import os
import urllib.request
from typing import Dict, Any, List

SPEC_URL = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"


def is_attribute(type_str: str) -> bool:
    """Determine if a spec type string represents an attribute rather than an operand.

    Args:
        type_str: The type string from the specification.

    Returns:
        True if the type represents an attribute, False otherwise.
    """
    t = type_str.lower()
    return (
        "constant" in t
        or "enum" in t
        or "function" in t
        or "region" in t
        or "attribute dictionary" in t
        or "floattype" in t
        or "si64" == t.strip("`")
        or "string" in t
        or "aliasing parts" in t
    )


def extract_ops() -> List[Dict[str, Any]]:
    """Download and parse spec.md to extract ops.

    Returns:
        List of parsed operations as dictionary objects.
    """
    print(f"Downloading {SPEC_URL}...")
    content = urllib.request.urlopen(SPEC_URL).read().decode("utf-8")
    print("Parsing operations...")

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
    out_path = os.path.abspath(os.path.join(out_dir, "stablehlo_v1.0.json"))

    with open(out_path, "w") as f:
        json.dump(snapshot_data, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(ghost_refs)} ops to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
