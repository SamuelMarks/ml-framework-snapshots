"""Script to generate an exhaustive JSON dump of NVIDIA SASS instructions."""

import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, Any, List


def main() -> None:
    """Run the script to generate the exhaustive SASS JSON dump."""
    input_path = "/tmp/isa.json"
    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "nvidia_sass_exhaustive.json",
    )

    with open(input_path, "r") as f:
        data = json.load(f)

    # We want to group by base_name
    # schema:
    # {
    #   "mnemonic": str,
    #   "modifiers": List[str],
    #   "operands": List[List[str]],
    #   "description": str,
    #   "architecture": str
    # }

    instructions: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"modifiers": set(), "operands": []}
    )

    for key, val in data.items():
        parsed = val.get("parsed", {})
        base_name = parsed.get("base_name")
        if not base_name:
            continue

        # Accumulate modifiers from parsed
        for mod in parsed.get("modifiers", []):
            instructions[base_name]["modifiers"].add(mod)

        # Also check opcode_modis from the root
        for mod in val.get("opcode_modis", []):
            instructions[base_name]["modifiers"].add(mod)

        # Build operand signature
        sig: List[str] = []
        for op in parsed.get("operands", []):
            op_type = op.get("type")
            if op_type == "RegOperand":
                sig.append(op.get("reg_type", "R"))
            elif op_type == "IntIMMOperand":
                sig.append("I")
            elif op_type == "FloatIMMOperand":
                sig.append("FI")
            elif op_type == "ConstantMemOperand":
                sig.append("c[x][y]" if not op.get("cx") else "cx[x][y]")
            elif op_type == "AddressOperand":
                sig.append("ADDR")
            else:
                sig.append(str(op_type))

        if sig not in instructions[base_name]["operands"]:
            instructions[base_name]["operands"].append(sig)

    exhaustive_list: List[Dict[str, Any]] = []
    for mnemonic, info in sorted(instructions.items()):
        exhaustive_list.append(
            {
                "mnemonic": mnemonic,
                "architecture": "sm_80+",  # Generalized minimum assuming standard Turing/Ampere/Hopper
                "description": f"NVIDIA SASS {mnemonic} instruction.",
                "modifiers": sorted(list(info["modifiers"])),
                "operands": info["operands"],
            }
        )

    with open(output_path, "w") as f:
        json.dump(exhaustive_list, f, indent=4)
    print(f"Dumped {len(exhaustive_list)} SASS instructions to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
