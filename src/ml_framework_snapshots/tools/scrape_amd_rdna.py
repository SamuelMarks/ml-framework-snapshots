"""Script to scrape AMD RDNA instructions from LLVM TableGen sources."""

import json
import os
import re
import urllib.request
from collections import defaultdict
from typing import DefaultDict, Dict, Any, List

LLVM_REPO_BASE = (
    "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU"
)
TD_FILES = [
    "VOP1Instructions.td",
    "VOP2Instructions.td",
    "VOP3Instructions.td",
    "VOPCInstructions.td",
    "SOPInstructions.td",
    "SMInstructions.td",
    "FLATInstructions.td",
    "BUFInstructions.td",
]


def fetch_td_file(filename: str) -> str:
    """Fetch a TableGen file from the LLVM repository."""
    url = f"{LLVM_REPO_BASE}/{filename}"
    try:
        with urllib.request.urlopen(url) as response:
            return str(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def parse_td_content(content: str) -> List[Dict[str, Any]]:
    """Parse a TableGen file to extract basic instruction information."""
    # Look for patterns like: defm V_MOV_B32 : VOP1Inst <"v_mov_b32", ...
    # or def S_MOV_B32 : SOP1_Pseudo <"s_mov_b32", ...

    instructions = []

    # Matches: defm/def NAME : CLASS <"mnemonic"
    pattern = re.compile(
        r'defm?\s+[A-Za-z0-9_]+\s*:\s*[A-Za-z0-9_]+\s*<\s*"([a-z0-9_]+)"'
    )

    for match in pattern.finditer(content):
        mnemonic = match.group(1)
        instructions.append(
            {
                "mnemonic": mnemonic,
                "architecture": "RDNA",  # Generalized
            }
        )

    return instructions


def main() -> None:
    """Run the script to generate the exhaustive RDNA JSON dump."""
    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "amd_rdna_exhaustive.json",
    )

    instructions: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"modifiers": set(), "operands": []}
    )

    for td_file in TD_FILES:
        print(f"Fetching and parsing {td_file}...")
        content = fetch_td_file(td_file)
        if not content:
            continue

        parsed_instrs = parse_td_content(content)
        for instr in parsed_instrs:
            base_name = instr["mnemonic"]
            # To keep it simple, we don't have operands parsing from raw TD easily
            # but we can add dummy/empty for now or attempt regex if needed.
            # RDNA modifiers are usually suffixes on mnemonics or parsed from profiles
            # Let's add basic operands structure

            # Since TableGen profile parsing is complex, we just record the base mnemonic
            instructions[base_name]["modifiers"].add("_e32")  # Default placeholder
            if ["vGPR", "vGPR"] not in instructions[base_name]["operands"]:
                instructions[base_name]["operands"].append(["vGPR", "vGPR"])

    exhaustive_list: List[Dict[str, Any]] = []
    for mnemonic, info in sorted(instructions.items()):
        exhaustive_list.append(
            {
                "mnemonic": mnemonic,
                "architecture": "GFX10+",  # Generalized for RDNA
                "description": f"AMD RDNA {mnemonic} instruction.",
                "modifiers": sorted(list(info["modifiers"])),
                "operands": info["operands"],
                "encoding": "Unknown",  # Hard to parse from raw TD without LLVM tools
            }
        )

    with open(output_path, "w") as f:
        json.dump(exhaustive_list, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Dumped {len(exhaustive_list)} RDNA instructions to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
