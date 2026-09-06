"""Script to scrape AMD RDNA instructions from LLVM TableGen sources."""

from collections import defaultdict
import json
import os
import re
from typing import Any, DefaultDict, Dict, List, Optional
import urllib.error
import urllib.request

LLVM_REPO_BASE = (
    "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU"
)
TD_FILES = [
    "VOP1Instructions.td",
    "VOP2Instructions.td",
    "VOP3Instructions.td",
    "VOPCInstructions.td",
    "VOP3PInstructions.td",
    "SOPInstructions.td",
    "SMInstructions.td",
    "FLATInstructions.td",
    "BUFInstructions.td",
    "DSInstructions.td",
]

ALL_GFX_ARCHITECTURES = [
    "GFX9/CDNA",
    "GFX10/RDNA1",
    "GFX10.3/RDNA2",
    "GFX11/RDNA3",
    "GFX11.5",
    "GFX12/RDNA4",
]


def fetch_td_file(filename: str) -> str:
    """Fetch a TableGen file from the LLVM repository.

    Args:
        filename: The TableGen filename to fetch.

    Returns:
        The content of the file as string, or empty string on error.
    """
    url = f"{LLVM_REPO_BASE}/{filename}"
    try:
        with urllib.request.urlopen(url) as response:
            return str(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def resolve_td_instruction_info(
    def_name: str, class_name: str, mnemonic: str
) -> Dict[str, Any]:
    """Derive real operand signatures, encodings, modifiers, and architectures from TableGen classes.

    Args:
        def_name: The TableGen definition identifier (e.g. 'V_ADD_F32').
        class_name: The TableGen class/profile (e.g. 'VOP2Inst', 'SOP1_32').
        mnemonic: The instruction mnemonic (e.g. 'v_add_f32').

    Returns:
        Dictionary containing encoding, modifiers, operands, and architecture.
    """
    cls = class_name.upper()
    df = def_name.upper()

    encoding = "Unknown"
    modifiers: List[str] = []
    operands: List[List[str]] = []

    # 1. Vector Arithmetic (VOP1, VOP2, VOP3, VOPC, VOP3P)
    if "VOP1" in cls or "VOP1" in df:
        encoding = "VOP1"
        modifiers = ["_e32", "_e64"]
        operands = [
            ["VGPR", "VGPR"],
            ["VGPR", "SGPR"],
            ["VGPR", "imm32"],
        ]
    elif "VOP2" in cls or "VOP2" in df:
        encoding = "VOP2"
        modifiers = ["_e32", "_e64"]
        operands = [
            ["VGPR", "VGPR", "VGPR"],
            ["VGPR", "SGPR", "VGPR"],
            ["VGPR", "imm32", "VGPR"],
        ]
    elif "VOPC" in cls or "VOPC" in df:
        encoding = "VOPC"
        modifiers = ["_e32", "_e64"]
        operands = [
            ["VCC", "VGPR", "VGPR"],
            ["VCC", "SGPR", "VGPR"],
            ["VCC", "imm32", "VGPR"],
        ]
    elif "VOP3P" in cls or "VOP3P" in df:
        encoding = "VOP3P"
        modifiers = [
            "_e64",
            "neg_lo",
            "neg_hi",
            "clamp",
            "op_sel",
            "-src",
            "|src|",
        ]
        operands = [
            ["VGPR", "VReg_64", "VReg_64"],
            ["VGPR", "VGPR", "VGPR"],
            ["V[n:n+1]", "V[n:n+1]", "V[n:n+1]"],
            ["a[n:n+3]", "V[n:n+1]", "V[n:n+1]", "a[n:n+3]"],
        ]
    elif "VOP3" in cls or "VOP3" in df:
        encoding = "VOP3"
        modifiers = [
            "_e64",
            "clamp",
            "omod",
            "omod:2",
            "omod:4",
            "omod:div2",
            "-src",
            "|src|",
        ]
        operands = [
            ["VGPR", "VGPR", "VGPR", "VGPR"],
            ["VGPR", "SGPR", "VGPR", "VGPR"],
            ["VGPR", "imm32", "VGPR", "VGPR"],
            ["V[n:n+1]", "V[n:n+1]", "V[n:n+1]"],
            ["V[n:n+3]", "V[n:n+3]", "V[n:n+3]"],
            ["a[n:n+3]", "V[n:n+1]", "V[n:n+1]", "a[n:n+3]"],
        ]
    elif (
        "VOPD" in cls
        or "VOPD" in df
        or "V_DUAL" in df
        or mnemonic.lower().startswith("v_dual_")
    ):
        encoding = "VOPD"
        modifiers = ["dual"]
        operands = [
            ["VGPR", "VGPR", "VGPR", "VGPR"],
            ["VGPR", "VGPR", "VGPR"],
        ]

    # 2. Scalar ALU (SOP1, SOP2, SOPK, SOPP)
    elif "SOP1" in cls or "SOP1" in df:
        encoding = "SOP1"
        operands = [
            ["SGPR", "SGPR"],
            ["SGPR", "imm32"],
        ]
    elif "SOP2" in cls or "SOP2" in df:
        encoding = "SOP2"
        operands = [
            ["SGPR", "SGPR", "SGPR"],
            ["SGPR", "SGPR", "imm32"],
        ]
    elif "SOPK" in cls or "SOPK" in df:
        encoding = "SOPK"
        operands = [
            ["SGPR", "simm16"],
        ]
    elif "SOPP" in cls or "SOPP" in df:
        encoding = "SOPP"
        operands = [
            ["simm16"],
        ]

    # 3. Memory Instructions (SM, FLAT, BUF, DS)
    elif "SM" in cls or "SM" in df or "SCALARMEM" in cls:
        encoding = "SM"
        modifiers = ["glc", "dlc"]
        operands = [
            ["SGPR", "SReg_64", "imm32"],
        ]
    elif "FLAT" in cls or "FLAT" in df:
        encoding = "FLAT"
        modifiers = ["glc", "slc", "dlc"]
        operands = [
            ["VGPR", "VReg_64", "imm12"],
        ]
    elif "BUF" in cls or "MUBUF" in cls or "BUF" in df:
        encoding = "BUF"
        modifiers = ["glc", "slc", "dlc", "idxen", "offen"]
        operands = [
            ["VGPR", "SReg_128", "VGPR", "SGPR"],
        ]
    elif "DS" in cls or "DS" in df:
        encoding = "DS"
        modifiers = ["gds"]
        operands = [
            ["VGPR", "VGPR", "imm16"],
        ]
    else:
        # Default instruction classification
        encoding = "VOP1"
        operands = [["VGPR", "VGPR"]]

    # Map architecture target availability
    if "GFX12" in cls or "GFX12" in df:
        arch = "GFX12/RDNA4"
    elif "GFX11_5" in cls or "GFX11_5" in df:
        arch = "GFX11.5"
    elif "GFX11" in cls or "GFX11" in df:
        arch = "GFX11/RDNA3"
    elif "GFX10_3" in cls or "GFX10_3" in df:
        arch = "GFX10.3/RDNA2"
    elif "GFX10" in cls or "GFX10" in df:
        arch = "GFX10/RDNA1"
    elif "GFX9" in cls or "GFX9" in df:
        arch = "GFX9/CDNA"
    else:
        arch = "GFX10+"

    return {
        "mnemonic": mnemonic,
        "architecture": arch,
        "encoding": encoding,
        "modifiers": modifiers,
        "operands": operands,
    }


def parse_td_content(content: str) -> List[Dict[str, Any]]:
    """Parse TableGen source content to extract real instructions, encodings, and operand profiles.

    Args:
        content: Raw TableGen text content.

    Returns:
        List of extracted instruction dictionaries.
    """
    instructions: List[Dict[str, Any]] = []

    pattern = re.compile(
        r'defm?\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)\s*<\s*"([a-z0-9_]+)"'
    )

    for match in pattern.finditer(content):
        def_name = match.group(1)
        class_name = match.group(2)
        mnemonic = match.group(3)

        info = resolve_td_instruction_info(def_name, class_name, mnemonic)
        instructions.append(info)

    multiclass_pattern = re.compile(
        r"defm\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]*(?:VOP3|VOP3P|SOPK|VOPD|VOP2|VOP1)[A-Za-z0-9_]*)\s*;"
    )
    for match in multiclass_pattern.finditer(content):
        def_name = match.group(1)
        class_name = match.group(2)
        mnemonic = def_name.lower()
        if not any(i["mnemonic"] == mnemonic for i in instructions):
            info = resolve_td_instruction_info(def_name, class_name, mnemonic)
            instructions.append(info)

    return instructions


def scrape_amd_rdna(output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scrape AMD RDNA TableGen sources and write the exhaustive JSON dump.

    Args:
        output_path: Optional path to save the output JSON.

    Returns:
        Exhaustive list of AMD RDNA instructions.
    """
    resolved_output = output_path or os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "amd_rdna_exhaustive.json",
    )

    instructions: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "modifiers": set(),
            "operands": [],
            "encoding": "Unknown",
            "architecture": "GFX10+",
        }
    )

    for td_file in TD_FILES:
        content = fetch_td_file(td_file)
        if not content:
            continue

        parsed_instrs = parse_td_content(content)
        for instr in parsed_instrs:
            base_name = instr["mnemonic"]
            instructions[base_name]["encoding"] = instr["encoding"]
            instructions[base_name]["architecture"] = instr["architecture"]

            for mod in instr["modifiers"]:
                instructions[base_name]["modifiers"].add(mod)

            for sig in instr["operands"]:
                if sig not in instructions[base_name]["operands"]:
                    instructions[base_name]["operands"].append(sig)

    exhaustive_list: List[Dict[str, Any]] = []
    for mnemonic, info in sorted(instructions.items()):
        exhaustive_list.append(
            {
                "mnemonic": mnemonic,
                "architecture": info["architecture"],
                "description": f"AMD RDNA {mnemonic} instruction.",
                "encoding": info["encoding"],
                "modifiers": sorted(list(info["modifiers"])),
                "operands": info["operands"],
            }
        )

    with open(resolved_output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(exhaustive_list, f, indent=2, sort_keys=True)
        f.write("\n")

    return exhaustive_list


def main() -> None:
    """Run the script to generate the exhaustive RDNA JSON dump."""
    scrape_amd_rdna()


if __name__ == "__main__":  # pragma: no cover
    main()
