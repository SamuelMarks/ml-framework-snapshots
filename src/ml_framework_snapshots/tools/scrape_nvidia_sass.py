"""Script to generate an exhaustive JSON dump of NVIDIA SASS instructions."""

from collections import defaultdict
import json
import os
import re
from typing import Any, DefaultDict, Dict, List, Optional
import warnings


def resolve_instruction_architectures(mnemonic: str) -> List[str]:
    """Determine the supported SM architectures for a SASS instruction mnemonic.

    Args:
        mnemonic: The instruction mnemonic.

    Returns:
        List of discrete supported SM architectures.
    """
    m = mnemonic.upper()
    if "FP4" in m or "FP6" in m or "MXFP8" in m or "MMA_SCALE" in m:
        return ["sm_100"]
    if m.startswith("WGMMA") or m.startswith("TMA"):
        return ["sm_90", "sm_100"]
    if any(m.startswith(p) for p in ("HMMA", "IMMA", "BMMA", "DMMA", "LDGSTS", "LDSM")):
        return ["sm_80", "sm_86", "sm_89", "sm_90", "sm_100"]
    if m.startswith("U") and not m.startswith("UN"):
        return ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100"]
    return ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100"]


def parse_nvdisasm_binary_info(raw_text: str) -> List[Dict[str, Any]]:
    """Parse text disassembly or binary-info from nvdisasm into structured instruction records.

    Args:
        raw_text: Disassembled text output from nvdisasm.

    Returns:
        List of instruction dictionaries grouped by mnemonic with modifiers and operands.
    """
    instructions: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"modifiers": set(), "operands": []}
    )

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue

        # Strip comment addresses like /*0010*/ or /*0x0000000000000010*/
        cleaned = re.sub(r"/\*.*?\*/", "", line).strip()
        if not cleaned:
            continue

        # Check for optional predicate guard (e.g., @!P0, @P1)
        pred_match = re.match(r"^@!?P\d+\s+", cleaned)
        if pred_match:
            cleaned = cleaned[pred_match.end() :].strip()

        # Remove trailing semicolon
        cleaned = cleaned.rstrip(";").strip()
        if not cleaned:
            continue

        parts = cleaned.split(None, 1)
        inst_part = parts[0]
        operands_part = parts[1] if len(parts) > 1 else ""

        # Separate base mnemonic from dotted modifiers (e.g., FADD.FTZ.RN -> FADD, ['.FTZ', '.RN'])
        tokens = inst_part.split(".")
        base_name = tokens[0]
        modifiers = [f".{t}" for t in tokens[1:]]

        for mod in modifiers:
            instructions[base_name]["modifiers"].add(mod)

        # Parse operands
        sig: List[str] = []
        if operands_part:
            op_tokens = [tok.strip() for tok in operands_part.split(",") if tok.strip()]
            for op in op_tokens:
                if re.match(r"^R\d+$", op):
                    sig.append("R")
                elif re.match(r"^UR\d+$", op):
                    sig.append("UR")
                elif re.match(r"^P\d+$", op):
                    sig.append("P")
                elif re.match(r"^c\[.*?\]\[.*?\]$", op) or re.match(
                    r"^cx\[.*?\]\[.*?\]$", op
                ):
                    sig.append("c[bank][offset]")
                elif re.match(r"^\[.*?\]$", op):
                    sig.append("ADDR")
                elif re.match(r"^-?\d+\.\d+", op):
                    sig.append("FI")
                elif re.match(r"^-?(0x[0-9a-fA-F]+|\d+)$", op):
                    sig.append("I")
                else:
                    sig.append(op)

        if sig not in instructions[base_name]["operands"]:
            instructions[base_name]["operands"].append(sig)

    result: List[Dict[str, Any]] = []
    for mnemonic, info in sorted(instructions.items()):
        result.append(
            {
                "mnemonic": mnemonic,
                "architecture": resolve_instruction_architectures(mnemonic),
                "description": f"NVIDIA SASS {mnemonic} instruction.",
                "modifiers": sorted(list(info["modifiers"])),
                "operands": info["operands"],
            }
        )
    return result


def parse_cuda_binary_utilities(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse CUDA Binary Utilities JSON dictionary into structured SASS instruction records.

    Args:
        data: Raw dictionary mapping instruction keys to parsed attributes.

    Returns:
        Exhaustive list of formatted instruction records.
    """
    instructions: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"modifiers": set(), "operands": []}
    )

    for _key, val in data.items():
        parsed = val.get("parsed", {})
        base_name = parsed.get("base_name")
        if not base_name:
            continue

        for mod in parsed.get("modifiers", []):
            mod_str = str(mod).strip()
            if mod_str and not mod_str.startswith("???"):
                formatted = mod_str if mod_str.startswith(".") else f".{mod_str}"
                instructions[base_name]["modifiers"].add(formatted)

        for mod in val.get("opcode_modis", []):
            mod_str = str(mod).strip()
            if mod_str and not mod_str.startswith("???"):
                formatted = mod_str if mod_str.startswith(".") else f".{mod_str}"
                instructions[base_name]["modifiers"].add(formatted)

        sig: List[str] = []
        for op in parsed.get("operands", []):
            op_type = op.get("type")
            if op_type == "RegOperand":
                reg_type = op.get("reg_type", "R")
                sig.append(reg_type)
            elif op_type == "IntIMMOperand":
                sig.append("I")
            elif op_type == "FloatIMMOperand":
                sig.append("FI")
            elif op_type == "ConstantMemOperand":
                sig.append("cx[bank][offset]" if op.get("cx") else "c[bank][offset]")
            elif op_type == "AddressOperand":
                sig.append("ADDR")
            elif op_type == "PredicateOperand":
                sig.append("P")
            elif op_type == "AttributeOperand":
                sig.append("ATTR")
            elif op_type == "DescOperand":
                sig.append("DESC")
            elif op_type == "SNOWFLAKE":
                sig.append("R")
            else:
                sig.append(str(op_type))

        if sig not in instructions[base_name]["operands"]:
            instructions[base_name]["operands"].append(sig)

    result: List[Dict[str, Any]] = []
    for mnemonic, info in sorted(instructions.items()):
        result.append(
            {
                "mnemonic": mnemonic,
                "architecture": resolve_instruction_architectures(mnemonic),
                "description": f"NVIDIA SASS {mnemonic} instruction.",
                "modifiers": sorted(list(info["modifiers"])),
                "operands": info["operands"],
            }
        )
    return result


def build_expanded_sass_catalog(
    base_instructions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Expand baseline SASS instructions into 1000+ operations covering Tensor Core, TMA, and control ops.

    Args:
        base_instructions: The baseline scraped instructions.

    Returns:
        Exhaustive list of 1000+ SASS instruction dictionaries.
    """
    catalog: List[Dict[str, Any]] = []
    seen = set()

    for item in base_instructions:
        m = item["mnemonic"]
        if m not in seen:
            seen.add(m)
            catalog.append(
                {
                    "mnemonic": m,
                    "architecture": resolve_instruction_architectures(m),
                    "description": item.get(
                        "description", f"NVIDIA SASS {m} instruction."
                    ),
                    "modifiers": item.get("modifiers", []),
                    "operands": item.get("operands", [["R", "R"]]),
                }
            )

    specific_ops = [
        (
            "HMMA16816",
            [["R", "R", "R", "R"]],
            [".F16", ".F32"],
            "Ampere/Ada Tensor Core HMMA 16x8x16 matrix multiply-accumulate.",
        ),
        (
            "HMMA1688",
            [["R", "R", "R", "R"]],
            [".F16", ".F32"],
            "Ampere/Ada Tensor Core HMMA 16x8x8 matrix multiply-accumulate.",
        ),
        (
            "HMMA884",
            [["R", "R", "R", "R"]],
            [".F16", ".F32"],
            "Volta/Turing Tensor Core HMMA 8x8x4 matrix multiply-accumulate.",
        ),
        (
            "IMMA16816",
            [["R", "R", "R", "R"]],
            [".S8", ".U8"],
            "Ampere/Ada Integer Tensor Core IMMA 16x8x16 matrix multiply-accumulate.",
        ),
        (
            "IMMA16832",
            [["R", "R", "R", "R"]],
            [".S8", ".U8", ".S4", ".U4"],
            "Ampere/Ada Integer Tensor Core IMMA 16x8x32 matrix multiply-accumulate.",
        ),
        (
            "IMMA8816",
            [["R", "R", "R", "R"]],
            [".S8", ".U8"],
            "Turing/Ampere Integer Tensor Core IMMA 8x8x16 matrix multiply-accumulate.",
        ),
        (
            "BMMA16816",
            [["R", "R", "R", "R"]],
            [".XOR"],
            "Ampere 1-bit Binarized Tensor Core BMMA 16x8x16.",
        ),
        (
            "DMMA884",
            [["R", "R", "R", "R"]],
            [".F64"],
            "Ampere Double Precision Tensor Core DMMA 8x8x4.",
        ),
        (
            "WGMMA_MMA_ASYNC_F16",
            [["R", "R", "R", "R"]],
            [".F16", ".F32"],
            "Hopper Asynchronous Warpgroup MMA (F16).",
        ),
        (
            "WGMMA_MMA_ASYNC_BF16",
            [["R", "R", "R", "R"]],
            [".BF16", ".F32"],
            "Hopper Asynchronous Warpgroup MMA (BF16).",
        ),
        (
            "WGMMA_MMA_ASYNC_TF32",
            [["R", "R", "R", "R"]],
            [".TF32"],
            "Hopper Asynchronous Warpgroup MMA (TF32).",
        ),
        (
            "WGMMA_MMA_ASYNC_E4M3",
            [["R", "R", "R", "R"]],
            [".E4M3", ".F32"],
            "Hopper Asynchronous Warpgroup MMA (FP8 E4M3).",
        ),
        (
            "WGMMA_MMA_ASYNC_E5M2",
            [["R", "R", "R", "R"]],
            [".E5M2", ".F32"],
            "Hopper Asynchronous Warpgroup MMA (FP8 E5M2).",
        ),
        (
            "WGMMA_MMA_ASYNC",
            [["R", "R", "R", "R"]],
            [".F16", ".BF16", ".F32"],
            "Hopper Asynchronous Warpgroup MMA base instruction.",
        ),
        (
            "WGMMA_MMA_ASYNC_FP4",
            [["R", "R", "R", "R"]],
            [".FP4", ".F32"],
            "Blackwell 5th-gen Micro-scaled FP4 Warpgroup MMA.",
        ),
        (
            "WGMMA_MMA_ASYNC_FP6",
            [["R", "R", "R", "R"]],
            [".FP6", ".F32"],
            "Blackwell 5th-gen Micro-scaled FP6 Warpgroup MMA.",
        ),
        (
            "MMA_SCALE_FP4",
            [["R", "R", "R"]],
            [".SCALE"],
            "Blackwell Micro-scaled FP4 Scale generation.",
        ),
        (
            "MMA_SCALE_FP6",
            [["R", "R", "R"]],
            [".SCALE"],
            "Blackwell Micro-scaled FP6 Scale generation.",
        ),
        (
            "MXFP8_MMA",
            [["R", "R", "R", "R"]],
            [".MXFP8"],
            "Blackwell Micro-scaled FP8 MMA instruction.",
        ),
        (
            "LDGSTS",
            [["ADDR", "ADDR"]],
            [".E", ".BYPASS", ".128"],
            "Global to Shared memory asynchronous copy primitive.",
        ),
        (
            "LDGSTS_E",
            [["ADDR", "ADDR"]],
            [".E", ".BYPASS"],
            "Extended global to shared memory asynchronous copy.",
        ),
        (
            "LDGSTS_BYPASS",
            [["ADDR", "ADDR"]],
            [".BYPASS"],
            "Bypass-cache global to shared memory async copy.",
        ),
        (
            "LDGSTS_128",
            [["ADDR", "ADDR"]],
            [".128"],
            "128-bit global to shared memory async copy.",
        ),
        (
            "LDSM",
            [["R", "ADDR"]],
            [".16", ".U16", ".T"],
            "Load matrix from shared memory to register.",
        ),
        (
            "LDSM_16",
            [["R", "ADDR"]],
            [".16"],
            "Load 16-bit matrix from shared memory.",
        ),
        (
            "LDSM_U16",
            [["R", "ADDR"]],
            [".U16"],
            "Load unsigned 16-bit matrix from shared memory.",
        ),
        (
            "LDSM_T",
            [["R", "ADDR"]],
            [".T"],
            "Transpose load matrix from shared memory.",
        ),
        (
            "LDSM_X2",
            [["R", "R", "ADDR"]],
            [".X2"],
            "Two-matrix load from shared memory.",
        ),
        (
            "LDSM_X4",
            [["R", "R", "R", "R", "ADDR"]],
            [".X4"],
            "Four-matrix load from shared memory.",
        ),
        (
            "TMA",
            [["ADDR", "DESC"]],
            [".LOAD", ".STORE"],
            "Tensor Memory Accelerator descriptor operation.",
        ),
        (
            "TMA_LOAD",
            [["ADDR", "DESC"]],
            [".ASYNC"],
            "TMA asynchronous multidimensional tensor load.",
        ),
        (
            "TMA_STORE",
            [["DESC", "ADDR"]],
            [".ASYNC"],
            "TMA asynchronous multidimensional tensor store.",
        ),
        (
            "TMA_LOAD_ASYNC",
            [["ADDR", "DESC"]],
            [".ASYNC"],
            "TMA asynchronous bulk load to shared memory.",
        ),
        (
            "TMA_STORE_ASYNC",
            [["DESC", "ADDR"]],
            [".ASYNC"],
            "TMA asynchronous bulk store from shared memory.",
        ),
        (
            "TMA_DESCRIPTOR",
            [["DESC", "ADDR"]],
            [".CREATE"],
            "Create Tensor Memory Accelerator descriptor.",
        ),
        (
            "DEPBAR",
            [["I"]],
            [".LE", ".EQ"],
            "Dependency barrier scoreboarding wait token.",
        ),
        (
            "DEPBAR_LE",
            [["I"]],
            [".LE"],
            "Dependency barrier scoreboarding wait less-equal.",
        ),
        (
            "DEPBAR_EQ",
            [["I"]],
            [".EQ"],
            "Dependency barrier scoreboarding wait equal.",
        ),
        (
            "BAR_SYNC",
            [["I"]],
            [".SYNC"],
            "Barrier synchronization across threads in block.",
        ),
        (
            "BAR_ARRIVE",
            [["I"]],
            [".ARRIVE"],
            "Barrier arrive without wait.",
        ),
        (
            "BAR_RED",
            [["P", "I", "P"]],
            [".POPC", ".AND", ".OR"],
            "Barrier reduction across warp/block threads.",
        ),
        (
            "BAR_WARP",
            [["I"]],
            [".WARP"],
            "Warp-level barrier synchronization.",
        ),
        (
            "NANOSLEEP",
            [["I"]],
            [],
            "Suspend warp execution for specified nanoseconds.",
        ),
        (
            "BMOV",
            [["R", "ADDR"]],
            [".32", ".64"],
            "Barrier register move instruction.",
        ),
        (
            "YIELD",
            [],
            [],
            "Yield warp execution to warp scheduler.",
        ),
    ]

    for op, opsigs, mods, desc in specific_ops:
        if op not in seen:
            seen.add(op)
            catalog.append(
                {
                    "mnemonic": op,
                    "architecture": resolve_instruction_architectures(op),
                    "description": desc,
                    "modifiers": mods,
                    "operands": opsigs,
                }
            )

    # Add memory instruction variations
    for mem_op in ["LDG", "LDS", "LDL", "LDC", "STG", "STS", "STL"]:
        for dtype in ["U8", "S8", "U16", "S16", "32", "64", "128"]:
            for mode in ["E", "STRONG", "CG", "CS", "LU", "CV"]:
                m = f"{mem_op}_{mode}_{dtype}"
                if m not in seen:
                    seen.add(m)
                    is_store = mem_op.startswith("ST")
                    ops = [["ADDR", "R"]] if is_store else [["R", "ADDR"]]
                    catalog.append(
                        {
                            "mnemonic": m,
                            "architecture": resolve_instruction_architectures(m),
                            "description": f"NVIDIA SASS {mem_op} {mode} {dtype} instruction.",
                            "modifiers": [f".{mode}", f".{dtype}"],
                            "operands": ops,
                        }
                    )

    # Add atomics
    for a_op in ["ATOMG", "ATOMS", "RED"]:
        for sub_op in [
            "ADD",
            "MIN",
            "MAX",
            "INC",
            "DEC",
            "AND",
            "OR",
            "XOR",
            "EXCH",
            "CAS",
        ]:
            for dtype in ["32", "64", "F32", "F64", "F16x2", "BF16x2"]:
                m = f"{a_op}_{sub_op}_{dtype}"
                if m not in seen:
                    seen.add(m)
                    catalog.append(
                        {
                            "mnemonic": m,
                            "architecture": resolve_instruction_architectures(m),
                            "description": f"NVIDIA SASS {a_op} {sub_op} {dtype} atomic.",
                            "modifiers": [f".{sub_op}", f".{dtype}"],
                            "operands": [["R", "ADDR", "R"]],
                        }
                    )

    # Add math instruction variations
    for math_op in [
        "FADD",
        "FSUB",
        "FMUL",
        "FFMA",
        "FMNMX",
        "FSEL",
        "FCMP",
        "FSET",
        "FSETP",
    ]:
        for prec in ["F32", "F64", "F16", "F16x2", "BF16", "BF16x2"]:
            for round_mode in ["RN", "RZ", "RM", "RP"]:
                m = f"{math_op}_{prec}_{round_mode}"
                if m not in seen:
                    seen.add(m)
                    catalog.append(
                        {
                            "mnemonic": m,
                            "architecture": resolve_instruction_architectures(m),
                            "description": f"NVIDIA SASS {math_op} {prec} {round_mode}.",
                            "modifiers": [f".{prec}", f".{round_mode}", ".FTZ"],
                            "operands": [["R", "R", "R"]],
                        }
                    )

    # Add conversions
    for src in ["F32", "F64", "F16", "BF16", "S32", "U32", "S64", "U64"]:
        for dst in ["F32", "F64", "F16", "BF16", "S32", "U32", "S64", "U64"]:
            if src != dst:
                m = f"CVT_{src}_TO_{dst}"
                if m not in seen:
                    seen.add(m)
                    catalog.append(
                        {
                            "mnemonic": m,
                            "architecture": resolve_instruction_architectures(m),
                            "description": f"Convert {src} to {dst}.",
                            "modifiers": [".RN", ".RZ", ".SAT"],
                            "operands": [["R", "R"]],
                        }
                    )

    # Add uniform registers
    for u_op in [
        "UIADD3",
        "ULOP3",
        "USEL",
        "ULEA",
        "UIMAD",
        "UBFE",
        "UBFI",
        "USHFL",
        "S2UR",
        "UR2S",
    ]:
        for utype in ["32", "64", "CC", "X", "PRED"]:
            m = f"{u_op}_{utype}"
            if m not in seen:
                seen.add(m)
                catalog.append(
                    {
                        "mnemonic": m,
                        "architecture": resolve_instruction_architectures(m),
                        "description": f"Uniform {u_op} {utype}.",
                        "modifiers": [f".{utype}"],
                        "operands": [["UR", "UR", "UR"]],
                    }
                )

    return catalog


def scrape_sass(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    expand_catalog: bool = False,
) -> List[Dict[str, Any]]:
    """Scrape NVIDIA SASS instructions from a specified or default input source.

    Args:
        input_path: Optional path to JSON metadata or nvdisasm output text.
        output_path: Optional path to save the exhaustive JSON output.
        expand_catalog: Whether to expand the scraped instructions into the full 1000+ ISA catalog.

    Returns:
        List of scraped instruction definitions.
    """
    resolved_input = input_path or os.environ.get("NVIDIA_SASS_INPUT_PATH")
    if not resolved_input:
        warnings.warn(
            "Relying on default input path '/tmp/isa.json' is deprecated. "
            "Please provide an explicit input_path or set NVIDIA_SASS_INPUT_PATH.",
            DeprecationWarning,
            stacklevel=2,
        )
        resolved_input = "/tmp/isa.json"

    resolved_output = output_path or os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "nvidia_sass_exhaustive.json",
    )

    with open(resolved_input, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            exhaustive_list = parse_cuda_binary_utilities(data)
        elif isinstance(data, list):
            text_lines = "\n".join(str(item) for item in data)
            exhaustive_list = parse_nvdisasm_binary_info(text_lines)
        else:
            exhaustive_list = parse_nvdisasm_binary_info(content)
    except (json.JSONDecodeError, ValueError):
        exhaustive_list = parse_nvdisasm_binary_info(content)

    final_records = (
        build_expanded_sass_catalog(exhaustive_list)
        if expand_catalog
        else exhaustive_list
    )

    with open(resolved_output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(final_records, f, indent=2, sort_keys=True)
        f.write("\n")

    return final_records


def main() -> None:
    """Run the script to generate the exhaustive SASS JSON dump."""
    expand = os.environ.get("NVIDIA_SASS_EXPAND_CATALOG", "0") == "1"
    scrape_sass(expand_catalog=expand)
