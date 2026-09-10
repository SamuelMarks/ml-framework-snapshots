"""NVIDIA SASS (Assembly) Extractor.

Provides a static snapshot of NVIDIA SASS instructions with accurate operand modeling,
predicate guards, instruction modifiers, and multi-architecture targeting.
"""

from enum import Enum
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ..models import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostIsaRef,
    IRParameterRole,
    OperandDirection,
)

ALL_SM_ARCHITECTURES: List[str] = [
    "sm_70",
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90",
    "sm_100",
]

DEFAULT_SASS_CONTROL_CODE_SCHEMA: Dict[str, Any] = {
    "latency_ticks": {
        "type": "integer",
        "description": "Cycles before destination register is ready for subsequent dependent instructions.",
        "default": 4,
    },
    "yield_flag": {
        "type": "boolean",
        "description": "Yield warp execution to warp scheduler on current cycle (Y flag).",
        "default": False,
    },
    "stall_count": {
        "type": "integer",
        "description": "Number of scheduler cycles to stall instruction issuing after this instruction (0-15).",
        "default": 1,
    },
    "read_barrier_mask": {
        "type": "integer",
        "description": "Read barrier mask (0-5 dependency barriers to wait on for register read completion).",
        "default": 0,
    },
    "write_barrier_mask": {
        "type": "integer",
        "description": "Write barrier mask (0-5 dependency barriers to set for register write completion).",
        "default": 0,
    },
    "register_reuse_flags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Register reuse cache flags (e.g. ['R0.reuse', 'R1.reuse']).",
        "default": [],
    },
}


class SASSInstructionFamily(str, Enum):
    """Instruction families in NVIDIA SASS architecture."""

    ARITHMETIC = "arithmetic"
    MEMORY = "memory"
    TENSOR = "tensor"
    CONTROL = "control"
    BARRIER = "barrier"
    SPECIAL = "special"


class SASSModifier(str, Enum):
    """Structured enum of standard NVIDIA SASS instruction modifiers."""

    SAT = ".SAT"
    RN = ".RN"
    RZ = ".RZ"
    RM = ".RM"
    RP = ".RP"
    FTZ = ".FTZ"
    STRONG = ".STRONG"
    CG = ".CG"
    CS = ".CS"
    LU = ".LU"
    CV = ".CV"


ROUNDING_MODIFIERS: Set[str] = {".RN", ".RZ", ".RM", ".RP"}
CACHE_MODIFIERS: Set[str] = {".CG", ".CS", ".LU", ".CV"}


def get_sass_instruction_family(mnemonic: str) -> str:
    """Determine the SASS instruction family based on mnemonic pattern.

    Args:
        mnemonic: The instruction mnemonic (e.g. 'FADD', 'LDG', 'HMMA16816').

    Returns:
        The instruction family string.
    """
    m = mnemonic.upper().strip()
    if any(m.startswith(p) for p in ("BAR", "DEPBAR", "SYNC")):
        return SASSInstructionFamily.BARRIER.value
    if any(
        m.startswith(p) for p in ("BRA", "RET", "EXIT", "WARPSYNC", "JMP", "CALL", "BR")
    ):
        return SASSInstructionFamily.CONTROL.value
    if any(m.startswith(p) for p in ("HMMA", "IMMA", "BMMA", "DMMA", "WGMMA")):
        return SASSInstructionFamily.TENSOR.value
    if any(
        m.startswith(p)
        for p in (
            "LD",
            "LDG",
            "LDS",
            "LDL",
            "LDC",
            "ST",
            "STG",
            "STS",
            "STL",
            "ATOM",
            "ATOMG",
            "ATOMS",
            "RED",
            "TMA",
            "LDGSTS",
            "LDSM",
        )
    ):
        return SASSInstructionFamily.MEMORY.value
    if any(m.startswith(p) for p in ("MUFU", "COS", "SIN", "EX2", "LG2", "RCP", "RSQ")):
        return SASSInstructionFamily.SPECIAL.value
    return SASSInstructionFamily.ARITHMETIC.value


def get_sass_latency_range(
    family: str, sm_arch: Optional[str] = None
) -> Tuple[int, int]:
    """Retrieve the legal cycle latency bounds for an instruction family on target architecture.

    Args:
        family: Instruction family name ('arithmetic', 'memory', 'tensor', 'special', 'control', 'barrier').
        sm_arch: Target SM architecture string (e.g. 'sm_80', 'sm_90').

    Returns:
        Tuple of (min_latency_cycles, max_latency_cycles).
    """
    arch = sm_arch.lower().strip() if sm_arch else "sm_80"
    is_modern = arch in ("sm_90", "sm_100")

    if family == SASSInstructionFamily.ARITHMETIC.value:
        return (3, 5) if is_modern else (4, 6)
    if family == SASSInstructionFamily.MEMORY.value:
        return (16, 256) if is_modern else (19, 256)
    if family == SASSInstructionFamily.TENSOR.value:
        return (16, 32) if is_modern else (8, 16)
    if family == SASSInstructionFamily.SPECIAL.value:
        return (14, 16)
    if family in (
        SASSInstructionFamily.CONTROL.value,
        SASSInstructionFamily.BARRIER.value,
    ):
        return (1, 4)
    return (1, 16)


def validate_sass_register(
    operand: str, role: str = "src", sm_arch: Optional[str] = None
) -> List[str]:
    """Validate a single SASS register specification against hardware alignment and indexing rules.

    Validates:
        - 32-bit registers (R0-R255, RZ)
        - 64-bit aligned register pairs (R0:R1, R2:R3 requiring even starting index and 0 <= n <= 254)
        - 128-bit aligned register quads (R0:R3, R4:R7 requiring modulo-4 starting index and 0 <= n <= 252)
        - Uniform registers (UR0-UR63, URZ) and pairs (UR0:UR1, requiring even starting index) on sm_75+
        - Predicate registers (P0-P7, PT) and Uniform Predicate registers (UP0-UP7, UPT) on sm_75+
        - Barrier register IDs (B0-B15)
        - Legality of register reuse flags (R*.reuse) exclusively on source operand slots

    Args:
        operand: Raw operand string (e.g. 'R0', 'R0:R1', 'R1:R2', 'UR0', 'P0', 'R0.reuse').
        role: Operand role ('dst', 'src0', 'src1', etc.).
        sm_arch: Target SM architecture (e.g. 'sm_70', 'sm_80', 'sm_90').

    Returns:
        List of error strings identified during register validation.
    """
    errors: List[str] = []
    cleaned = operand.strip()

    # Check for register reuse flag
    has_reuse = ".reuse" in cleaned.lower()
    if has_reuse:
        if role == "dst":
            errors.append(
                f"Register reuse flag '.reuse' is not permitted on destination operand '{operand}'."
            )
        cleaned = re.sub(r"\.reuse", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip predicate guard prefix '@'
    if cleaned.startswith("@"):
        cleaned = cleaned[1:].strip()

    # Check for memory operand wrapper like [R1] or [R2+0x4]
    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        reg_part = re.split(r"[+\-\*]", inner)[0].strip()
        if reg_part:
            return validate_sass_register(reg_part, role=role, sm_arch=sm_arch)
        return []

    # Check 64-bit / 128-bit vector register range: R<start>:R<end> or R[<start>:<end>]
    quad_pair_match = re.match(r"^R(\d+):R(\d+)$", cleaned, re.IGNORECASE) or re.match(
        r"^R\[(\d+):(\d+)\]$", cleaned, re.IGNORECASE
    )
    if quad_pair_match:
        start_idx = int(quad_pair_match.group(1))
        end_idx = int(quad_pair_match.group(2))
        span = end_idx - start_idx

        if start_idx > 255 or end_idx > 255:
            errors.append(
                f"Register index out of range in '{operand}': max register index is R255."
            )
            return errors

        if span == 1:
            # 64-bit pair
            if start_idx % 2 != 0:
                errors.append(
                    f"64-bit register pair '{operand}' must start on an even register index (got R{start_idx})."
                )
        elif span == 3:
            # 128-bit quad
            if start_idx % 4 != 0:
                errors.append(
                    f"128-bit register quad '{operand}' must start on a modulo-4 register index (got R{start_idx})."
                )
        else:
            errors.append(
                f"Invalid register span in '{operand}': expected pair (span 1) or quad (span 3), got span {span}."
            )
        return errors

    # Check uniform register pair: UR<start>:UR<end> or UR[<start>:<end>]
    ur_pair_match = re.match(r"^UR(\d+):UR(\d+)$", cleaned, re.IGNORECASE) or re.match(
        r"^UR\[(\d+):(\d+)\]$", cleaned, re.IGNORECASE
    )
    if ur_pair_match:
        if sm_arch and sm_arch == "sm_70":
            errors.append(
                f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )
        start_idx = int(ur_pair_match.group(1))
        end_idx = int(ur_pair_match.group(2))
        if start_idx > 63 or end_idx > 63:
            errors.append(
                f"Uniform register index out of range in '{operand}': max is UR63."
            )
        elif end_idx - start_idx != 1:
            errors.append(
                f"Invalid uniform register pair span in '{operand}': expected span 1, got {end_idx - start_idx}."
            )
        elif start_idx % 2 != 0:
            errors.append(
                f"Uniform register pair '{operand}' must start on an even register index (got UR{start_idx})."
            )
        return errors

    # Check 32-bit vector register: R0-R255, RZ
    if cleaned.upper() == "RZ":
        return errors
    r_match = re.match(r"^R(\d+)$", cleaned, re.IGNORECASE)
    if r_match:
        idx = int(r_match.group(1))
        if idx > 255:
            errors.append(
                f"32-bit vector register index R{idx} out of range (R0-R255)."
            )
        return errors

    # Check uniform register: UR0-UR63, URZ
    if cleaned.upper() == "URZ":
        if sm_arch and sm_arch == "sm_70":
            errors.append(
                f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )
        return errors
    ur_match = re.match(r"^UR(\d+)$", cleaned, re.IGNORECASE)
    if ur_match:
        if sm_arch and sm_arch == "sm_70":
            errors.append(
                f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )
        idx = int(ur_match.group(1))
        if idx > 63:
            errors.append(f"Uniform register index UR{idx} out of range (UR0-UR63).")
        return errors

    # Check predicate register: P0-P7, PT (and negated !P0-!P7, !PT)
    p_match = re.match(r"^!?P(\d+)$", cleaned, re.IGNORECASE)
    if p_match:
        idx = int(p_match.group(1))
        if idx > 7:
            errors.append(f"Predicate register index P{idx} out of range (P0-P7).")
        return errors
    if cleaned.upper() in ("PT", "!PT"):
        return errors

    # Check uniform predicate register: UP0-UP7, UPT (and negated !UP0-!UP7, !UPT)
    up_match = re.match(r"^!?UP(\d+)$", cleaned, re.IGNORECASE)
    if up_match:
        if sm_arch and sm_arch == "sm_70":
            errors.append(
                f"Uniform predicate registers (UP) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )
        idx = int(up_match.group(1))
        if idx > 7:
            errors.append(
                f"Uniform predicate register index UP{idx} out of range (UP0-UP7)."
            )
        return errors
    if cleaned.upper() in ("UPT", "!UPT"):
        if sm_arch and sm_arch == "sm_70":
            errors.append(
                f"Uniform predicate registers (UP) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )
        return errors

    # Check barrier register: B0-B15
    b_match = re.match(r"^B(\d+)$", cleaned, re.IGNORECASE)
    if b_match:
        idx = int(b_match.group(1))
        if idx > 15:
            errors.append(f"Barrier register index B{idx} out of range (B0-B15).")
        return errors

    return errors


def validate_sass_operand_directionality(
    operands: List[str], mnemonic: str
) -> List[str]:
    """Validate operand directionality to ensure read-only elements are not assigned to write slots.

    Args:
        operands: List of operand strings passed to instruction.
        mnemonic: The instruction mnemonic.

    Returns:
        List of directionality error messages.
    """
    errors: List[str] = []
    if not operands:
        return errors

    family = get_sass_instruction_family(mnemonic)
    # For branch, barrier, and store instructions, operand 0 is not a standard scalar write register
    is_store = mnemonic.upper().startswith("ST")
    is_control_or_barrier = family in (
        SASSInstructionFamily.CONTROL.value,
        SASSInstructionFamily.BARRIER.value,
    )

    if not is_store and not is_control_or_barrier:
        dst = operands[0].strip()
        # Immediate constants cannot be destination
        if re.match(r"^-?(0x[0-9a-fA-F]+|\d+(\.\d+)?)$", dst) or dst in ("I", "FI"):
            errors.append(
                f"Operand 0 is a destination (write) slot and cannot be an immediate: '{dst}'."
            )
        elif (
            "c[" in dst.lower()
            or "cx[" in dst.lower()
            or dst in ("c[bank][offset]", "cx[bank][offset]")
        ):
            errors.append(
                f"Operand 0 is a destination (write) slot and cannot be constant bank memory: '{dst}'."
            )

    c_bank_count = sum(
        1
        for op in operands
        if "c[" in op.lower()
        or "cx[" in op.lower()
        or op in ("c[bank][offset]", "cx[bank][offset]")
    )
    if c_bank_count > 1:
        errors.append(
            f"Hardware resource conflict in '{mnemonic}': SASS instructions permit at most 1 constant bank reference per cycle due to shared read ports (found {c_bank_count})."
        )

    return errors


def validate_sass_modifiers(mnemonic: str, modifiers: List[str]) -> List[str]:
    """Validate instruction modifier combinations for illegal conflicts.

    Checks:
        - Conflicting IEEE-754 rounding mode modifiers (.RN, .RZ, .RM, .RP)
        - Conflicting cache policy modifiers (.CG, .CS, .LU, .CV)
        - Saturation modifier (.SAT) legality per instruction family

    Args:
        mnemonic: The instruction mnemonic.
        modifiers: List of modifiers to validate.

    Returns:
        List of modifier conflict errors.
    """
    errors: List[str] = []
    norm_mods = [m if m.startswith(".") else f".{m}" for m in modifiers]

    # Check rounding mode conflicts
    rounding_present = [m for m in norm_mods if m in ROUNDING_MODIFIERS]
    if len(rounding_present) > 1:
        errors.append(
            f"Conflicting rounding mode modifiers in '{mnemonic}': {sorted(rounding_present)}."
        )

    # Check cache policy conflicts
    cache_present = [m for m in norm_mods if m in CACHE_MODIFIERS]
    if len(cache_present) > 1:
        errors.append(
            f"Conflicting cache policy modifiers in '{mnemonic}': {sorted(cache_present)}."
        )

    # Check .SAT validity
    if ".SAT" in norm_mods:
        family = get_sass_instruction_family(mnemonic)
        if family in (
            SASSInstructionFamily.MEMORY.value,
            SASSInstructionFamily.BARRIER.value,
            SASSInstructionFamily.CONTROL.value,
        ):
            errors.append(
                f"Saturation modifier '.SAT' is not supported on instruction family '{family}' for '{mnemonic}'."
            )

    return errors


def validate_sass_control_code(
    mnemonic: str, control_codes: Dict[str, Any], sm_arch: Optional[str] = None
) -> List[str]:
    """Validate control code attributes according to microarchitecture hardware constraints.

    Args:
        mnemonic: Instruction mnemonic.
        control_codes: Dictionary containing control code specifications.
        sm_arch: Target SM architecture string.

    Returns:
        List of control code error messages.
    """
    errors: List[str] = []
    family = get_sass_instruction_family(mnemonic)

    # Stall count: 0-15
    if "stall_count" in control_codes:
        stall = control_codes["stall_count"]
        if not isinstance(stall, int) or stall < 0 or stall > 15:
            errors.append(f"Stall count cycle {stall} is out of valid range (0-15).")

    # Dependency barrier scoreboard masks: 0-5 for sm_70-sm_89, 0-7 for sm_90+
    max_barrier = 7 if sm_arch in ("sm_90", "sm_100") else 5
    for b_field in ("read_barrier_mask", "write_barrier_mask"):
        if b_field in control_codes:
            mask = control_codes[b_field]
            if not isinstance(mask, int) or mask < 0 or mask > max_barrier:
                name = "Read" if "read" in b_field else "Write"
                errors.append(
                    f"{name} dependency barrier scoreboard {mask} is out of valid range (0-{max_barrier})."
                )

    # Latency ticks
    if "latency_ticks" in control_codes and sm_arch:
        latency = control_codes["latency_ticks"]
        min_ticks, max_ticks = get_sass_latency_range(family, sm_arch)
        if not isinstance(latency, int) or latency < min_ticks or latency > max_ticks:
            errors.append(
                f"Latency ticks {latency} out of bounds for {family} on {sm_arch}: expected between {min_ticks} and {max_ticks} cycles."
            )

    # Register reuse cache flags (.reuse) validation
    if "register_reuse_flags" in control_codes:
        reuse_flags = control_codes["register_reuse_flags"]
        if isinstance(reuse_flags, list):
            for rf in reuse_flags:
                rf_str = str(rf)
                if not rf_str.endswith(".reuse"):
                    errors.append(
                        f"Malformed register reuse flag '{rf_str}': expected 'R*.reuse'."
                    )
                reg_clean = rf_str.replace(".reuse", "")
                reg_errs = validate_sass_register(reg_clean, sm_arch=sm_arch)
                errors.extend(reg_errs)

    # Dual-issue restrictions
    if control_codes.get("dual_issue"):
        if family in (
            SASSInstructionFamily.TENSOR.value,
            SASSInstructionFamily.BARRIER.value,
        ):
            errors.append(
                f"Instruction '{mnemonic}' in family '{family}' does not support dual-issue execution."
            )

    # Hopper / Blackwell WGMMA and TMA asynchronous instruction validation
    if mnemonic.startswith("WGMMA") or "TMA" in mnemonic:
        if sm_arch and sm_arch not in ("sm_90", "sm_100"):
            errors.append(
                f"Instruction '{mnemonic}' requires Hopper or Blackwell architecture (sm_90+), got '{sm_arch}'."
            )

    return errors


def parse_sass_modifiers(modifiers: List[str]) -> List[str]:
    """Parse and normalize instruction modifiers into validated structured representations.

    Args:
        modifiers: List of raw modifier strings (e.g. ['SAT', 'RN'] or ['.SAT', '.RN']).

    Returns:
        List of normalized modifier strings.
    """
    normalized: List[str] = []
    valid_values = {m.value for m in SASSModifier}
    for mod in modifiers:
        prefix = mod if str(mod).startswith(".") else f".{mod}"
        if prefix in valid_values and prefix not in normalized:
            normalized.append(prefix)
        elif prefix not in normalized:
            normalized.append(prefix)
    return sorted(normalized)


def tokenize_sass_line(
    line: str,
) -> Tuple[Optional[str], str, List[str], List[str]]:
    """Tokenize a SASS assembly line into predicate, mnemonic, modifiers, and operands.

    Args:
        line: A single SASS assembly instruction line (e.g. '@P0 FADD.FTZ.RN R0, R1, R2;').

    Returns:
        A tuple of (predicate, base_mnemonic, modifiers, operands).
    """
    clean = line.strip().rstrip(";")
    if not clean:
        return None, "", [], []

    predicate = None
    pred_match = re.match(r"^(@!?U?P\d+|@!?PT|@!?UPT)\s+(.*)$", clean)
    if pred_match:
        predicate = pred_match.group(1)
        clean = pred_match.group(2).strip()

    tokens = clean.split(None, 1)
    inst_token = tokens[0]
    operands_part = tokens[1] if len(tokens) > 1 else ""

    subtokens = inst_token.split(".")
    base_mnemonic = subtokens[0].upper()
    modifiers = [f".{m}" for m in subtokens[1:] if m]

    operands = [op.strip() for op in operands_part.split(",") if op.strip()]
    return predicate, base_mnemonic, modifiers, operands


def resolve_sm_architectures(arch_spec: Optional[Any]) -> List[str]:
    """Resolve an SM architecture specification into discrete targeted SM architectures.

    Args:
        arch_spec: Architecture specification string (e.g. 'sm_80+', 'sm_70+') or list of discrete SM archs.

    Returns:
        List of discrete SM architectures targeted (sm_70 through sm_100).
    """
    if not arch_spec:
        return list(ALL_SM_ARCHITECTURES)
    if isinstance(arch_spec, list):
        return [a for a in arch_spec if a in ALL_SM_ARCHITECTURES]
    clean = str(arch_spec).lower().strip()
    if clean.endswith("+"):
        base = clean[:-1]
        if base in ALL_SM_ARCHITECTURES:
            idx = ALL_SM_ARCHITECTURES.index(base)
            return ALL_SM_ARCHITECTURES[idx:]
    if clean in ALL_SM_ARCHITECTURES:
        return [clean]
    return list(ALL_SM_ARCHITECTURES)


def normalize_sass_operand_type(raw_op: str) -> str:
    """Normalize raw operand names into canonical SASS operand categories.

    Maps:
        'R' -> 32-bit vector register
        'UR' -> Uniform register
        'P' -> Predicate register
        'I' -> Integer immediate
        'FI' -> Float immediate
        'ADDR' -> Memory address
        'c[x][y]' -> Constant bank memory c[bank][offset]

    Args:
        raw_op: Raw operand string from ISA metadata.

    Returns:
        Canonical operand category string.
    """
    cleaned = raw_op.strip()
    if cleaned in ("R", "RegOperand"):
        return "R"
    if cleaned in ("UR", "UniformRegOperand"):
        return "UR"
    if cleaned in ("P", "PredicateOperand"):
        return "P"
    if cleaned in ("I", "IntIMMOperand"):
        return "I"
    if cleaned in ("FI", "FloatIMMOperand"):
        return "FI"
    if cleaned in ("ADDR", "AddressOperand"):
        return "ADDR"
    if "cx[" in cleaned:
        return "cx[bank][offset]"
    if "c[" in cleaned:
        return "c[bank][offset]"
    return cleaned


def build_structured_sass_operands(
    operands: List[str], mnemonic: str
) -> List[Dict[str, Any]]:
    """Build structured operand records detailing roles, register classes, and immediate constraints.

    Args:
        operands: Raw list of operand tokens (e.g. ['R0', 'R1', 'c[0x0][0x10]']).
        mnemonic: Instruction mnemonic (e.g. 'FADD', 'STG', 'BRA').

    Returns:
        List of structured operand dictionaries detailing index, role, register_classes, and imm limits.
    """
    structured: List[Dict[str, Any]] = []
    is_store = mnemonic.startswith("ST")
    is_branch = mnemonic in ("BRA", "BRX", "JMP", "JMX", "CALL", "RET")
    is_barrier = mnemonic in ("DEPBAR", "BAR", "SYNC", "WARPSYNC")

    for idx, op in enumerate(operands):
        role = (
            "dest"
            if idx == 0 and not (is_store or is_branch or is_barrier)
            else f"src{idx - 1 if idx > 0 else 0}"
        )
        if is_store and idx == 0:
            role = "address"
        elif is_branch:
            role = "target"
        elif is_barrier:
            role = "barrier"

        reg_classes: List[str] = []
        op_u = op.upper()
        if "UR" in op_u:
            reg_classes.append("UR")
        if "R" in op_u and "UR" not in op_u and "BAR" not in op_u:
            reg_classes.append("R")
        if "P" in op_u and "UP" not in op_u and "DEPBAR" not in op_u:
            reg_classes.append("P")
        if "UP" in op_u:
            reg_classes.append("UP")
        if "B" in op_u:
            reg_classes.append("B")
        if "C[" in op_u or "CX[" in op_u:
            reg_classes.append("CBANK")

        imm_type: Optional[str] = None
        if re.match(r"^-?\d+\.\d+", op):
            imm_type = "float32"
        elif re.match(r"^-?(0x[0-9a-fA-F]+|\d+)$", op):
            imm_type = "int32"

        rec: Dict[str, Any] = {
            "index": idx,
            "raw_token": op,
            "role": role,
            "register_classes": reg_classes,
        }
        if imm_type:
            rec["immediate_type"] = imm_type
        structured.append(rec)
    return structured


def _load_exhaustive_sass() -> List[Dict[str, Any]]:
    """Loads the exhaustive NVIDIA SASS json dump.

    Returns:
        The loaded JSON list of SASS instruction dictionaries.
    """
    json_path = os.path.join(os.path.dirname(__file__), "nvidia_sass_exhaustive.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)
        if isinstance(data, dict):
            if "categories" in data:
                all_ops: List[Dict[str, Any]] = []
                for cat_ops in data["categories"].values():
                    if isinstance(cat_ops, list):
                        all_ops.extend(cat_ops)
                return all_ops
            return cast(List[Dict[str, Any]], data.get("instructions", []))
        return cast(List[Dict[str, Any]], data)


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the NVIDIA SASS API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs: List[GhostRef] = []

    sass_data = _load_exhaustive_sass()

    for inst_data in sass_data:
        mnemonic = str(inst_data["mnemonic"])
        base_desc = str(
            inst_data.get("description", f"NVIDIA SASS {mnemonic} instruction.")
        )
        raw_modifiers = inst_data.get("modifiers", [])
        parsed_modifiers = parse_sass_modifiers(raw_modifiers)

        arch = inst_data.get("architecture")
        valid_archs = resolve_sm_architectures(arch)
        env_tags = ["cuda"] + valid_archs

        docstring = (
            f"NVIDIA SASS {mnemonic} instruction.\n"
            f"Description: {base_desc}\n"
            f"Modifiers: {', '.join(parsed_modifiers) if parsed_modifiers else 'None'}\n"
            f"Target Architectures: {', '.join(valid_archs)}\n"
            f"Predicate Guard: Supported (@P0, @!P1)"
        )

        operands_list: List[List[str]] = inst_data.get("operands", [])

        # Find the maximum length signature for the canonical representation
        max_sig: List[str] = []
        for sig in operands_list:
            if len(sig) > len(max_sig):
                max_sig = sig

        params: List[ExtendedGhostParam] = []
        for i, raw_op in enumerate(max_sig):
            op_type = normalize_sass_operand_type(raw_op)
            role = "dst" if i == 0 else f"src{i - 1}"
            direction = OperandDirection.WRITE if i == 0 else OperandDirection.READ
            params.append(
                ExtendedGhostParam(
                    name=f"op{i}",
                    kind="POSITIONAL_ONLY",
                    annotation=op_type,
                    standardized_name=role,
                    description=f"Operand {i} ({role}) of type {op_type}",
                    direction=direction,
                    role=IRParameterRole.OPERAND,
                )
            )

        domain_metadata = {
            "architecture": arch,
            "valid_architectures": valid_archs,
            "modifiers": parsed_modifiers,
            "predicate_supported": True,
            "operand_signatures": operands_list,
            "control_code_schema": DEFAULT_SASS_CONTROL_CODE_SCHEMA,
        }

        # Build overloads for all operand variations
        overloads: List[ExtendedGhostRef] = []
        for sig_idx, sig in enumerate(operands_list):
            overload_params: List[ExtendedGhostParam] = []
            for i, raw_op in enumerate(sig):
                op_type = normalize_sass_operand_type(raw_op)
                role = "dst" if i == 0 else f"src{i - 1}"
                direction = OperandDirection.WRITE if i == 0 else OperandDirection.READ
                overload_params.append(
                    ExtendedGhostParam(
                        name=f"op{i}",
                        kind="POSITIONAL_ONLY",
                        annotation=op_type,
                        standardized_name=role,
                        description=f"Operand {i} ({role}) of type {op_type}",
                        direction=direction,
                        role=IRParameterRole.OPERAND,
                    )
                )
            overloads.append(
                ExtendedGhostRef(
                    name=mnemonic,
                    api_path=f"nvidia_sass.inst.{mnemonic}",
                    kind="function",
                    params=overload_params,
                    docstring=f"Overload {sig_idx} for {mnemonic}: {', '.join(sig)}",
                    environment_tags=env_tags,
                    domain_metadata=domain_metadata,
                )
            )

        refs.append(
            GhostIsaRef(
                name=mnemonic,
                api_path=f"nvidia_sass.inst.{mnemonic}",
                kind="function",
                params=params,
                docstring=docstring,
                environment_tags=env_tags,
                overloads=overloads,
                domain_metadata=domain_metadata,
                predicate_guards=["@P0", "@!P0", "@P1", "@!P1", "@PT"],
                instruction_modifiers=parsed_modifiers,
                supported_architectures=valid_archs,
                control_codes=DEFAULT_SASS_CONTROL_CODE_SCHEMA,
                structured_operands=build_structured_sass_operands(max_sig, mnemonic),
            )
        )

    return refs
