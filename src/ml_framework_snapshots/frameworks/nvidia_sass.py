"""NVIDIA SASS (Assembly) Extractor.

Provides a static snapshot of NVIDIA SASS instructions with accurate operand modeling,
predicate guards, instruction modifiers, and multi-architecture targeting.
"""

from enum import Enum
import json
import os
from typing import Any, Dict, List, Optional

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ..models import (
    ExtendedGhostParam,
    ExtendedGhostRef,
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


def _load_exhaustive_sass() -> List[Dict[str, Any]]:
    """Loads the exhaustive NVIDIA SASS json dump."""
    json_path = os.path.join(os.path.dirname(__file__), "nvidia_sass_exhaustive.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
        return data


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
            ExtendedGhostRef(
                name=mnemonic,
                api_path=f"nvidia_sass.inst.{mnemonic}",
                kind="function",
                params=params,
                docstring=docstring,
                environment_tags=env_tags,
                overloads=overloads,
                domain_metadata=domain_metadata,
            )
        )

    return refs
