"""NVIDIA PTX ISA ground truth framework extractor.

Extracts, validates, and normalizes NVIDIA Parallel Thread Execution (PTX)
assembly instructions, types, state spaces, vector widths, and operands.
Provides ground-truth verification against LLM hallucinations when generating
CUDA C++, Triton, or TVM compiled kernels.
"""

from enum import Enum
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, cast

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    SemanticTier,
)
from ..models import (
    ExtendedGhostParam,
    IRParameterRole,
)


class PTXStateSpace(str, Enum):
    """Memory state spaces in NVIDIA PTX architecture."""

    REG = ".reg"
    SREG = ".sreg"
    CONST = ".const"
    GLOBAL = ".global"
    LOCAL = ".local"
    PARAM = ".param"
    SHARED = ".shared"


class PTXType(str, Enum):
    """Primitive scalar and vector data types in NVIDIA PTX ISA."""

    # Untyped bit representations
    B8 = ".b8"
    B16 = ".b16"
    B32 = ".b32"
    B64 = ".b64"
    B128 = ".b128"

    # Unsigned integers
    U8 = ".u8"
    U16 = ".u16"
    U32 = ".u32"
    U64 = ".u64"

    # Signed integers
    S8 = ".s8"
    S16 = ".s16"
    S32 = ".s32"
    S64 = ".s64"

    # Floating point representations
    F16 = ".f16"
    F16X2 = ".f16x2"
    BF16 = ".bf16"
    BF16X2 = ".bf16x2"
    TF32 = ".tf32"
    F32 = ".f32"
    F64 = ".f64"
    E4M3 = ".e4m3"
    E5M2 = ".e5m2"

    # Predicate
    PRED = ".pred"


class PTXVectorWidth(str, Enum):
    """Vector packing widths supported in PTX instructions."""

    V2 = ".v2"
    V4 = ".v4"


class PTXScope(str, Enum):
    """Memory scope qualifiers in NVIDIA PTX architecture."""

    CTA = ".cta"
    CLUSTER = ".cluster"
    GPU = ".gpu"
    SYS = ".sys"


class PTXInstructionCategory(str, Enum):
    """Functional categories of PTX instructions."""

    ARITHMETIC = "arithmetic"
    LOGICAL = "logical"
    COMPARISON = "comparison"
    MOVEMENT = "movement"
    MEMORY = "memory"
    BARRIER = "barrier"
    TENSOR = "tensor"


ALL_PTX_TYPES: Set[str] = {t.value for t in PTXType}
ALL_PTX_STATE_SPACES: Set[str] = {s.value for s in PTXStateSpace}
ALL_PTX_SCOPES: Set[str] = {s.value for s in PTXScope}
ALL_PTX_VECTOR_WIDTHS: Set[str] = {v.value for v in PTXVectorWidth}


def validate_ptx_type(type_str: str) -> List[str]:
    """Validate whether a given type string is a legal PTX data type.

    Args:
        type_str: Type qualifier to validate (e.g. '.f32', '.u64', 'f32').

    Returns:
        List of validation error messages, or empty list if valid.
    """
    cleaned = type_str.strip()
    norm = cleaned if cleaned.startswith(".") else f".{cleaned}"
    if norm not in ALL_PTX_TYPES:
        return [
            f"Invalid PTX data type '{type_str}': must be one of {sorted(ALL_PTX_TYPES)}."
        ]
    return []


def validate_ptx_state_space(space_str: str) -> List[str]:
    """Validate whether a given memory space qualifier is a legal PTX state space.

    Args:
        space_str: State space identifier (e.g. '.global', '.shared', 'global').

    Returns:
        List of validation error messages, or empty list if valid.
    """
    cleaned = space_str.strip()
    norm = cleaned if cleaned.startswith(".") else f".{cleaned}"
    if norm not in ALL_PTX_STATE_SPACES:
        return [
            f"Invalid PTX state space '{space_str}': must be one of {sorted(ALL_PTX_STATE_SPACES)}."
        ]
    return []


def validate_ptx_scope(scope_str: str) -> List[str]:
    """Validate whether a given memory scope qualifier is a legal PTX scope.

    Args:
        scope_str: Scope identifier (e.g. '.gpu', '.cta', '.sys', '.cluster').

    Returns:
        List of validation error messages, or empty list if valid.
    """
    cleaned = scope_str.strip()
    norm = cleaned if cleaned.startswith(".") else f".{cleaned}"
    if norm not in ALL_PTX_SCOPES:
        return [
            f"Invalid PTX memory scope '{scope_str}': must be one of {sorted(ALL_PTX_SCOPES)}."
        ]
    return []


def validate_ptx_vector_width(width_str: str) -> List[str]:
    """Validate whether a given vector width qualifier is a legal PTX vector width.

    Args:
        width_str: Vector width identifier (e.g. '.v2', '.v4').

    Returns:
        List of validation error messages, or empty list if valid.
    """
    cleaned = width_str.strip()
    norm = cleaned if cleaned.startswith(".") else f".{cleaned}"
    if norm not in ALL_PTX_VECTOR_WIDTHS:
        return [
            f"Invalid PTX vector width '{width_str}': must be one of {sorted(ALL_PTX_VECTOR_WIDTHS)}."
        ]
    return []


def validate_ptx_register(operand: str) -> List[str]:
    """Validate a PTX register operand specification.

    Checks:
        - Predicate registers (%p0-%p15, %p, %pt)
        - Virtual registers (%r0-%r255, %rd0-%rd255, %f0-%f255, %fd0-%fd255)
        - Special registers (%tid, %ctaid, %nctaid, %laneid)
        - Memory dereferences [%r0], [%rd0 + 4]

    Args:
        operand: Raw operand identifier string.

    Returns:
        List of register validation errors.
    """
    errors: List[str] = []
    cleaned = operand.strip()

    # Memory dereference: [%reg + offset]
    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        reg_match = re.match(r"^%[A-Za-z0-9_]+", inner)
        if reg_match:
            errors.extend(validate_ptx_register(reg_match.group(0)))
            return errors
        return [f"Malformed PTX memory dereference operand '{operand}'."]

    # Special registers
    if cleaned in (
        "%tid",
        "%tid.x",
        "%tid.y",
        "%tid.z",
        "%ctaid",
        "%ctaid.x",
        "%ctaid.y",
        "%ctaid.z",
        "%nctaid",
        "%nctaid.x",
        "%nctaid.y",
        "%nctaid.z",
        "%laneid",
        "%warpid",
        "%smid",
        "%gridid",
        "%dynamic_smem_size",
        "%clock",
        "%clock64",
    ):
        return []

    # Predicate registers
    if cleaned in (
        "%pt",
        "!%pt",
        "%p0",
        "%p1",
        "%p2",
        "%p3",
        "%p4",
        "%p5",
        "%p6",
        "%p7",
    ):
        return []
    if re.match(r"^!?%p\d+$", cleaned):
        idx = int(re.sub(r"^!?%p", "", cleaned))
        if idx > 63:
            errors.append(f"PTX predicate register index %p{idx} out of bounds (0-63).")
        return errors

    # Virtual registers
    vreg_match = re.match(r"^%([A-Za-z_]+)(\d+)?$", cleaned)
    if not vreg_match and not cleaned.startswith("%"):
        # Immediate constant or label
        return []

    return errors


def validate_ptx_instruction(
    mnemonic: str,
    types: Optional[List[str]] = None,
    operands: Optional[List[str]] = None,
    state_space: Optional[str] = None,
    scope: Optional[str] = None,
    vector_width: Optional[str] = None,
    sm_arch: Optional[str] = None,
) -> List[str]:
    """Validate a complete PTX instruction invocation against ISA constraints.

    Args:
        mnemonic: Base instruction mnemonic (e.g. 'add', 'ld', 'st', 'wmma.mma').
        types: List of type qualifiers (e.g. ['.f32'], ['.u64']).
        operands: List of operand strings (e.g. ['%f0', '%f1', '%f2']).
        state_space: Optional memory state space qualifier (e.g. '.global').
        scope: Optional memory scope qualifier (e.g. '.gpu', '.cta', '.sys').
        vector_width: Optional vector width qualifier (e.g. '.v2', '.v4').
        sm_arch: Target SM architecture (e.g. 'sm_80', 'sm_90').

    Returns:
        List of instruction verification error messages.
    """
    errors: List[str] = []
    clean_mnem = mnemonic.strip().lower()

    ptx_db = {inst["mnemonic"]: inst for inst in _load_exhaustive_ptx()}
    if clean_mnem not in ptx_db:
        errors.append(f"Unknown PTX instruction mnemonic '{mnemonic}'.")
        return errors

    inst_meta = ptx_db[clean_mnem]

    # Validate state space
    if state_space:
        errors.extend(validate_ptx_state_space(state_space))
        allowed_spaces = inst_meta.get("state_spaces", [])
        norm_space = state_space if state_space.startswith(".") else f".{state_space}"
        if not allowed_spaces:
            errors.append(
                f"State space '{state_space}' is not valid for instruction '{mnemonic}': instruction does not accept memory state spaces."
            )
        elif norm_space not in allowed_spaces:
            errors.append(
                f"State space '{state_space}' is not valid for instruction '{mnemonic}' (supported: {allowed_spaces})."
            )

    # Validate memory scope
    if scope:
        errors.extend(validate_ptx_scope(scope))
        cat = inst_meta.get("category")
        if cat not in ("memory", "barrier"):
            errors.append(
                f"Memory scope '{scope}' is not valid for instruction '{mnemonic}': only memory and barrier instructions accept scopes."
            )

    # Validate vector width
    if vector_width:
        errors.extend(validate_ptx_vector_width(vector_width))
        allowed_widths = inst_meta.get("vector_widths", [])
        norm_vw = vector_width if vector_width.startswith(".") else f".{vector_width}"
        if not allowed_widths:
            errors.append(
                f"Vector width '{vector_width}' is not supported for PTX instruction '{mnemonic}'."
            )
        elif norm_vw not in allowed_widths:
            errors.append(
                f"Vector width '{vector_width}' is not supported for PTX instruction '{mnemonic}' (supported: {allowed_widths})."
            )

    # Validate types
    if types:
        for t in types:
            errors.extend(validate_ptx_type(t))
            norm_t = t if t.startswith(".") else f".{t}"
            supported = inst_meta.get("supported_types", [])
            if supported and norm_t not in supported:
                errors.append(
                    f"Type '{t}' is not supported for PTX instruction '{mnemonic}' (supported: {supported})."
                )

    # Validate operands count and registers
    if operands is not None:
        expected_ops = inst_meta.get("operands", [])
        if len(operands) != len(expected_ops):
            errors.append(
                f"PTX instruction '{mnemonic}' expects {len(expected_ops)} operands, got {len(operands)}."
            )
        for op in operands:
            errors.extend(validate_ptx_register(op))

    # Architecture requirement
    if sm_arch:
        min_sm = inst_meta.get("min_sm", "sm_50")
        min_num = int(re.sub(r"\D", "", min_sm) or "0")
        target_num = int(re.sub(r"\D", "", sm_arch) or "0")
        if target_num < min_num:
            errors.append(
                f"PTX instruction '{mnemonic}' requires at least {min_sm}, but target is '{sm_arch}'."
            )

    return errors


def _get_canonical_fallback_ptx() -> List[Dict[str, Any]]:
    """Return canonical baseline PTX instructions for offline operation when unbuilt.

    Returns:
        List of PTX instruction metadata dictionaries.
    """
    return [
        {
            "mnemonic": "add",
            "description": "Add two values.",
            "min_sm": "sm_50",
            "vector_widths": [".v2", ".v4"],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "First operand"},
                {"name": "b", "role": "src", "description": "Second operand"},
            ],
        },
        {
            "mnemonic": "sub",
            "description": "Subtract two values.",
            "min_sm": "sm_50",
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "First operand"},
                {"name": "b", "role": "src", "description": "Second operand"},
            ],
        },
        {
            "mnemonic": "mul",
            "description": "Multiply two values.",
            "min_sm": "sm_50",
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "First operand"},
                {"name": "b", "role": "src", "description": "Second operand"},
            ],
        },
        {
            "mnemonic": "div",
            "description": "Divide two values.",
            "min_sm": "sm_50",
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "First operand"},
                {"name": "b", "role": "src", "description": "Second operand"},
            ],
        },
        {
            "mnemonic": "fma",
            "description": "Fused multiply-add.",
            "min_sm": "sm_50",
            "supported_types": [".f16", ".f16x2", ".f32", ".f64"],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "First operand"},
                {"name": "b", "role": "src", "description": "Second operand"},
                {"name": "c", "role": "src", "description": "Third operand"},
            ],
        },
        {
            "mnemonic": "wgmma.mma_async",
            "description": "Asynchronous warp-group matrix multiply and accumulate.",
            "min_sm": "sm_90",
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "First operand"},
                {"name": "b", "role": "src", "description": "Second operand"},
            ],
        },
        {
            "mnemonic": "cp.async",
            "description": "Asynchronous copy from global memory to shared memory.",
            "min_sm": "sm_80",
            "operands": [
                {
                    "name": "dst",
                    "role": "dst",
                    "description": "Shared memory destination",
                },
                {
                    "name": "src",
                    "role": "src",
                    "description": "Global memory source",
                },
                {
                    "name": "size",
                    "role": "imm",
                    "description": "Copy size in bytes",
                },
            ],
        },
        {
            "mnemonic": "mbarrier.init",
            "description": "Initialize a memory barrier.",
            "category": "barrier",
            "min_sm": "sm_80",
            "state_spaces": [".shared"],
            "supported_types": [".b64"],
            "operands": [
                {
                    "name": "addr",
                    "role": "dst",
                    "description": "Barrier address",
                },
                {
                    "name": "count",
                    "role": "imm",
                    "description": "Thread count",
                },
            ],
        },
        {
            "mnemonic": "ldmatrix",
            "description": "Load matrix from shared memory into registers.",
            "category": "memory",
            "min_sm": "sm_75",
            "vector_widths": [".v1", ".v2", ".v4"],
            "supported_types": [".b16"],
            "operands": [
                {
                    "name": "r",
                    "role": "dst",
                    "description": "Destination register",
                },
                {
                    "name": "p",
                    "role": "src",
                    "description": "Source address",
                },
            ],
        },
        {
            "mnemonic": "tanh",
            "description": "Hyperbolic tangent.",
            "min_sm": "sm_75",
            "supported_types": [".f32"],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "Source operand"},
            ],
        },
        {
            "mnemonic": "ld",
            "description": "Load from memory.",
            "category": "memory",
            "min_sm": "sm_50",
            "state_spaces": [".global", ".shared", ".local", ".const", ".param"],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination register"},
                {"name": "a", "role": "src", "description": "Source address"},
            ],
        },
        {
            "mnemonic": "st",
            "description": "Store to memory.",
            "category": "memory",
            "min_sm": "sm_50",
            "state_spaces": [".global", ".shared", ".local", ".param"],
            "operands": [
                {"name": "a", "role": "dst", "description": "Destination address"},
                {"name": "b", "role": "src", "description": "Source value"},
            ],
        },
        {
            "mnemonic": "atom",
            "description": "Atomic reduction operation.",
            "category": "memory",
            "min_sm": "sm_50",
            "state_spaces": [".global", ".shared"],
            "scopes": [".cta", ".gpu", ".sys"],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination"},
                {"name": "a", "role": "src", "description": "Address"},
                {"name": "b", "role": "src", "description": "Value"},
            ],
        },
        {
            "mnemonic": "bar.sync",
            "description": "Barrier synchronization.",
            "min_sm": "sm_50",
            "operands": [
                {"name": "a", "role": "src", "description": "Barrier identifier"},
            ],
        },
    ]


def _load_exhaustive_ptx() -> List[Dict[str, Any]]:
    """Load the exhaustive NVIDIA PTX JSON database or fallback to canonical records.

    Returns:
        List of PTX instruction metadata dictionaries.
    """
    json_path = os.path.join(os.path.dirname(__file__), "nvidia_ptx_exhaustive.json")
    if not os.path.exists(json_path):
        if os.environ.get("PTX_DISABLE_FALLBACK") == "1":
            return []
        return _get_canonical_fallback_ptx()
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
    """Collect NVIDIA PTX ISA definitions into standard GhostRef representations.

    Args:
        category: Semantic category to collect (UTIL contains ISA instructions).
        include_nonpublic: Flag indicating whether non-public symbols are collected.

    Returns:
        List of GhostRef representations for PTX instructions.
    """
    if category != SemanticTier.UTIL:
        return []

    instructions = _load_exhaustive_ptx()
    refs: List[GhostRef] = []

    for inst in instructions:
        mnem = inst["mnemonic"]
        desc = inst.get("description", f"PTX {mnem} instruction.")
        params: List[GhostParam] = []

        for idx, op in enumerate(inst.get("operands", [])):
            role = (
                IRParameterRole.RESULT
                if op.get("role") == "dst"
                else IRParameterRole.OPERAND
            )
            params.append(
                ExtendedGhostParam(
                    name=op.get("name", f"op{idx}"),
                    annotation="PTXOperand",
                    default=None,
                    kind="POSITIONAL_ONLY",
                    standardized_name=op.get("role", "operand"),
                    description=op.get("description", "PTX instruction operand"),
                    role=role,
                )
            )

        refs.append(
            GhostRef(
                api_path=f"nvidia_ptx.inst.{mnem}",
                name=mnem,
                class_name=None,
                kind="function",
                is_public=True,
                has_varargs=False,
                environment_tags=["cuda", "ptx", inst.get("min_sm", "sm_50")],
                aliases=[],
                overloads=[],
                params=params,
                returns_type=None,
                returns_description=None,
                raises=[],
                docstring=desc,
            )
        )

    return refs
