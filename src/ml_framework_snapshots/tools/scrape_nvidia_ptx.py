"""Tool for scraping and compiling exhaustive NVIDIA PTX ISA definitions.

Parses LLVM NVPTX TableGen files (NVPTXInstrInfo.td, NVPTXIntrinsics.td) or
builds structured PTX ISA definitions covering arithmetic, conversions,
asynchronous data movement, tensor core matrix operations, barriers,
and memory qualifiers across PTX ISA 7.x and 8.x.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

LLVM_NVPTX_BASE_URL = (
    "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/NVPTX"
)
NVPTX_TD_FILES = [
    "NVPTXInstrInfo.td",
    "NVPTXIntrinsics.td",
]


def fetch_nvptx_td_file(filename: str, local_dir: Optional[str] = None) -> str:
    """Fetch a TableGen file from local checkout or LLVM repository.

    Args:
        filename: Name of the TableGen file.
        local_dir: Optional local directory containing LLVM NVPTX TableGen files.

    Returns:
        Content of the file as string, or empty string on error.
    """
    resolved_dir = local_dir or os.environ.get("LLVM_NVPTX_TD_DIR")
    if resolved_dir:
        candidate = os.path.join(resolved_dir, filename)
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()

    url = f"{LLVM_NVPTX_BASE_URL}/{filename}"
    try:
        with urllib.request.urlopen(url) as response:
            return str(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def parse_nvptx_td_content(content: str) -> List[Dict[str, Any]]:
    """Parse NVPTX TableGen text content to extract instruction mnemonics.

    Args:
        content: Raw TableGen file content.

    Returns:
        List of parsed instruction dictionaries.
    """
    instructions: List[Dict[str, Any]] = []
    pattern = re.compile(
        r'def\s+([A-Za-z0-9_]+)\s*:\s*[A-Za-z0-9_]*\s*<.*?"([a-z0-9_\.]+)"'
    )
    for match in pattern.finditer(content):
        _def_name = match.group(1)
        mnemonic = match.group(2)
        instructions.append(
            {
                "mnemonic": mnemonic,
                "category": "instruction",
                "description": f"PTX {mnemonic} instruction parsed from TableGen.",
                "supported_types": [".b32", ".b64"],
                "vector_widths": [],
                "state_spaces": [],
                "operands": [],
                "min_sm": "sm_50",
            }
        )
    return instructions


def build_exhaustive_ptx_catalog(
    base_instructions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the comprehensive NVIDIA PTX ISA 7.x / 8.x instruction catalog.

    Includes:
        - Arithmetic, transcendental, and bitwise ops (add, sub, mul, fma, mad, div, min, max, tanh, ex2)
        - Comparisons & selection (setp, selp)
        - Movement, addressing, conversions (mov, cvt, cvta, cvt.rna, cvt.rni)
        - Memory and atomics (ld, st, ld.global.nc, st.global, atom, red)
        - Asynchronous data movement (cp.async, cp.async.bulk, cp.async.commit_group, cp.async.wait_group, cp.reduce.async)
        - Barrier and warp synchronization (bar.sync, bar.arrive, bar.red, mbarrier.init, mbarrier.arrive, mbarrier.test_wait, barrier.cluster)
        - Tensor Core MMA (wmma.mma, mma.sync, wgmma.mma_async, wgmma.fence, wgmma.commit_group, wgmma.wait_group, ldmatrix, stmatrix)

    Args:
        base_instructions: Optional base list of instructions from TableGen scraping.

    Returns:
        Exhaustive list of PTX instruction metadata dictionaries.
    """
    catalog: Dict[str, Dict[str, Any]] = {}

    # Seed with base parsed instructions if provided
    if base_instructions:
        for inst in base_instructions:
            m = inst.get("mnemonic")
            if m and m not in catalog:
                catalog[m] = inst

    # Comprehensive PTX 7.x / 8.x instruction definitions
    standard_ops: List[Dict[str, Any]] = [
        # Arithmetic & Transcendental
        {
            "mnemonic": "add",
            "description": "Integer or floating-point addition.",
            "category": "arithmetic",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "sub",
            "description": "Integer or floating-point subtraction.",
            "category": "arithmetic",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "mul",
            "description": "Integer or floating-point multiplication.",
            "category": "arithmetic",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "fma",
            "description": "Fused multiply-add: d = (a * b) + c with single rounding step.",
            "category": "arithmetic",
            "supported_types": [".f16", ".f16x2", ".bf16", ".bf16x2", ".f32", ".f64"],
            "vector_widths": [".v2"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First multiplicand"},
                {"name": "b", "role": "src1", "description": "Second multiplicand"},
                {"name": "c", "role": "src2", "description": "Addend operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "mad",
            "description": "Multiply and add: d = (a * b) + c.",
            "category": "arithmetic",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First multiplicand"},
                {"name": "b", "role": "src1", "description": "Second multiplicand"},
                {"name": "c", "role": "src2", "description": "Addend operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "div",
            "description": "Divide: d = a / b.",
            "category": "arithmetic",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Numerator"},
                {"name": "b", "role": "src1", "description": "Denominator"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "min",
            "description": "Minimum of two values.",
            "category": "arithmetic",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "max",
            "description": "Maximum of two values.",
            "category": "arithmetic",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "tanh",
            "description": "Hyperbolic tangent: d = tanh(a).",
            "category": "arithmetic",
            "supported_types": [".f16", ".f16x2", ".f32"],
            "vector_widths": [".v2"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Source operand"},
            ],
            "min_sm": "sm_75",
        },
        {
            "mnemonic": "ex2",
            "description": "Base-2 exponential approximation: d = 2^a.",
            "category": "arithmetic",
            "supported_types": [".f16", ".f16x2", ".f32"],
            "vector_widths": [".v2"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Source exponent"},
            ],
            "min_sm": "sm_50",
        },
        # Logical & Bitwise
        {
            "mnemonic": "and",
            "description": "Bitwise AND operation.",
            "category": "logical",
            "supported_types": [".b16", ".b32", ".b64", ".pred"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "or",
            "description": "Bitwise OR operation.",
            "category": "logical",
            "supported_types": [".b16", ".b32", ".b64", ".pred"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "xor",
            "description": "Bitwise XOR operation.",
            "category": "logical",
            "supported_types": [".b16", ".b32", ".b64", ".pred"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "not",
            "description": "Bitwise NOT operation.",
            "category": "logical",
            "supported_types": [".b16", ".b32", ".b64", ".pred"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Source operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "shl",
            "description": "Shift bits left.",
            "category": "logical",
            "supported_types": [".b16", ".b32", ".b64"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Value to shift"},
                {"name": "b", "role": "src1", "description": "Shift amount"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "shr",
            "description": "Shift bits right (logical or arithmetic).",
            "category": "logical",
            "supported_types": [
                ".b16",
                ".b32",
                ".b64",
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Value to shift"},
                {"name": "b", "role": "src1", "description": "Shift amount"},
            ],
            "min_sm": "sm_50",
        },
        # Comparison & Selection
        {
            "mnemonic": "setp",
            "description": "Set predicate based on operand comparison.",
            "category": "comparison",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "p", "role": "dst", "description": "Destination predicate"},
                {"name": "a", "role": "src0", "description": "First operand"},
                {"name": "b", "role": "src1", "description": "Second operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "selp",
            "description": "Select between two values based on a predicate condition.",
            "category": "comparison",
            "supported_types": [
                ".b16",
                ".b32",
                ".b64",
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Selected if true"},
                {"name": "b", "role": "src1", "description": "Selected if false"},
                {
                    "name": "p",
                    "role": "src2",
                    "description": "Condition predicate operand",
                },
            ],
            "min_sm": "sm_50",
        },
        # Movement & Conversion
        {
            "mnemonic": "mov",
            "description": "Move data between registers or load immediate constant.",
            "category": "movement",
            "supported_types": [
                ".b16",
                ".b32",
                ".b64",
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f32",
                ".f64",
                ".pred",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Source operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "cvt",
            "description": "Convert data from one type to another with optional rounding mode.",
            "category": "movement",
            "supported_types": [
                ".u16",
                ".u32",
                ".u64",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".bf16",
                ".bf16x2",
                ".tf32",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2"],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Source operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "cvt.rna",
            "description": "Convert floating-point value to integer rounding to nearest away from zero.",
            "category": "movement",
            "supported_types": [
                ".s16",
                ".s32",
                ".s64",
                ".u16",
                ".u32",
                ".u64",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Source operand"},
            ],
            "min_sm": "sm_70",
        },
        {
            "mnemonic": "cvt.rni",
            "description": "Convert floating-point value to integer rounding to nearest even integer.",
            "category": "movement",
            "supported_types": [
                ".s16",
                ".s32",
                ".s64",
                ".u16",
                ".u32",
                ".u64",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination operand"},
                {"name": "a", "role": "src0", "description": "Source operand"},
            ],
            "min_sm": "sm_70",
        },
        {
            "mnemonic": "cvta",
            "description": "Convert generic address to specific state-space address or vice-versa.",
            "category": "movement",
            "supported_types": [".u32", ".u64"],
            "vector_widths": [],
            "state_spaces": [".global", ".local", ".shared", ".const", ".param"],
            "operands": [
                {
                    "name": "d",
                    "role": "dst",
                    "description": "Converted address destination",
                },
                {"name": "a", "role": "src0", "description": "Source address"},
            ],
            "min_sm": "sm_50",
        },
        # Memory & Atomic Operations
        {
            "mnemonic": "ld",
            "description": "Load data from generic or state space memory into registers.",
            "category": "memory",
            "supported_types": [
                ".b8",
                ".b16",
                ".b32",
                ".b64",
                ".b128",
                ".u8",
                ".u16",
                ".u32",
                ".u64",
                ".s8",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".bf16",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [".global", ".local", ".shared", ".const", ".param"],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination register"},
                {
                    "name": "a",
                    "role": "src0",
                    "description": "Memory address operand",
                },
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "ld.global.nc",
            "description": "Non-coherent load from global memory caching in L2/Texture cache.",
            "category": "memory",
            "supported_types": [
                ".b8",
                ".b16",
                ".b32",
                ".b64",
                ".u8",
                ".u16",
                ".u32",
                ".u64",
                ".s8",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [".global"],
            "operands": [
                {"name": "d", "role": "dst", "description": "Destination register"},
                {"name": "a", "role": "src0", "description": "Global memory address"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "st",
            "description": "Store data from registers into generic or state space memory.",
            "category": "memory",
            "supported_types": [
                ".b8",
                ".b16",
                ".b32",
                ".b64",
                ".b128",
                ".u8",
                ".u16",
                ".u32",
                ".u64",
                ".s8",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".bf16",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [".global", ".local", ".shared", ".param"],
            "operands": [
                {"name": "a", "role": "dst", "description": "Memory address operand"},
                {"name": "b", "role": "src0", "description": "Source register value"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "st.global",
            "description": "Store data directly into global memory with memory scope qualifiers.",
            "category": "memory",
            "supported_types": [
                ".b8",
                ".b16",
                ".b32",
                ".b64",
                ".b128",
                ".u8",
                ".u16",
                ".u32",
                ".u64",
                ".s8",
                ".s16",
                ".s32",
                ".s64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [".global"],
            "operands": [
                {"name": "a", "role": "dst", "description": "Global memory address"},
                {"name": "b", "role": "src0", "description": "Source register value"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "atom",
            "description": "Atomic read-modify-write operation with memory scope qualifiers.",
            "category": "memory",
            "supported_types": [
                ".u32",
                ".u64",
                ".s32",
                ".s64",
                ".b32",
                ".b64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [".global", ".shared"],
            "operands": [
                {
                    "name": "d",
                    "role": "dst",
                    "description": "Destination receiving old value",
                },
                {"name": "a", "role": "src0", "description": "Memory address operand"},
                {"name": "b", "role": "src1", "description": "Source operand"},
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "red",
            "description": "Atomic reduction in memory without returning previous value.",
            "category": "memory",
            "supported_types": [
                ".u32",
                ".u64",
                ".s32",
                ".s64",
                ".b32",
                ".b64",
                ".f16",
                ".f16x2",
                ".f32",
                ".f64",
            ],
            "vector_widths": [],
            "state_spaces": [".global", ".shared"],
            "operands": [
                {"name": "a", "role": "dst", "description": "Memory address operand"},
                {"name": "b", "role": "src0", "description": "Source operand"},
            ],
            "min_sm": "sm_50",
        },
        # Asynchronous Data Movement
        {
            "mnemonic": "cp.async",
            "description": "Asynchronous copy from global memory to shared memory bypassing register file.",
            "category": "memory",
            "supported_types": [".cg", ".ca"],
            "vector_widths": [],
            "state_spaces": [".shared", ".global"],
            "operands": [
                {
                    "name": "dst",
                    "role": "dst",
                    "description": "Shared memory destination address",
                },
                {
                    "name": "src",
                    "role": "src0",
                    "description": "Global memory source address",
                },
                {
                    "name": "cp_size",
                    "role": "src1",
                    "description": "Byte count (4, 8, 16)",
                },
            ],
            "min_sm": "sm_80",
        },
        {
            "mnemonic": "cp.async.bulk",
            "description": "TMA / Bulk asynchronous data copy between global and shared memory.",
            "category": "memory",
            "supported_types": [".b8", ".b32"],
            "vector_widths": [],
            "state_spaces": [".shared", ".global"],
            "operands": [
                {
                    "name": "dst",
                    "role": "dst",
                    "description": "Shared memory destination address",
                },
                {
                    "name": "src",
                    "role": "src0",
                    "description": "Global memory source address",
                },
                {
                    "name": "size",
                    "role": "src1",
                    "description": "Bulk transfer byte size",
                },
                {
                    "name": "mbar",
                    "role": "src2",
                    "description": "Mbarrier object address for synchronization",
                },
            ],
            "min_sm": "sm_90",
        },
        {
            "mnemonic": "cp.async.commit_group",
            "description": "Commits all prior cp.async instructions into a pending asynchronous group.",
            "category": "memory",
            "supported_types": [],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [],
            "min_sm": "sm_80",
        },
        {
            "mnemonic": "cp.async.wait_group",
            "description": "Waits until at most N asynchronous copy groups remain pending.",
            "category": "memory",
            "supported_types": [],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {
                    "name": "n",
                    "role": "src0",
                    "description": "Number of remaining pending groups",
                }
            ],
            "min_sm": "sm_80",
        },
        {
            "mnemonic": "cp.reduce.async",
            "description": "Asynchronous reduction from shared memory to global memory.",
            "category": "memory",
            "supported_types": [".u32", ".u64", ".f16", ".f32"],
            "vector_widths": [],
            "state_spaces": [".global", ".shared"],
            "operands": [
                {
                    "name": "dst",
                    "role": "dst",
                    "description": "Global memory destination address",
                },
                {
                    "name": "src",
                    "role": "src0",
                    "description": "Shared memory source address",
                },
            ],
            "min_sm": "sm_90",
        },
        # Barrier & Warp Synchronization
        {
            "mnemonic": "bar.sync",
            "description": "Synchronize threads in a CTA at a named barrier.",
            "category": "barrier",
            "supported_types": [".u32"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {
                    "name": "name",
                    "role": "src0",
                    "description": "Barrier index or count",
                }
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "bar.arrive",
            "description": "Signal arrival at a barrier without blocking thread execution.",
            "category": "barrier",
            "supported_types": [".u32"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "name", "role": "src0", "description": "Barrier index"}
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "bar.red",
            "description": "Perform reduction across threads arriving at a barrier.",
            "category": "barrier",
            "supported_types": [".pred"],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "p", "role": "dst", "description": "Result predicate"},
                {"name": "name", "role": "src0", "description": "Barrier index"},
                {
                    "name": "pred_val",
                    "role": "src1",
                    "description": "Input predicate value",
                },
            ],
            "min_sm": "sm_50",
        },
        {
            "mnemonic": "barrier.cluster",
            "description": "Synchronize all CTAs within a Thread Block Cluster.",
            "category": "barrier",
            "supported_types": [],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [],
            "min_sm": "sm_90",
        },
        {
            "mnemonic": "mbarrier.init",
            "description": "Initialize an asynchronous memory barrier object in shared memory.",
            "category": "barrier",
            "supported_types": [".b64"],
            "vector_widths": [],
            "state_spaces": [".shared"],
            "operands": [
                {
                    "name": "addr",
                    "role": "dst",
                    "description": "Shared memory address of barrier",
                },
                {
                    "name": "count",
                    "role": "src0",
                    "description": "Expected arrival count",
                },
            ],
            "min_sm": "sm_80",
        },
        {
            "mnemonic": "mbarrier.arrive",
            "description": "Signal arrival at an asynchronous memory barrier object.",
            "category": "barrier",
            "supported_types": [".b64"],
            "vector_widths": [],
            "state_spaces": [".shared"],
            "operands": [
                {
                    "name": "state",
                    "role": "dst",
                    "description": "Output arrival token/state",
                },
                {
                    "name": "addr",
                    "role": "src0",
                    "description": "Shared memory address of barrier",
                },
            ],
            "min_sm": "sm_80",
        },
        {
            "mnemonic": "mbarrier.test_wait",
            "description": "Test whether all expected threads have arrived at an mbarrier object.",
            "category": "barrier",
            "supported_types": [".b64"],
            "vector_widths": [],
            "state_spaces": [".shared"],
            "operands": [
                {
                    "name": "p",
                    "role": "dst",
                    "description": "Result boolean predicate",
                },
                {
                    "name": "addr",
                    "role": "src0",
                    "description": "Shared memory address of barrier",
                },
                {"name": "state", "role": "src1", "description": "Arrival token/state"},
            ],
            "min_sm": "sm_80",
        },
        # Tensor Core Matrix Operations
        {
            "mnemonic": "wmma.mma",
            "description": "Warp-level matrix multiply and accumulate on Tensor Cores (Volta / Turing / Ampere).",
            "category": "tensor",
            "supported_types": [
                ".f16",
                ".f32",
                ".s8",
                ".u8",
                ".s32",
                ".bf16",
                ".tf32",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Accumulator destination"},
                {"name": "a", "role": "src0", "description": "Matrix A fragment"},
                {"name": "b", "role": "src1", "description": "Matrix B fragment"},
                {"name": "c", "role": "src2", "description": "Matrix C accumulator"},
            ],
            "min_sm": "sm_70",
        },
        {
            "mnemonic": "mma.sync",
            "description": "Synchronous warp-level matrix multiply-accumulate across threads.",
            "category": "tensor",
            "supported_types": [
                ".f16",
                ".f32",
                ".s8",
                ".u8",
                ".s32",
                ".bf16",
                ".tf32",
                ".e4m3",
                ".e5m2",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Accumulator destination"},
                {"name": "a", "role": "src0", "description": "Matrix A fragment"},
                {"name": "b", "role": "src1", "description": "Matrix B fragment"},
                {"name": "c", "role": "src2", "description": "Matrix C accumulator"},
            ],
            "min_sm": "sm_80",
        },
        {
            "mnemonic": "wgmma.mma_async",
            "description": "Warpgroup-level asynchronous matrix multiply and accumulate (Hopper).",
            "category": "tensor",
            "supported_types": [
                ".f16",
                ".f32",
                ".bf16",
                ".tf32",
                ".e4m3",
                ".e5m2",
                ".s32",
            ],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {"name": "d", "role": "dst", "description": "Accumulator destination"},
                {
                    "name": "a",
                    "role": "src0",
                    "description": "Matrix A shared memory descriptor or register fragment",
                },
                {
                    "name": "b",
                    "role": "src1",
                    "description": "Matrix B shared memory descriptor",
                },
            ],
            "min_sm": "sm_90",
        },
        {
            "mnemonic": "wgmma.fence",
            "description": "Enforces ordering between register accesses and wgmma.mma_async instructions.",
            "category": "tensor",
            "supported_types": [],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [],
            "min_sm": "sm_90",
        },
        {
            "mnemonic": "wgmma.commit_group",
            "description": "Commits all prior wgmma.mma_async instructions into a warpgroup batch.",
            "category": "tensor",
            "supported_types": [],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [],
            "min_sm": "sm_90",
        },
        {
            "mnemonic": "wgmma.wait_group",
            "description": "Waits until at most N warpgroup batches remain outstanding.",
            "category": "tensor",
            "supported_types": [],
            "vector_widths": [],
            "state_spaces": [],
            "operands": [
                {
                    "name": "n",
                    "role": "src0",
                    "description": "Number of remaining pending batches",
                }
            ],
            "min_sm": "sm_90",
        },
        {
            "mnemonic": "ldmatrix",
            "description": "Load matrix fragment collectively across warp from shared memory into registers.",
            "category": "tensor",
            "supported_types": [".b16"],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [".shared"],
            "operands": [
                {
                    "name": "d",
                    "role": "dst",
                    "description": "Destination register tuple",
                },
                {
                    "name": "a",
                    "role": "src0",
                    "description": "Shared memory base address",
                },
            ],
            "min_sm": "sm_75",
        },
        {
            "mnemonic": "stmatrix",
            "description": "Store matrix fragment collectively across warp from registers into shared memory.",
            "category": "tensor",
            "supported_types": [".b16"],
            "vector_widths": [".v2", ".v4"],
            "state_spaces": [".shared"],
            "operands": [
                {
                    "name": "a",
                    "role": "dst",
                    "description": "Shared memory base address",
                },
                {"name": "b", "role": "src0", "description": "Source register tuple"},
            ],
            "min_sm": "sm_90",
        },
    ]

    for op in standard_ops:
        catalog[op["mnemonic"]] = op

    return list(catalog.values())


def scrape_ptx(
    td_dir: Optional[str] = None, output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Scrape NVPTX TableGen definitions and write the exhaustive PTX ISA dump.

    Args:
        td_dir: Optional directory containing local NVPTX TableGen files.
        output_path: Target path to write exhaustive JSON.

    Returns:
        List of generated instruction metadata records.
    """
    base_instructions: List[Dict[str, Any]] = []
    for td_file in NVPTX_TD_FILES:
        content = fetch_nvptx_td_file(td_file, local_dir=td_dir)
        if content:
            base_instructions.extend(parse_nvptx_td_content(content))

    catalog = build_exhaustive_ptx_catalog(base_instructions)

    resolved_output = output_path or os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "nvidia_ptx_exhaustive.json",
    )

    with open(resolved_output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, sort_keys=True)
        f.write("\n")

    return catalog


def main() -> None:
    """Entrypoint script for running the PTX ISA scraper."""
    scrape_ptx()


if __name__ == "__main__":  # pragma: no cover
    main()
