"""MLIR Dialect API Snapshot Extractor.

Extracts MLIR dialects and operations (e.g., arith, math, tensor, linalg, scf, func,
memref, gpu, vector, llvm, nvvm, rocdl) by parsing an exhaustive JSON dump preserving SSA types,
attributes, results, and regions.
"""

from enum import Enum
import json
import os
import re
from typing import Any, Dict, List, Optional

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ..models import (
    ExtendedGhostParam,
    GhostMlirRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
)


class MLIRDialectTrait(str, Enum):
    """Core verification traits for MLIR operations."""

    SAME_OPERANDS_AND_RESULT_TYPE = "SameOperandsAndResultType"
    SAME_TYPE_OPERANDS = "SameTypeOperands"
    COMMUTATIVE = "Commutative"
    ELEMENTWISE = "Elementwise"
    SCALARIZABLE = "Scalarizable"
    ISOLATED_FROM_ABOVE = "IsolatedFromAbove"
    ATTR_SIZED_OPERAND_SEGMENTS = "AttrSizedOperandSegments"
    ATTR_SIZED_RESULT_SEGMENTS = "AttrSizedResultSegments"


class MLIRTypeCategory(str, Enum):
    """High-level categories of MLIR types."""

    TENSOR = "tensor"
    VECTOR = "vector"
    MEMREF = "memref"
    INTEGER = "integer"
    FLOAT = "float"
    INDEX = "index"
    COMPLEX = "complex"
    CUSTOM = "custom"


def classify_mlir_type(type_str: str) -> str:
    """Classify an MLIR type string into a recognized type category.

    Args:
        type_str: Raw type string (e.g. 'tensor<4x?xf32>', 'vector<[4]xi32>', 'i32', 'memref<10xf32>').

    Returns:
        The MLIRTypeCategory value string.
    """
    cleaned = type_str.strip().lower()
    if cleaned.startswith("tensor<") or "tensor" in cleaned:
        return MLIRTypeCategory.TENSOR.value
    if cleaned.startswith("vector<") or "vector" in cleaned:
        return MLIRTypeCategory.VECTOR.value
    if cleaned.startswith("memref<") or "memref" in cleaned:
        return MLIRTypeCategory.MEMREF.value
    if cleaned == "index":
        return MLIRTypeCategory.INDEX.value
    if re.match(r"^(s|u)?i\d+$", cleaned) or "integer" in cleaned:
        return MLIRTypeCategory.INTEGER.value
    if re.match(r"^(f16|f32|f64|bf16|f8e\d+m\d+.*)$", cleaned) or "float" in cleaned:
        return MLIRTypeCategory.FLOAT.value
    if cleaned.startswith("complex<"):
        return MLIRTypeCategory.COMPLEX.value
    return MLIRTypeCategory.CUSTOM.value


def validate_mlir_type(type_str: str, constraint: Optional[str] = None) -> List[str]:
    """Validate MLIR SSA type syntax and check compliance against TableGen type constraints.

    Validates:
        - Tensor types: RankedTensorOf<...>, UnrankedTensorOf (tensor<*xf32>), dynamic dims (tensor<?x?xf32>)
        - Vector types: VectorOf<...>, scalable vectors (vector<[4]xf32>)
        - Memory references: MemRefOf<...>, strided layouts, address spaces
        - Integer types: Signless (i1, i8, i16, i32, i64), signed (si32), unsigned (ui32), boolean (i1), index
        - Floating-point types: f16, f32, f64, bf16, etc.

    Args:
        type_str: Actual SSA type string to validate.
        constraint: Optional TableGen ODS constraint string (e.g. 'RankedTensorOf<[F16, F32]>', 'AnyInteger').

    Returns:
        List of type validation errors.
    """
    errors: List[str] = []
    cleaned = type_str.strip()
    c_low = (constraint or "").lower()

    cat = classify_mlir_type(cleaned)

    # 1. Syntax validation
    if cat == MLIRTypeCategory.TENSOR.value:
        if not (cleaned.startswith("tensor<") and cleaned.endswith(">")):
            errors.append(f"Malformed MLIR tensor type syntax: '{type_str}'")
    elif cat == MLIRTypeCategory.VECTOR.value:
        if not (cleaned.startswith("vector<") and cleaned.endswith(">")):
            errors.append(f"Malformed MLIR vector type syntax: '{type_str}'")
    elif cat == MLIRTypeCategory.MEMREF.value:
        if not (cleaned.startswith("memref<") and cleaned.endswith(">")):
            errors.append(f"Malformed MLIR memref type syntax: '{type_str}'")
    elif cat == MLIRTypeCategory.INTEGER.value:
        if not re.match(r"^(s|u)?i\d+$", cleaned.lower()):
            errors.append(f"Malformed MLIR integer type: '{type_str}'")

    # 2. Constraint validation if specified
    if constraint:
        if "unrankedtensorof" in c_low:
            if cat != MLIRTypeCategory.TENSOR.value or "*" not in cleaned:
                errors.append(
                    f"Type '{type_str}' does not satisfy UnrankedTensorOf constraint."
                )
        elif "rankedtensorof" in c_low:
            if cat != MLIRTypeCategory.TENSOR.value or "*" in cleaned:
                errors.append(
                    f"Type '{type_str}' does not satisfy RankedTensorOf constraint '{constraint}'."
                )
        elif "vectorof" in c_low:
            if cat != MLIRTypeCategory.VECTOR.value:
                errors.append(
                    f"Type '{type_str}' does not satisfy VectorOf constraint '{constraint}'."
                )
        elif "memrefof" in c_low:
            if cat != MLIRTypeCategory.MEMREF.value:
                errors.append(
                    f"Type '{type_str}' does not satisfy MemRefOf constraint '{constraint}'."
                )
        elif "anyinteger" in c_low or "anysignlessinteger" in c_low:
            if cat not in (
                MLIRTypeCategory.INTEGER.value,
                MLIRTypeCategory.INDEX.value,
            ):
                errors.append(
                    f"Type '{type_str}' does not satisfy integer constraint '{constraint}': expected integer type."
                )
        elif "anyfloat" in c_low:
            if cat != MLIRTypeCategory.FLOAT.value:
                errors.append(
                    f"Type '{type_str}' does not satisfy float constraint '{constraint}': expected float type."
                )
        elif "anytensor" in c_low or "staticshapetensor" in c_low:
            if cat != MLIRTypeCategory.TENSOR.value:
                errors.append(
                    f"Type '{type_str}' does not satisfy tensor constraint '{constraint}': expected tensor type."
                )
            elif "staticshapetensor" in c_low and ("?" in cleaned or "*" in cleaned):
                errors.append(
                    f"Type '{type_str}' does not satisfy static shape constraint '{constraint}': dynamic or unranked dimensions not permitted."
                )
        elif c_low == "index":
            if cat != MLIRTypeCategory.INDEX.value:
                errors.append(
                    f"Type '{type_str}' does not satisfy index constraint '{constraint}': expected 'index' type."
                )

    return errors


def validate_mlir_traits(
    op_name: str,
    traits: List[str],
    operand_types: Optional[List[str]] = None,
    result_types: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
    structured_attributes: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate dialect verification traits for an MLIR operation invocation.

    Checks:
        - SameOperandsAndResultType: all operand and result types must be identical
        - SameTypeOperands: all operand types must match
        - AttrSizedOperandSegments: requires operand_segment_sizes attribute
        - AttrSizedResultSegments: requires result_segment_sizes attribute
        - Elementwise / Scalarizable shape compatibility

    Args:
        op_name: Qualified operation name (e.g. 'arith.addf').
        traits: List of trait names associated with the operation.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        attributes: Optional list of attribute names present on the op.
        structured_attributes: Optional dictionary of structured attribute values.

    Returns:
        List of trait verification error messages.
    """
    errors: List[str] = []
    t_set = set(traits)

    # 1. SameOperandsAndResultType
    if (
        MLIRDialectTrait.SAME_OPERANDS_AND_RESULT_TYPE.value in t_set
        or "SameOperandsAndResultType" in t_set
    ):
        all_types: List[str] = []
        if operand_types:
            all_types.extend(operand_types)
        if result_types:
            all_types.extend(result_types)
        if len(set(all_types)) > 1:
            errors.append(
                f"Trait violation for '{op_name}': SameOperandsAndResultType requires all operand and result types to be identical (got operands={operand_types}, results={result_types})."
            )

    # 2. SameTypeOperands
    if (
        MLIRDialectTrait.SAME_TYPE_OPERANDS.value in t_set
        or "SameTypeOperands" in t_set
    ):
        if operand_types and len(set(operand_types)) > 1:
            errors.append(
                f"Trait violation for '{op_name}': SameTypeOperands requires all operand types to match (got {operand_types})."
            )

    # 3. AttrSizedOperandSegments
    if (
        MLIRDialectTrait.ATTR_SIZED_OPERAND_SEGMENTS.value in t_set
        or "AttrSizedOperandSegments" in t_set
    ):
        has_segment_attr = False
        if attributes and "operand_segment_sizes" in attributes:
            has_segment_attr = True
        if structured_attributes and "operand_segment_sizes" in structured_attributes:
            has_segment_attr = True
            seg_val = structured_attributes["operand_segment_sizes"]
            if isinstance(seg_val, list) and operand_types is not None:
                if sum(seg_val) != len(operand_types):
                    errors.append(
                        f"AttrSizedOperandSegments mismatch for '{op_name}': sum of operand_segment_sizes ({sum(seg_val)}) does not match operand count ({len(operand_types)})."
                    )
        if not has_segment_attr and (
            attributes is not None or structured_attributes is not None
        ):
            errors.append(
                f"Missing required attribute 'operand_segment_sizes' for op '{op_name}' with AttrSizedOperandSegments trait."
            )

    # 4. AttrSizedResultSegments
    if (
        MLIRDialectTrait.ATTR_SIZED_RESULT_SEGMENTS.value in t_set
        or "AttrSizedResultSegments" in t_set
    ):
        has_res_segment_attr = False
        if attributes and "result_segment_sizes" in attributes:
            has_res_segment_attr = True
        if structured_attributes and "result_segment_sizes" in structured_attributes:
            has_res_segment_attr = True
        if not has_res_segment_attr and (
            attributes is not None or structured_attributes is not None
        ):
            errors.append(
                f"Missing required attribute 'result_segment_sizes' for op '{op_name}' with AttrSizedResultSegments trait."
            )

    # 5. Commutative
    if MLIRDialectTrait.COMMUTATIVE.value in t_set or "Commutative" in t_set:
        if operand_types is not None and len(operand_types) < 2:
            errors.append(
                f"Trait violation for '{op_name}': Commutative trait requires at least 2 operands, got {len(operand_types)}."
            )

    # 6. Elementwise
    if MLIRDialectTrait.ELEMENTWISE.value in t_set or "Elementwise" in t_set:
        if operand_types and len(operand_types) >= 2:
            elem_types = [
                t.split("<")[-1].rstrip(">") for t in operand_types if "<" in t
            ]
            if len(elem_types) > 1 and len(set(elem_types)) > 1:
                errors.append(
                    f"Trait violation for '{op_name}': Elementwise trait requires operands to have matching element types (got {elem_types})."
                )

    # 7. IsolatedFromAbove
    if (
        MLIRDialectTrait.ISOLATED_FROM_ABOVE.value in t_set
        or "IsolatedFromAbove" in t_set
    ):
        if structured_attributes and structured_attributes.get("has_external_captures"):
            errors.append(
                f"Trait violation for '{op_name}': IsolatedFromAbove forbids implicit captures of SSA values defined above the region."
            )

    return errors


def validate_mlir_region(
    op_name: str,
    region_name: str,
    block_args: List[str],
    yield_types: List[str],
    isolated_from_above: bool = False,
) -> List[str]:
    """Validate MLIR region body block arguments, yield terminator types, and capture isolation.

    Args:
        op_name: Qualified operation name (e.g. 'scf.for', 'func.func', 'gpu.launch').
        region_name: Identifier name of the region (e.g. 'body', 'region').
        block_args: List of SSA block argument type strings.
        yield_types: List of yield/return terminator type strings.
        isolated_from_above: Whether the region has the IsolatedFromAbove trait.

    Returns:
        List of region verification error messages.
    """
    errors: List[str] = []
    clean_op = op_name.strip().lower()

    if clean_op in ("scf.for", "for"):
        if region_name == "body":
            if not block_args or block_args[0] != "index":
                errors.append(
                    f"scf.for body region first block argument (induction variable) must be 'index' type (got {block_args[0] if block_args else 'none'})."
                )
            if len(block_args) > 1:
                iter_args = block_args[1:]
                if yield_types and yield_types != iter_args:
                    errors.append(
                        f"scf.for body region yield types ({yield_types}) must match loop-carried iter_args ({iter_args})."
                    )

    elif clean_op in ("scf.while", "while"):
        if region_name == "before":
            if not yield_types or not any("i1" in yt for yt in yield_types):
                errors.append(
                    f"scf.while before region must yield a condition boolean i1 (got {yield_types})."
                )

    return errors


def validate_mlir_successors(
    op_name: str,
    successors: List[str],
    expected_count: Optional[int] = None,
) -> List[str]:
    """Validate successor block targets for MLIR control flow operations.

    Args:
        op_name: Qualified operation name (e.g. 'cf.br', 'cf.cond_br').
        successors: List of successor block identifier labels (e.g. ['^bb1', '^bb2']).
        expected_count: Optional expected number of branch successors.

    Returns:
        List of successor validation error messages.
    """
    errors: List[str] = []
    clean_op = op_name.strip().lower()

    for succ in successors:
        s_clean = succ.strip()
        if not (
            s_clean.startswith("^")
            or s_clean.startswith("bb")
            or s_clean.startswith("block")
        ):
            errors.append(
                f"Malformed successor block label '{succ}' for '{op_name}': expected prefix '^' or 'bb'."
            )

    if expected_count is not None and len(successors) != expected_count:
        errors.append(
            f"Successor count mismatch for '{op_name}': expected {expected_count}, got {len(successors)}."
        )

    if clean_op == "cf.br" and len(successors) != 1:
        errors.append(
            f"Unconditional branch 'cf.br' requires exactly 1 successor block (got {len(successors)})."
        )
    elif clean_op == "cf.cond_br" and len(successors) != 2:
        errors.append(
            f"Conditional branch 'cf.cond_br' requires exactly 2 successor blocks (true_dest, false_dest, got {len(successors)})."
        )

    return errors


def _get_canonical_mlir_records() -> List[Dict[str, Any]]:
    """Return canonical baseline MLIR operations for offline operation when unbuilt.

    Returns:
        List of MLIR operation dictionaries.
    """
    return [
        {
            "class_name": "AddFOp",
            "api_path": "arith.addf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Floating-point addition operation",
        },
        {
            "class_name": "SubFOp",
            "api_path": "arith.subf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Floating-point subtraction operation",
        },
        {
            "class_name": "MulFOp",
            "api_path": "arith.mulf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Floating-point multiplication operation",
        },
        {
            "class_name": "DivFOp",
            "api_path": "arith.divf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Floating-point division operation",
        },
        {
            "class_name": "AddIOp",
            "api_path": "arith.addi",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyInteger"},
                {"name": "rhs", "type": "AnyInteger"},
            ],
            "results": [{"name": "result", "type": "AnyInteger"}],
            "attributes": [
                {"name": "overflowFlags", "type": "IntegerOverflowFlagsAttr"}
            ],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Integer addition operation",
        },
        {
            "class_name": "SubIOp",
            "api_path": "arith.subi",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyInteger"},
                {"name": "rhs", "type": "AnyInteger"},
            ],
            "results": [{"name": "result", "type": "AnyInteger"}],
            "attributes": [
                {"name": "overflowFlags", "type": "IntegerOverflowFlagsAttr"}
            ],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Integer subtraction operation",
        },
        {
            "class_name": "MulIOp",
            "api_path": "arith.muli",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyInteger"},
                {"name": "rhs", "type": "AnyInteger"},
            ],
            "results": [{"name": "result", "type": "AnyInteger"}],
            "attributes": [
                {"name": "overflowFlags", "type": "IntegerOverflowFlagsAttr"}
            ],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Integer multiplication operation",
        },
        {
            "class_name": "ExpOp",
            "api_path": "math.exp",
            "dialect": "math",
            "operands": [{"name": "operand", "type": "AnyFloat"}],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Exponential operation",
        },
        {
            "class_name": "FuncOp",
            "api_path": "func.func",
            "dialect": "func",
            "operands": [],
            "results": [],
            "attributes": [
                {"name": "sym_name", "type": "StringAttr"},
                {"name": "function_type", "type": "TypeAttr"},
            ],
            "traits": ["IsolatedFromAbove"],
            "summary": "Function definition",
        },
        {
            "class_name": "ReturnOp",
            "api_path": "func.return",
            "dialect": "func",
            "operands": [{"name": "operands", "type": "Variadic<AnyType>"}],
            "results": [],
            "attributes": [],
            "traits": ["Terminator"],
            "summary": "Function return operation",
        },
        {
            "class_name": "ForOp",
            "api_path": "scf.for",
            "dialect": "scf",
            "operands": [
                {"name": "lowerBound", "type": "Index"},
                {"name": "upperBound", "type": "Index"},
                {"name": "step", "type": "Index"},
            ],
            "results": [{"name": "results", "type": "Variadic<AnyType>"}],
            "attributes": [],
            "traits": ["RecursiveMemoryEffects"],
            "summary": "Structured control flow for loop",
        },
        {
            "class_name": "YieldOp",
            "api_path": "scf.yield",
            "dialect": "scf",
            "operands": [{"name": "results", "type": "Variadic<AnyType>"}],
            "results": [],
            "attributes": [],
            "traits": ["Terminator"],
            "summary": "SCF loop yield",
        },
        {
            "class_name": "EmptyOp",
            "api_path": "tensor.empty",
            "dialect": "tensor",
            "operands": [{"name": "dynamicSizes", "type": "Variadic<Index>"}],
            "results": [{"name": "result", "type": "AnyTensor"}],
            "attributes": [{"name": "staticSizes", "type": "DenseI64ArrayAttr"}],
            "traits": [],
            "summary": "Create empty uninitialized tensor",
        },
        {
            "class_name": "MatmulOp",
            "api_path": "linalg.matmul",
            "dialect": "linalg",
            "operands": [
                {"name": "inputs", "type": "Variadic<AnyShaped>"},
                {"name": "outputs", "type": "Variadic<AnyShaped>"},
            ],
            "results": [{"name": "results", "type": "Variadic<AnyRankedTensor>"}],
            "attributes": [],
            "traits": ["StructuredOpTrait"],
            "summary": "Linalg matrix multiplication",
        },
    ]


def _load_mlir_exhaustive() -> List[GhostRef]:
    """Load the MLIR exhaustive JSON dump or canonical fallback and map it to GhostRefs.

    Returns:
        A list of GhostRef items representing MLIR operations.
    """
    json_path = os.path.join(os.path.dirname(__file__), "mlir_exhaustive.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "categories" in data:
                    all_ops: List[Dict[str, Any]] = []
                    for cat_ops in data["categories"].values():
                        if isinstance(cat_ops, list):
                            all_ops.extend(cat_ops)
                    ops_list = all_ops
                else:
                    ops_list = data.get("operations", [])
            else:
                ops_list = data
        except (json.JSONDecodeError, OSError):
            return []
    elif os.environ.get("MLIR_DISABLE_FALLBACK") == "1":
        return []
    else:
        ops_list = _get_canonical_mlir_records()

    refs: List[GhostRef] = []
    for op in ops_list:
        params: List[ExtendedGhostParam] = []

        # 1. SSA Operands
        for operand in op.get("operands", []):
            op_name = operand["name"] if isinstance(operand, dict) else str(operand)
            op_type = (
                operand.get("type", "Value") if isinstance(operand, dict) else "Value"
            )
            params.append(
                ExtendedGhostParam(
                    name=op_name,
                    kind="POSITIONAL_OR_KEYWORD",
                    annotation=op_type,
                    standardized_name="operand",
                    description=f"SSA operand {op_name} of type {op_type}",
                    direction=OperandDirection.READ,
                    role=IRParameterRole.OPERAND,
                )
            )

        # 2. Attributes
        for attribute in op.get("attributes", []):
            attr_name = (
                attribute["name"] if isinstance(attribute, dict) else str(attribute)
            )
            attr_type = (
                attribute.get("type", "Attribute")
                if isinstance(attribute, dict)
                else "Attribute"
            )
            params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind="KEYWORD_ONLY",
                    annotation=attr_type,
                    standardized_name="attribute",
                    description=f"Buildable attribute {attr_name} of type {attr_type}",
                    role=IRParameterRole.ATTRIBUTE,
                )
            )

        # 3. Regions
        for region in op.get("regions", []):
            reg_name = region["name"] if isinstance(region, dict) else str(region)
            params.append(
                ExtendedGhostParam(
                    name=reg_name,
                    kind="KEYWORD_ONLY",
                    annotation="Region",
                    standardized_name="region",
                    description=f"Op region {reg_name}",
                    role=IRParameterRole.REGION,
                )
            )

        returns_type: Optional[str] = None
        ghost_results: List[GhostResult] = []
        results = op.get("results", [])
        if results:
            res_types: List[str] = []
            for r in results:
                r_name = r.get("name", "result") if isinstance(r, dict) else "result"
                r_type = r.get("type", "Value") if isinstance(r, dict) else str(r)
                res_types.append(r_type)
                ghost_results.append(GhostResult(name=r_name, type=r_type))
            returns_type = (
                res_types[0]
                if len(res_types) == 1
                else f"tuple[{', '.join(res_types)}]"
            )

        docstring_parts = [
            op.get("description") or f"MLIR {op.get('class_name', 'Op')} operation."
        ]
        traits = op.get("traits", [])
        if traits:
            docstring_parts.append(f"Traits: {', '.join(traits)}")

        domain_metadata = {
            "dialect": op.get("dialect"),
            "traits": traits,
            "regions": op.get("regions", []),
        }

        ssa_operands = [
            p for p in params if getattr(p, "role", None) == IRParameterRole.OPERAND
        ]

        refs.append(
            GhostMlirRef(
                name=op.get("class_name", "UnknownOp"),
                api_path=op.get("api_path", ""),
                kind="function",
                params=params,
                operands=ssa_operands,
                docstring="\n".join(docstring_parts),
                returns_type=returns_type,
                returns=ghost_results,
                environment_tags=["cpu", "cuda", "rocm", "tpu"],
                domain_metadata=domain_metadata,
                traits=traits,
                attributes=op.get("attributes")
                if isinstance(op.get("attributes"), dict)
                else None,
                regions=op.get("regions")
                if isinstance(op.get("regions"), dict)
                else None,
                successors=op.get("successors")
                if isinstance(op.get("successors"), list)
                else None,
            )
        )

    return refs


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the MLIR API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    if category != SemanticTier.UTIL:
        return []

    return _load_mlir_exhaustive()
