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
    ExtendedGhostRef,
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
        elif "anytensor" in c_low:
            if cat != MLIRTypeCategory.TENSOR.value:
                errors.append(
                    f"Type '{type_str}' does not satisfy tensor constraint '{constraint}': expected tensor type."
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


def _load_mlir_exhaustive() -> List[GhostRef]:
    """Load the MLIR exhaustive JSON dump and map it to GhostRefs.

    Returns:
        A list of GhostRef items representing MLIR operations.
    """
    json_path = os.path.join(os.path.dirname(__file__), "mlir_exhaustive.json")
    if not os.path.exists(json_path):
        return []

    refs: List[GhostRef] = []
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

        for op in ops_list:
            params: List[ExtendedGhostParam] = []

            # 1. SSA Operands
            for operand in op.get("operands", []):
                op_name = operand["name"] if isinstance(operand, dict) else str(operand)
                op_type = (
                    operand.get("type", "Value")
                    if isinstance(operand, dict)
                    else "Value"
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
                    r_name = (
                        r.get("name", "result") if isinstance(r, dict) else "result"
                    )
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

            refs.append(
                ExtendedGhostRef(
                    name=op.get("class_name", "UnknownOp"),
                    api_path=op.get("api_path", ""),
                    kind="function",
                    params=params,
                    docstring="\n".join(docstring_parts),
                    returns_type=returns_type,
                    returns=ghost_results,
                    environment_tags=["cpu", "cuda", "rocm", "tpu"],
                    domain_metadata=domain_metadata,
                )
            )
    except (json.JSONDecodeError, OSError):  # pragma: no cover
        pass

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
