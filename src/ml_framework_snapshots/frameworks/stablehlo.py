"""StableHLO Dialect API Snapshot Extractor.

Extracts StableHLO dialect operations by parsing an exhaustive JSON dump
representing the formal StableHLO TableGen ODS and spec definitions.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ..models import (
    ExtendedGhostParam,
    GhostMlirRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
)

STRUCTURED_STABLEHLO_SCHEMAS = {
    "DotDimensionNumbersAttr": {
        "type": "object",
        "properties": {
            "lhs_batch_dimensions": {"type": "array", "items": {"type": "integer"}},
            "rhs_batch_dimensions": {"type": "array", "items": {"type": "integer"}},
            "lhs_contracting_dimensions": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "rhs_contracting_dimensions": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": [
            "lhs_batch_dimensions",
            "rhs_batch_dimensions",
            "lhs_contracting_dimensions",
            "rhs_contracting_dimensions",
        ],
    },
    "ConvDimensionNumbersAttr": {
        "type": "object",
        "properties": {
            "input_batch_dimension": {"type": "integer"},
            "input_feature_dimension": {"type": "integer"},
            "input_spatial_dimensions": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "kernel_input_feature_dimension": {"type": "integer"},
            "kernel_output_feature_dimension": {"type": "integer"},
            "kernel_spatial_dimensions": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "output_batch_dimension": {"type": "integer"},
            "output_feature_dimension": {"type": "integer"},
            "output_spatial_dimensions": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": [
            "input_batch_dimension",
            "input_feature_dimension",
            "input_spatial_dimensions",
            "kernel_input_feature_dimension",
            "kernel_output_feature_dimension",
            "kernel_spatial_dimensions",
            "output_batch_dimension",
            "output_feature_dimension",
            "output_spatial_dimensions",
        ],
    },
    "ScatterDimensionNumbersAttr": {
        "type": "object",
        "properties": {
            "update_window_dims": {"type": "array", "items": {"type": "integer"}},
            "inserted_window_dims": {"type": "array", "items": {"type": "integer"}},
            "input_batching_dims": {"type": "array", "items": {"type": "integer"}},
            "scatter_indices_batching_dims": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "scatter_dims_to_operand_dims": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "index_vector_dim": {"type": "integer"},
        },
        "required": [
            "update_window_dims",
            "inserted_window_dims",
            "scatter_dims_to_operand_dims",
            "index_vector_dim",
        ],
    },
    "GatherDimensionNumbersAttr": {
        "type": "object",
        "properties": {
            "offset_dims": {"type": "array", "items": {"type": "integer"}},
            "collapsed_slice_dims": {"type": "array", "items": {"type": "integer"}},
            "operand_batching_dims": {"type": "array", "items": {"type": "integer"}},
            "start_indices_batching_dims": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "start_index_map": {"type": "array", "items": {"type": "integer"}},
            "index_vector_dim": {"type": "integer"},
        },
        "required": [
            "offset_dims",
            "collapsed_slice_dims",
            "start_index_map",
            "index_vector_dim",
        ],
    },
    "ComparisonDirectionAttr": {
        "type": "string",
        "enum": ["EQ", "NE", "GE", "GT", "LE", "LT"],
    },
    "PrecisionAttr": {
        "type": "string",
        "enum": ["DEFAULT", "HIGH", "HIGHEST"],
    },
}

STRUCTURED_REGION_SIGNATURES = {
    "stablehlo.reduce": {
        "body": {
            "description": "Reduction region",
            "block_arguments": ["tensor<T>", "tensor<T>"],
            "yield_types": ["tensor<T>"],
        }
    },
    "stablehlo.while": {
        "cond": {
            "description": "Condition evaluation region",
            "block_arguments": ["(args...)"],
            "yield_types": ["tensor<i1>"],
        },
        "body": {
            "description": "Loop body region",
            "block_arguments": ["(args...)"],
            "yield_types": ["(results...)"],
        },
    },
    "stablehlo.sort": {
        "comparator": {
            "description": "Comparator region for sorting",
            "block_arguments": ["tensor<T>", "tensor<T>"],
            "yield_types": ["tensor<i1>"],
        }
    },
}


def validate_dot_dimension_numbers(
    dim_numbers: Dict[str, Any],
    lhs_rank: Optional[int] = None,
    rhs_rank: Optional[int] = None,
) -> List[str]:
    """Validate DotDimensionNumbersAttr constraints against input operand ranks.

    Checks:
        - lhs_batch_dimensions count equals rhs_batch_dimensions count
        - lhs_contracting_dimensions count equals rhs_contracting_dimensions count
        - Dimension indices within rank bounds
        - Batch and contracting dimensions are disjoint within each operand

    Args:
        dim_numbers: Dictionary representing DotDimensionNumbersAttr.
        lhs_rank: Optional rank of LHS operand.
        rhs_rank: Optional rank of RHS operand.

    Returns:
        List of validation error strings.
    """
    errors: List[str] = []
    lhs_b = dim_numbers.get("lhs_batch_dimensions", [])
    rhs_b = dim_numbers.get("rhs_batch_dimensions", [])
    lhs_c = dim_numbers.get("lhs_contracting_dimensions", [])
    rhs_c = dim_numbers.get("rhs_contracting_dimensions", [])

    if len(lhs_b) != len(rhs_b):
        errors.append(
            f"Batch dimension count mismatch in DotDimensionNumbersAttr: lhs has {len(lhs_b)}, rhs has {len(rhs_b)}."
        )
    if len(lhs_c) != len(rhs_c):
        errors.append(
            f"Contracting dimension count mismatch in DotDimensionNumbersAttr: lhs has {len(lhs_c)}, rhs has {len(rhs_c)}."
        )

    # Disjointness checks
    if set(lhs_b) & set(lhs_c):
        errors.append(
            f"LHS batch and contracting dimensions must be disjoint (overlap: {set(lhs_b) & set(lhs_c)})."
        )
    if set(rhs_b) & set(rhs_c):
        errors.append(
            f"RHS batch and contracting dimensions must be disjoint (overlap: {set(rhs_b) & set(rhs_c)})."
        )

    # Rank bounds check
    if lhs_rank is not None:
        for d in lhs_b + lhs_c:
            if not isinstance(d, int) or d < 0 or d >= lhs_rank:
                errors.append(
                    f"LHS dimension {d} in DotDimensionNumbersAttr is out of bounds for rank {lhs_rank}."
                )
    if rhs_rank is not None:
        for d in rhs_b + rhs_c:
            if not isinstance(d, int) or d < 0 or d >= rhs_rank:
                errors.append(
                    f"RHS dimension {d} in DotDimensionNumbersAttr is out of bounds for rank {rhs_rank}."
                )

    return errors


def validate_conv_dimension_numbers(
    dim_numbers: Dict[str, Any],
    input_rank: Optional[int] = None,
    kernel_rank: Optional[int] = None,
    output_rank: Optional[int] = None,
) -> List[str]:
    """Validate ConvDimensionNumbersAttr spatial counts, completeness, and disjointness.

    Args:
        dim_numbers: Dictionary representing ConvDimensionNumbersAttr.
        input_rank: Optional rank of input tensor.
        kernel_rank: Optional rank of kernel tensor.
        output_rank: Optional rank of output tensor.

    Returns:
        List of validation error strings.
    """
    errors: List[str] = []

    in_b = dim_numbers.get("input_batch_dimension")
    in_f = dim_numbers.get("input_feature_dimension")
    in_spatial = dim_numbers.get("input_spatial_dimensions", [])

    k_in_f = dim_numbers.get("kernel_input_feature_dimension")
    k_out_f = dim_numbers.get("kernel_output_feature_dimension")
    k_spatial = dim_numbers.get("kernel_spatial_dimensions", [])

    out_b = dim_numbers.get("output_batch_dimension")
    out_f = dim_numbers.get("output_feature_dimension")
    out_spatial = dim_numbers.get("output_spatial_dimensions", [])

    # Spatial counts must match
    if not (len(in_spatial) == len(k_spatial) == len(out_spatial)):
        errors.append(
            f"Spatial dimensions count mismatch in ConvDimensionNumbersAttr: input={len(in_spatial)}, kernel={len(k_spatial)}, output={len(out_spatial)}."
        )

    # Input disjointness
    if in_b is not None and in_f is not None:
        in_dims = [in_b, in_f] + in_spatial
        if len(set(in_dims)) != len(in_dims):
            errors.append(
                f"Input dimensions in ConvDimensionNumbersAttr must be distinct: {in_dims}."
            )
        if input_rank is not None and any(d >= input_rank or d < 0 for d in in_dims):
            errors.append(
                f"Input dimension in ConvDimensionNumbersAttr exceeds input rank {input_rank}."
            )

    # Kernel disjointness
    if k_in_f is not None and k_out_f is not None:
        k_dims = [k_in_f, k_out_f] + k_spatial
        if len(set(k_dims)) != len(k_dims):
            errors.append(
                f"Kernel dimensions in ConvDimensionNumbersAttr must be distinct: {k_dims}."
            )
        if kernel_rank is not None and any(d >= kernel_rank or d < 0 for d in k_dims):
            errors.append(
                f"Kernel dimension in ConvDimensionNumbersAttr exceeds kernel rank {kernel_rank}."
            )

    # Output disjointness
    if out_b is not None and out_f is not None:
        out_dims = [out_b, out_f] + out_spatial
        if len(set(out_dims)) != len(out_dims):
            errors.append(
                f"Output dimensions in ConvDimensionNumbersAttr must be distinct: {out_dims}."
            )
        if output_rank is not None and any(d >= output_rank or d < 0 for d in out_dims):
            errors.append(
                f"Output dimension in ConvDimensionNumbersAttr exceeds output rank {output_rank}."
            )

    return errors


def validate_scatter_dimension_numbers(dim_numbers: Dict[str, Any]) -> List[str]:
    """Validate ScatterDimensionNumbersAttr disjointness and mappings.

    Args:
        dim_numbers: Dictionary representing ScatterDimensionNumbersAttr.

    Returns:
        List of validation error strings.
    """
    errors: List[str] = []
    upd_w = dim_numbers.get("update_window_dims", [])
    ins_w = dim_numbers.get("inserted_window_dims", [])

    if set(upd_w) & set(ins_w):
        errors.append(
            f"update_window_dims and inserted_window_dims in ScatterDimensionNumbersAttr must be disjoint (overlap: {set(upd_w) & set(ins_w)})."
        )

    return errors


def validate_gather_dimension_numbers(dim_numbers: Dict[str, Any]) -> List[str]:
    """Validate GatherDimensionNumbersAttr disjointness and mappings.

    Args:
        dim_numbers: Dictionary representing GatherDimensionNumbersAttr.

    Returns:
        List of validation error strings.
    """
    errors: List[str] = []
    off_d = dim_numbers.get("offset_dims", [])
    col_d = dim_numbers.get("collapsed_slice_dims", [])

    if set(off_d) & set(col_d):
        errors.append(
            f"offset_dims and collapsed_slice_dims in GatherDimensionNumbersAttr must be disjoint (overlap: {set(off_d) & set(col_d)})."
        )

    return errors


def validate_stablehlo_region(
    op_name: str,
    region_name: str,
    block_args: List[str],
    yield_types: List[str],
) -> List[str]:
    """Validate StableHLO region body arguments and terminator return types.

    Args:
        op_name: Qualified operation name (e.g. 'stablehlo.reduce', 'stablehlo.while').
        region_name: Name of region ('body', 'cond', 'comparator').
        block_args: List of block argument type strings.
        yield_types: List of yield/return type strings.

    Returns:
        List of region verification error strings.
    """
    errors: List[str] = []
    clean_op = op_name.lower().strip()

    if clean_op == "stablehlo.reduce":
        if len(block_args) < 2 or len(block_args) % 2 != 0:
            errors.append(
                f"stablehlo.reduce region takes 2 scalar arguments per reduce operand (got {len(block_args)})."
            )
        elif len(yield_types) * 2 != len(block_args):
            errors.append(
                f"stablehlo.reduce terminator yield count ({len(yield_types)}) must match reduce operand count ({len(block_args) // 2})."
            )

    elif clean_op == "stablehlo.while":
        if region_name == "cond":
            if not yield_types or not any("i1" in yt for yt in yield_types):
                errors.append(
                    f"stablehlo.while condition region must terminate with tensor<i1> (got {yield_types})."
                )
        elif region_name == "body":
            if block_args and yield_types and block_args != yield_types:
                errors.append(
                    f"stablehlo.while body region yield types ({yield_types}) must match block arguments ({block_args})."
                )

    elif clean_op == "stablehlo.sort":
        if region_name == "comparator":
            if len(block_args) != 2:
                errors.append(
                    f"stablehlo.sort comparator region takes 2 scalar arguments (got {len(block_args)})."
                )
            if not yield_types or not any("i1" in yt for yt in yield_types):
                errors.append(
                    f"stablehlo.sort comparator region must terminate with tensor<i1> (got {yield_types})."
                )

    elif clean_op in ("scf.for", "for"):
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

    return errors


def validate_broadcast_in_dim(
    operand_shape: List[Any],
    result_shape: List[Any],
    broadcast_dimensions: List[int],
) -> List[str]:
    """Validate broadcast_in_dim mapping length and dimension expansion.

    Args:
        operand_shape: Shape list of operand tensor.
        result_shape: Shape list of result tensor.
        broadcast_dimensions: Array of target dimension indices.

    Returns:
        List of broadcast validation errors.
    """
    errors: List[str] = []

    if len(broadcast_dimensions) != len(operand_shape):
        errors.append(
            f"broadcast_dimensions length ({len(broadcast_dimensions)}) must match operand rank ({len(operand_shape)})."
        )
        return errors

    for i, b_dim in enumerate(broadcast_dimensions):
        if not isinstance(b_dim, int) or b_dim < 0 or b_dim >= len(result_shape):
            errors.append(
                f"broadcast_dimensions index {b_dim} out of bounds for result rank {len(result_shape)}."
            )
            continue

        op_d = operand_shape[i]
        res_d = result_shape[b_dim]
        # Ignore dynamic dims ('?' or None)
        if op_d not in (1, "?", None) and res_d not in ("?", None):
            if op_d != res_d:
                errors.append(
                    f"Broadcast dimension mismatch at operand dim {i} (size {op_d}) mapping to result dim {b_dim} (size {res_d})."
                )

    return errors


def validate_binary_broadcast(
    lhs_shape: List[Any], rhs_shape: List[Any]
) -> Tuple[bool, List[Any], List[str]]:
    """Validate standard elementwise binary broadcasting between two shapes.

    Args:
        lhs_shape: Shape list of LHS operand.
        rhs_shape: Shape list of RHS operand.

    Returns:
        Tuple of (is_valid, broadcasted_result_shape, errors).
    """
    errors: List[str] = []
    res_shape: List[Any] = []

    max_rank = max(len(lhs_shape), len(rhs_shape))
    pad_lhs = [1] * (max_rank - len(lhs_shape)) + list(lhs_shape)
    pad_rhs = [1] * (max_rank - len(rhs_shape)) + list(rhs_shape)

    for i in range(max_rank):
        lhs_dim = pad_lhs[i]
        rhs_dim = pad_rhs[i]
        if lhs_dim in ("?", None) or rhs_dim in ("?", None):
            res_shape.append("?")
        elif lhs_dim == rhs_dim:
            res_shape.append(lhs_dim)
        elif lhs_dim == 1:
            res_shape.append(rhs_dim)
        elif rhs_dim == 1:
            res_shape.append(lhs_dim)
        else:
            errors.append(
                f"Incompatible broadcast dimensions at axis {i - max_rank}: LHS={lhs_dim}, RHS={rhs_dim}."
            )
            res_shape.append(None)

    return len(errors) == 0, res_shape, errors


def validate_stablehlo_op(
    op_name: str,
    operands: Optional[List[Dict[str, Any]]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    regions: Optional[Dict[str, Any]] = None,
    operand_ranks: Optional[List[int]] = None,
) -> List[str]:
    """Validate a StableHLO operation against dimension numbers, regions, and operand ranks.

    Checks:
        - DotDimensionNumbersAttr validation on stablehlo.dot_general
        - ConvDimensionNumbersAttr validation on stablehlo.convolution
        - ScatterDimensionNumbersAttr validation on stablehlo.scatter
        - GatherDimensionNumbersAttr validation on stablehlo.gather
        - Region block argument arity and terminator yield types

    Args:
        op_name: Qualified operation name (e.g. 'stablehlo.dot_general', 'stablehlo.reduce').
        operands: Optional list of SSA operand descriptor dictionaries.
        attributes: Optional dictionary of attributes passed to the operation.
        regions: Optional dictionary of regions attached to the operation.
        operand_ranks: Optional list of integer ranks for operands.

    Returns:
        List of validation error messages.
    """
    errors: List[str] = []
    clean_op = op_name.strip().lower()

    # 1. Validate structured dimension number attributes
    if attributes:
        for attr_k, attr_v in attributes.items():
            k_low = attr_k.lower().replace("_", "")
            if "dotdimensionnumbers" in k_low or "dotdimension" in k_low:
                if isinstance(attr_v, dict):
                    lhs_rk = (
                        operand_ranks[0]
                        if operand_ranks and len(operand_ranks) > 0
                        else None
                    )
                    rhs_rk = (
                        operand_ranks[1]
                        if operand_ranks and len(operand_ranks) > 1
                        else None
                    )
                    errors.extend(
                        validate_dot_dimension_numbers(
                            attr_v, lhs_rank=lhs_rk, rhs_rank=rhs_rk
                        )
                    )
                else:
                    errors.append(
                        f"DotDimensionNumbersAttr for '{op_name}' must be a dictionary specification."
                    )
            elif "convdimensionnumbers" in k_low or k_low == "dimensionnumbers":
                if isinstance(attr_v, dict):
                    errors.extend(validate_conv_dimension_numbers(attr_v))
                else:
                    errors.append(
                        f"ConvDimensionNumbersAttr for '{op_name}' must be a dictionary specification."
                    )
            elif "scatterdimensionnumbers" in k_low:
                if isinstance(attr_v, dict):
                    errors.extend(validate_scatter_dimension_numbers(attr_v))
                else:
                    errors.append(
                        f"ScatterDimensionNumbersAttr for '{op_name}' must be a dictionary specification."
                    )
            elif "gatherdimensionnumbers" in k_low:
                if isinstance(attr_v, dict):
                    errors.extend(validate_gather_dimension_numbers(attr_v))
                else:
                    errors.append(
                        f"GatherDimensionNumbersAttr for '{op_name}' must be a dictionary specification."
                    )

    # 2. Validate regions if present
    if regions and isinstance(regions, dict):
        for reg_name, reg_val in regions.items():
            block_args: List[str] = []
            yield_types: List[str] = []
            if isinstance(reg_val, dict):
                block_args = reg_val.get("block_arguments", [])
                yield_types = reg_val.get("yield_types", [])
            errors.extend(
                validate_stablehlo_region(
                    op_name=clean_op,
                    region_name=reg_name,
                    block_args=block_args,
                    yield_types=yield_types,
                )
            )

    return errors


def _get_canonical_stablehlo_records() -> List[Dict[str, Any]]:
    """Return canonical baseline StableHLO operations for offline operation when unbuilt.

    Returns:
        List of StableHLO operation dictionaries.
    """
    return [
        {
            "class_name": "DotGeneralOp",
            "api_path": "stablehlo.dot_general",
            "operands": [
                {"name": "lhs", "type": "tensor"},
                {"name": "rhs", "type": "tensor"},
            ],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [
                {"name": "dot_dimension_numbers", "type": "DotDimensionNumbersAttr"},
                {"name": "precision_config", "type": "PrecisionConfigAttr"},
            ],
            "traits": [],
            "summary": "General dot product of two tensors.",
        },
        {
            "class_name": "ConvolutionOp",
            "api_path": "stablehlo.convolution",
            "operands": [
                {"name": "lhs", "type": "tensor"},
                {"name": "rhs", "type": "tensor"},
            ],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [
                {"name": "dimension_numbers", "type": "ConvDimensionNumbersAttr"},
                {"name": "window_strides", "type": "DenseIntElementsAttr"},
                {"name": "padding", "type": "DenseIntElementsAttr"},
            ],
            "traits": [],
            "summary": "Convolution operation.",
        },
        {
            "class_name": "AddOp",
            "api_path": "stablehlo.add",
            "operands": [
                {"name": "lhs", "type": "tensor"},
                {"name": "rhs", "type": "tensor"},
            ],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Elementwise addition.",
        },
        {
            "class_name": "SubtractOp",
            "api_path": "stablehlo.subtract",
            "operands": [
                {"name": "lhs", "type": "tensor"},
                {"name": "rhs", "type": "tensor"},
            ],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Elementwise subtraction.",
        },
        {
            "class_name": "MultiplyOp",
            "api_path": "stablehlo.multiply",
            "operands": [
                {"name": "lhs", "type": "tensor"},
                {"name": "rhs", "type": "tensor"},
            ],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Elementwise multiplication.",
        },
        {
            "class_name": "DivideOp",
            "api_path": "stablehlo.divide",
            "operands": [
                {"name": "lhs", "type": "tensor"},
                {"name": "rhs", "type": "tensor"},
            ],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Elementwise division.",
        },
        {
            "class_name": "BroadcastInDimOp",
            "api_path": "stablehlo.broadcast_in_dim",
            "operands": [{"name": "operand", "type": "tensor"}],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [
                {"name": "broadcast_dimensions", "type": "DenseIntElementsAttr"}
            ],
            "traits": [],
            "summary": "Broadcast an array into a higher-ranked shape.",
        },
        {
            "class_name": "ReduceOp",
            "api_path": "stablehlo.reduce",
            "operands": [
                {"name": "inputs", "type": "tensor"},
                {"name": "init_values", "type": "tensor"},
            ],
            "results": [{"name": "results", "type": "tensor"}],
            "attributes": [{"name": "dimensions", "type": "DenseIntElementsAttr"}],
            "traits": [],
            "summary": "Apply a reduction to multidimensional inputs.",
        },
        {
            "class_name": "AbsOp",
            "api_path": "stablehlo.abs",
            "operands": [{"name": "operand", "type": "tensor"}],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Elementwise absolute value operation.",
        },
        {
            "class_name": "WhileOp",
            "api_path": "stablehlo.while",
            "operands": [{"name": "operand", "type": "Variadic<tensor>"}],
            "results": [{"name": "results", "type": "Variadic<tensor>"}],
            "attributes": [],
            "regions": ["cond", "body"],
            "traits": [],
            "summary": "While loop operation with condition and body regions.",
        },
        {
            "class_name": "SortOp",
            "api_path": "stablehlo.sort",
            "operands": [{"name": "inputs", "type": "Variadic<tensor>"}],
            "results": [{"name": "results", "type": "Variadic<tensor>"}],
            "attributes": [{"name": "dimension", "type": "I64Attr"}],
            "regions": ["comparator"],
            "traits": [],
            "summary": "Sort operation with comparator region.",
        },
        {
            "class_name": "ScatterOp",
            "api_path": "stablehlo.scatter",
            "operands": [
                {"name": "inputs", "type": "Variadic<tensor>"},
                {"name": "scatter_indices", "type": "tensor"},
                {"name": "updates", "type": "Variadic<tensor>"},
            ],
            "results": [{"name": "results", "type": "Variadic<tensor>"}],
            "attributes": [
                {
                    "name": "scatter_dimension_numbers",
                    "type": "ScatterDimensionNumbersAttr",
                },
                {"name": "indices_are_sorted", "type": "BoolAttr"},
                {"name": "unique_indices", "type": "BoolAttr"},
            ],
            "regions": ["update_computation"],
            "traits": [],
            "summary": "Scatter operation.",
        },
        {
            "class_name": "GatherOp",
            "api_path": "stablehlo.gather",
            "operands": [
                {"name": "operand", "type": "tensor"},
                {"name": "start_indices", "type": "tensor"},
            ],
            "results": [{"name": "result", "type": "tensor"}],
            "attributes": [
                {"name": "dimension_numbers", "type": "GatherDimensionNumbersAttr"},
                {"name": "slice_sizes", "type": "DenseIntElementsAttr"},
                {"name": "indices_are_sorted", "type": "BoolAttr"},
            ],
            "traits": [],
            "summary": "Gather operation.",
        },
    ]


def _load_stablehlo_exhaustive() -> List[GhostRef]:
    """Load the StableHLO exhaustive JSON dump or canonical fallback and map it to GhostRefs.

    Returns:
        A list of GhostRef items representing StableHLO operations.
    """
    json_path = os.path.join(os.path.dirname(__file__), "stablehlo_exhaustive.json")
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
    elif os.environ.get("STABLEHLO_DISABLE_FALLBACK") == "1":
        return []
    else:
        ops_list = _get_canonical_stablehlo_records()

    refs: List[GhostRef] = []
    for op in ops_list:
        params: List[ExtendedGhostParam] = []

        # 1. SSA Operands
        for operand in op.get("operands", []):
            op_name = operand["name"] if isinstance(operand, dict) else str(operand)
            op_type = (
                operand.get("type", "tensor") if isinstance(operand, dict) else "tensor"
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
        op_attributes: Dict[str, Any] = {}
        for attribute in op.get("attributes", []):
            attr_name = (
                attribute["name"] if isinstance(attribute, dict) else str(attribute)
            )
            attr_type = (
                attribute.get("type", "attribute")
                if isinstance(attribute, dict)
                else "attribute"
            )
            params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind="KEYWORD_ONLY",
                    annotation=attr_type,
                    standardized_name="attribute",
                    description=f"Attribute {attr_name} of type {attr_type}",
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
            matched_schema = None
            for schema_key, schema_val in STRUCTURED_STABLEHLO_SCHEMAS.items():
                clean_key = schema_key.lower().replace("attr", "")
                if clean_key in attr_name.lower().replace(
                    "_", ""
                ) or clean_key in attr_type.lower().replace("_", ""):
                    matched_schema = schema_val
                    break
            if matched_schema is not None:
                op_attributes[attr_name] = matched_schema
            else:
                op_attributes[attr_name] = {"type": attr_type}

        # 3. Op Regions
        for region in op.get("regions", []):
            reg_name = region["name"] if isinstance(region, dict) else str(region)
            params.append(
                ExtendedGhostParam(
                    name=reg_name,
                    kind="KEYWORD_ONLY",
                    annotation="Region",
                    standardized_name="region",
                    description=f"Op region block {reg_name}",
                    role=IRParameterRole.REGION,
                )
            )

        traits = op.get("traits", [])
        base_desc = (
            op.get("description")
            or f"StableHLO {op.get('class_name', 'Op')} operation."
        )
        docstring_parts = [base_desc]
        if traits:
            docstring_parts.append(f"Traits: {', '.join(traits)}")
        regions = op.get("regions", [])
        if regions:
            docstring_parts.append(f"Regions: {', '.join(str(r) for r in regions)}")

        returns_type: Optional[str] = None
        ghost_results: List[GhostResult] = []
        results = op.get("results", [])
        if results:
            res_types: List[str] = []
            for r in results:
                r_name = r.get("name", "result") if isinstance(r, dict) else "result"
                r_type = r.get("type", "tensor") if isinstance(r, dict) else str(r)
                res_types.append(r_type)
                ghost_results.append(GhostResult(name=r_name, type=r_type))
            returns_type = (
                res_types[0]
                if len(res_types) == 1
                else f"tuple[{', '.join(res_types)}]"
            )

        api_path = op.get("api_path", "")
        domain_metadata = {
            "traits": traits,
            "regions": STRUCTURED_REGION_SIGNATURES.get(api_path, regions),
            "structured_attribute_schemas": STRUCTURED_STABLEHLO_SCHEMAS,
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
                attributes=op_attributes if op_attributes else None,
                regions=STRUCTURED_REGION_SIGNATURES.get(api_path)
                if isinstance(STRUCTURED_REGION_SIGNATURES.get(api_path), dict)
                else None,
            )
        )

    return refs


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the StableHLO API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    if category != SemanticTier.UTIL:
        return []

    return _load_stablehlo_exhaustive()
