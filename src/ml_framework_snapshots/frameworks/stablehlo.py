"""StableHLO Dialect API Snapshot Extractor.

Extracts StableHLO dialect operations by parsing an exhaustive JSON dump
representing the formal StableHLO TableGen ODS and spec definitions.
"""

import json
import os
from typing import List, Optional

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ..models import (
    ExtendedGhostParam,
    ExtendedGhostRef,
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


def _load_stablehlo_exhaustive() -> List[GhostRef]:
    """Load the StableHLO exhaustive JSON dump and map it to GhostRefs.

    Returns:
        A list of GhostRef items representing StableHLO operations.
    """
    json_path = os.path.join(os.path.dirname(__file__), "stablehlo_exhaustive.json")
    if not os.path.exists(json_path):
        return []

    refs: List[GhostRef] = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for op in data:
            params: List[ExtendedGhostParam] = []

            # 1. SSA Operands
            for operand in op.get("operands", []):
                op_name = operand["name"] if isinstance(operand, dict) else str(operand)
                op_type = (
                    operand.get("type", "tensor")
                    if isinstance(operand, dict)
                    else "tensor"
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
                    r_name = (
                        r.get("name", "result") if isinstance(r, dict) else "result"
                    )
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
    except (json.JSONDecodeError, OSError):
        pass

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
