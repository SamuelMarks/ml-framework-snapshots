"""MLIR Dialect API Snapshot Extractor.

Extracts MLIR dialects and operations (e.g., arith, math, tensor, linalg, scf, func,
memref, gpu, vector) by parsing an exhaustive JSON dump preserving SSA types,
attributes, results, and regions.
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

        for op in data:
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
