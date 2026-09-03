"""MLIR Dialect API Snapshot Extractor.

Extracts MLIR dialects and operations (e.g., arith, math, scf, stablehlo) by parsing
an exhaustive JSON dump scraped directly from MLIR and StableHLO documentation.
"""

import json
import os
from typing import List

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostParam


def _load_mlir_exhaustive() -> List[GhostRef]:
    """Load the MLIR exhaustive JSON dump and map it to GhostRefs.

    Returns:
        A list of GhostRef items representing MLIR operations.
    """
    json_path = os.path.join(os.path.dirname(__file__), "mlir_exhaustive.json")
    if not os.path.exists(json_path):
        return []

    refs = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for op in data:
            params = []
            for operand in op.get("operands", []):
                params.append(GhostParam(name=operand, kind="POSITIONAL_OR_KEYWORD"))
            for attribute in op.get("attributes", []):
                params.append(GhostParam(name=attribute, kind="KEYWORD_ONLY"))

            refs.append(
                GhostRef(
                    name=op.get("class_name", "UnknownOp"),
                    api_path=op.get("api_path", ""),
                    kind="CLASS",
                    params=params,
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
