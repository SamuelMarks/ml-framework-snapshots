"""NVIDIA SASS (Assembly) Extractor.

Provides a static snapshot of NVIDIA SASS instructions.
"""

import json
import os
from typing import List, Dict, Any

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostParam


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
        docstring = str(
            inst_data.get("description", f"NVIDIA SASS {mnemonic} instruction.")
        )
        modifiers = inst_data.get("modifiers", [])
        if modifiers:
            docstring += f"\nModifiers: {', '.join(modifiers)}"

        operands_list: List[List[str]] = inst_data.get("operands", [])

        # To map operands structurally, we find the maximum length signature
        max_sig: List[str] = []
        for sig in operands_list:
            if len(sig) > len(max_sig):
                max_sig = sig

        params = []
        for i, op_type in enumerate(max_sig):
            params.append(
                GhostParam(
                    name=f"op{i}",
                    kind="POSITIONAL_ONLY",
                    annotation=op_type,
                    description=f"Operand {i} of type {op_type}",
                )
            )

        refs.append(
            GhostRef(
                name=mnemonic,
                api_path=f"nvidia_sass.inst.{mnemonic}",
                kind="function",
                params=params,
                docstring=docstring,
            )
        )

    return refs
