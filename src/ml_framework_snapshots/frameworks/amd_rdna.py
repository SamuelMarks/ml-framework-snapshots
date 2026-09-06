"""AMD RDNA (Assembly) Extractor.

Provides a static snapshot of AMD RDNA instructions with TableGen-derived encodings,
register classes, operand directionality, and GFX architecture targeting.
"""

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

ALL_GFX_ARCHITECTURES: List[str] = [
    "GFX9/CDNA",
    "GFX10/RDNA1",
    "GFX10.3/RDNA2",
    "GFX11/RDNA3",
    "GFX11.5",
    "GFX12/RDNA4",
]


def resolve_gfx_architectures(arch_spec: Optional[str]) -> List[str]:
    """Resolve an architecture specification into discrete targeted GFX microarchitectures.

    Args:
        arch_spec: Architecture specification string (e.g. 'GFX10+', 'GFX11/RDNA3').

    Returns:
        List of targeted GFX architectures.
    """
    if not arch_spec:
        return [
            "GFX10/RDNA1",
            "GFX10.3/RDNA2",
            "GFX11/RDNA3",
            "GFX11.5",
            "GFX12/RDNA4",
        ]

    spec = str(arch_spec).strip()
    if spec == "GFX9/CDNA":
        return ["GFX9/CDNA"]
    if spec == "GFX10/RDNA1":
        return ["GFX10/RDNA1"]
    if spec == "GFX10.3/RDNA2":
        return ["GFX10.3/RDNA2"]
    if spec == "GFX11/RDNA3":
        return ["GFX11/RDNA3"]
    if spec == "GFX11.5":
        return ["GFX11.5"]
    if spec == "GFX12/RDNA4":
        return ["GFX12/RDNA4"]
    if "GFX11" in spec:
        return ["GFX11/RDNA3", "GFX11.5", "GFX12/RDNA4"]
    if "GFX12" in spec:
        return ["GFX12/RDNA4"]

    # GFX10+ default
    return [
        "GFX10/RDNA1",
        "GFX10.3/RDNA2",
        "GFX11/RDNA3",
        "GFX11.5",
        "GFX12/RDNA4",
    ]


def _load_exhaustive_rdna() -> List[Dict[str, Any]]:
    """Loads the exhaustive AMD RDNA json dump."""
    json_path = os.path.join(os.path.dirname(__file__), "amd_rdna_exhaustive.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
        return data


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the AMD RDNA API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs: List[GhostRef] = []

    rdna_data = _load_exhaustive_rdna()

    for inst_data in rdna_data:
        mnemonic = str(inst_data["mnemonic"])
        encoding = str(inst_data.get("encoding", "Unknown"))
        base_desc = str(
            inst_data.get("description", f"AMD RDNA {mnemonic} instruction.")
        )
        modifiers = inst_data.get("modifiers", [])
        arch = inst_data.get("architecture")
        valid_archs = resolve_gfx_architectures(arch)
        env_tags = ["rocm"] + valid_archs

        docstring = (
            f"AMD RDNA {mnemonic} instruction.\n"
            f"Encoding: {encoding}\n"
            f"Description: {base_desc}\n"
            f"Modifiers: {', '.join(modifiers) if modifiers else 'None'}\n"
            f"Target Architectures: {', '.join(valid_archs)}"
        )

        operands_list: List[List[str]] = inst_data.get("operands", [])

        # Find the maximum length signature
        max_sig: List[str] = []
        for sig in operands_list:
            if len(sig) > len(max_sig):
                max_sig = sig

        params: List[ExtendedGhostParam] = []
        for i, op_type in enumerate(max_sig):
            if op_type in ("VCC", "SCC"):
                role = "condition"
                direction = OperandDirection.PREDICATE
            elif op_type == "EXEC":
                role = "exec_mask"
                direction = OperandDirection.READ
            else:
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
            "encoding": encoding,
            "architecture": arch,
            "valid_architectures": valid_archs,
            "modifiers": modifiers,
            "operand_signatures": operands_list,
        }

        # Build overloads for all operand variations
        overloads: List[ExtendedGhostRef] = []
        for sig_idx, sig in enumerate(operands_list):
            overload_params: List[ExtendedGhostParam] = []
            for i, op_type in enumerate(sig):
                if op_type in ("VCC", "SCC"):
                    role = "condition"
                    direction = OperandDirection.PREDICATE
                elif op_type == "EXEC":
                    role = "exec_mask"
                    direction = OperandDirection.READ
                else:
                    role = "dst" if i == 0 else f"src{i - 1}"
                    direction = (
                        OperandDirection.WRITE if i == 0 else OperandDirection.READ
                    )

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
                    api_path=f"amd_rdna.inst.{mnemonic}",
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
                api_path=f"amd_rdna.inst.{mnemonic}",
                kind="function",
                params=params,
                docstring=docstring,
                environment_tags=env_tags,
                overloads=overloads,
                domain_metadata=domain_metadata,
            )
        )

    return refs
