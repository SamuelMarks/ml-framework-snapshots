"""Anti-regression test suite for ML framework snapshots.

Enforces:
1. Zero unparsed regex tokens (e.g. '???0', 'SNOWFLAKE') in bundled snapshots.
2. No decay to (*args, **kwargs) for top PyTorch operators.
3. Zero duplicate placeholder operand signatures in AMD RDNA and NVIDIA SASS.
4. Complete separation of SSA operands from buildable attributes in MLIR/StableHLO.
"""

import json
import os
from typing import Any, List, Set


def get_bundled_json(filename: str) -> Any:
    """Load bundled JSON file from frameworks directory.

    Args:
        filename: Name of the JSON file in frameworks directory.

    Returns:
        Parsed JSON content list.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fpath = os.path.join(
        pkg_dir, "src", "ml_framework_snapshots", "frameworks", filename
    )
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "categories" in data:
            items: List[Any] = []
            for cat_items in data["categories"].values():
                if isinstance(cat_items, list):
                    items.extend(cat_items)
            return items
        return data


def test_zero_unparsed_regex_tokens_in_bundled_snapshots() -> None:
    """Verify zero unparsed regex or snowflake placeholder tokens in any bundled snapshot."""
    forbidden_tokens = ["???0", "???1", "SNOWFLAKE", "<placeholder>", "UNKNOWN_OPERAND"]
    snapshots = [
        "nvidia_sass_exhaustive.json",
        "amd_rdna_exhaustive.json",
        "mlir_exhaustive.json",
        "stablehlo_exhaustive.json",
    ]

    for sname in snapshots:
        data = get_bundled_json(sname)
        raw_text = json.dumps(data)
        for tok in forbidden_tokens:
            assert (
                tok not in raw_text
            ), f"Forbidden unparsed token '{tok}' found in {sname}"


def test_no_decay_to_varargs_for_top_pytorch_ops() -> None:
    """Verify top PyTorch functions do not decay to (*args, **kwargs)."""
    import torch
    from ml_framework_snapshots.models import GhostInspector

    top_ops = [
        (torch.relu, "torch.relu"),
        (torch.matmul, "torch.matmul"),
        (torch.add, "torch.add"),
        (torch.sum, "torch.sum"),
        (torch.cat, "torch.cat"),
        (torch.mm, "torch.mm"),
        (torch.bmm, "torch.bmm"),
    ]

    for op, api_path in top_ops:
        ref = GhostInspector.inspect(op, api_path)
        # Check that it did not decay to inexact (*args, **kwargs)
        if len(ref.params) == 2:
            names = [p.name for p in ref.params]
            assert not (
                names == ["args", "kwargs"]
            ), f"Operator {api_path} decayed to (*args, **kwargs)!"
        assert "inexact_signature" not in (
            ref.environment_tags or []
        ), f"Operator {api_path} tagged with inexact_signature!"


def test_zero_duplicate_placeholder_operand_signatures() -> None:
    """Verify zero duplicate placeholder operand signatures in AMD RDNA and NVIDIA SASS dumps."""
    # 1. NVIDIA SASS
    sass_data = get_bundled_json("nvidia_sass_exhaustive.json")
    for item in sass_data:
        op_sigs = item.get("operand_signatures", [])
        sig_tuples = [tuple(sig) for sig in op_sigs if isinstance(sig, list)]
        # No duplicate operand signatures for the same instruction
        assert len(sig_tuples) == len(
            set(sig_tuples)
        ), f"Duplicate operand signatures found in SASS mnemonic {item.get('mnemonic')}"

    # 2. AMD RDNA
    rdna_data = get_bundled_json("amd_rdna_exhaustive.json")
    for item in rdna_data:
        op_sigs = item.get("operand_signatures", [])
        sig_tuples = [tuple(sig) for sig in op_sigs if isinstance(sig, list)]
        assert len(sig_tuples) == len(
            set(sig_tuples)
        ), f"Duplicate operand signatures found in RDNA mnemonic {item.get('mnemonic')}"


def test_separation_of_ssa_operands_from_attributes_in_dialects() -> None:
    """Verify complete separation of SSA operands from buildable attributes in MLIR and StableHLO."""
    # 1. MLIR
    mlir_data = get_bundled_json("mlir_exhaustive.json")
    mlir_ops = (
        mlir_data
        if isinstance(mlir_data, list)
        else [op for items in mlir_data.get("categories", {}).values() for op in items]
    )
    for op in mlir_ops:
        operands: List[Any] = op.get("operands", [])
        attributes: List[Any] = op.get("attributes", [])

        op_names: Set[str] = {
            str(o.get("name") if isinstance(o, dict) else o) for o in operands
        }
        attr_names: Set[str] = {
            str(a.get("name") if isinstance(a, dict) else a) for a in attributes
        }

        # Intersection between operands and attributes should be empty
        overlap = op_names.intersection(attr_names)
        assert (
            len(overlap) == 0
        ), f"Overlap between SSA operands and attributes in MLIR op {op.get('name')}: {overlap}"

    # 2. StableHLO
    hlo_data = get_bundled_json("stablehlo_exhaustive.json")
    hlo_ops = (
        hlo_data
        if isinstance(hlo_data, list)
        else [op for items in hlo_data.get("categories", {}).values() for op in items]
    )
    for op in hlo_ops:
        operands = op.get("operands", [])
        attributes = op.get("attributes", [])

        op_names = {str(o.get("name") if isinstance(o, dict) else o) for o in operands}
        attr_names = {
            str(a.get("name") if isinstance(a, dict) else a) for a in attributes
        }

        overlap = op_names.intersection(attr_names)
        assert (
            len(overlap) == 0
        ), f"Overlap between SSA operands and attributes in StableHLO op {op.get('name')}: {overlap}"
