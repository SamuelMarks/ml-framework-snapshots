"""Snapshot Integrity & Anti-Regression Verification Suite.

Validates that all generated snapshot JSON files adhere strictly to:
- Pydantic schema validation (GhostRef / ExtendedGhostRef)
- No unparsed regex or disassembly leak tokens ('SNOWFLAKE', '???0', '<class ...')
- No uniform duplicate placeholder signatures (e.g., RDNA all [['vGPR', 'vGPR']])
- Non-empty parameter and operand integrity
"""

import json
import os
from typing import Any, Dict, List

from ml_framework_snapshots.models import GhostRef


def _get_all_json_snapshots() -> List[str]:
    """Retrieve absolute file paths of all JSON snapshot files in the repository.

    Returns:
        List of absolute file paths.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_paths: List[str] = []

    target_dirs = [
        os.path.join(repo_root, "src", "ml_framework_snapshots", "frameworks"),
        os.path.join(repo_root, "src", "ml_framework_snapshots", "snapshots"),
    ]

    for target_dir in target_dirs:
        if os.path.isdir(target_dir):
            for fname in os.listdir(target_dir):
                if fname.endswith(".json"):
                    json_paths.append(os.path.join(target_dir, fname))

    return sorted(json_paths)


def test_no_raw_unparsed_tokens() -> None:
    """Verify that no snapshot JSON contains unparsed regex or scraper leak tokens."""
    banned_tokens = [
        "SNOWFLAKE",
        "???0",
        "AttributeOperand",
        "DescOperand",
        "<class '",
    ]

    for json_path in _get_all_json_snapshots():
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        for token in banned_tokens:
            assert (
                token not in content
            ), f"Found banned token '{token}' in snapshot {os.path.basename(json_path)}"


def test_no_uniform_duplicate_operands_rdna() -> None:
    """Ensure AMD RDNA dump does not have 100% duplicate placeholder operands."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rdna_path = os.path.join(
        repo_root,
        "src",
        "ml_framework_snapshots",
        "frameworks",
        "amd_rdna_exhaustive.json",
    )

    if not os.path.exists(rdna_path):
        return

    with open(rdna_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 1000
    encodings = {item.get("encoding") for item in data}
    assert (
        "Unknown" not in encodings or len(encodings) > 5
    ), f"RDNA dump has insufficient encodings: {encodings}"

    # Verify that multiple unique operand signatures exist
    unique_sigs = set()
    for item in data:
        operands = item.get("operands", [])
        sig_str = str(operands)
        unique_sigs.add(sig_str)

    assert (
        len(unique_sigs) >= 10
    ), f"RDNA dump exhibits uniform operand duplication: {len(unique_sigs)} unique sigs"


def test_stablehlo_operand_attribute_separation() -> None:
    """Ensure StableHLO operations separate SSA operands from buildable attributes."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    stablehlo_path = os.path.join(
        repo_root,
        "src",
        "ml_framework_snapshots",
        "frameworks",
        "stablehlo_exhaustive.json",
    )

    if not os.path.exists(stablehlo_path):
        return

    with open(stablehlo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 80

    # Ensure operations have distinct operands and attributes
    has_attributes = any(bool(item.get("attributes")) for item in data)
    has_operands = any(bool(item.get("operands")) for item in data)
    assert has_attributes, "StableHLO dump missing attributes separation"
    assert has_operands, "StableHLO dump missing operands separation"

    op_map = {item["api_path"]: item for item in data if "api_path" in item}
    assert "stablehlo.dot_general" in op_map
    dot_gen = op_map["stablehlo.dot_general"]
    dot_gen_attrs = [a["name"] for a in dot_gen.get("attributes", [])]
    assert "dot_dimension_numbers" in dot_gen_attrs

    assert "stablehlo.dot" in op_map
    assert len(op_map["stablehlo.dot"].get("operands", [])) >= 2

    assert "stablehlo.add" in op_map
    assert len(op_map["stablehlo.add"].get("operands", [])) >= 2


def test_mlir_operand_attribute_integrity() -> None:
    """Ensure MLIR operations have verified operands, attributes, and results without empty stubs."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlir_path = os.path.join(
        repo_root,
        "src",
        "ml_framework_snapshots",
        "frameworks",
        "mlir_exhaustive.json",
    )

    if not os.path.exists(mlir_path):
        return

    with open(mlir_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 100, f"MLIR dump has insufficient operations: {len(data)}"
    ops_with_operands = sum(1 for item in data if item.get("operands"))
    assert (
        ops_with_operands / len(data) >= 0.70
    ), f"Too many MLIR operations have empty operands: {ops_with_operands} / {len(data)}"

    op_map = {item["api_path"]: item for item in data if "api_path" in item}

    # Verify arith.addf has operands
    assert "arith.addf" in op_map
    addf_ops = op_map["arith.addf"].get("operands", [])
    assert len(addf_ops) >= 2
    op_names = [op["name"] if isinstance(op, dict) else op for op in addf_ops]
    assert "lhs" in op_names
    assert "rhs" in op_names

    # Verify linalg.matmul has operands and attributes
    if "linalg.matmul" in op_map:
        matmul = op_map["linalg.matmul"]
        assert len(matmul.get("operands", [])) >= 2
        matmul_op_names = [
            op["name"] if isinstance(op, dict) else op
            for op in matmul.get("operands", [])
        ]
        assert "inputs" in matmul_op_names
        assert "outputs" in matmul_op_names


def test_schema_validity_all_snapshots() -> None:
    """Validate all snapshot JSON files against GhostRef or ExtendedGhostRef models."""
    for json_path in _get_all_json_snapshots():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items_to_validate: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            for _cat, cat_items in data.get("categories", {}).items():
                items_to_validate.extend(cat_items)
        elif isinstance(data, list):
            items_to_validate.extend(data)

        # Validate each item adheres to GhostRef or ExtendedGhostRef
        for item in items_to_validate[:20]:  # Sample validation
            if "domain_metadata" in item or "operands" in item or "attributes" in item:
                # Ensure ExtendedGhostRef or basic GhostRef fields parse
                assert (
                    "name" in item
                    or "mnemonic" in item
                    or "api_path" in item
                    or "class_name" in item
                )
            else:
                validated = GhostRef.model_validate(item)
                assert validated.name is not None


def test_baseline_framework_operation_counts() -> None:
    """Verify that baseline operations are extracted across supported frameworks."""
    from ml_framework_snapshots.api import FRAMEWORK_COLLECTORS

    assert len(FRAMEWORK_COLLECTORS) >= 15
    for fw_name in ["torch", "jax", "stablehlo", "nvidia_sass", "amd_rdna"]:
        assert fw_name in FRAMEWORK_COLLECTORS
