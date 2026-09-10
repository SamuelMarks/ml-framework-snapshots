"""Offline mode verification test suite.

Ensures that all MCP grounding tools (check_hallucination, check_sass_instruction,
check_rdna_instruction, check_ptx_instruction, check_mlir_op, check_stablehlo_op)
operate strictly without network access, querying bundled ground-truth JSON files.
"""

import socket
from typing import Any
from unittest import mock
import pytest

from ml_framework_snapshots import mcp_server


@pytest.fixture(autouse=True)
def forbid_network_connections() -> Any:
    """Fixture that forbids any outgoing socket connections during offline test execution.

    Yields:
        None after patching socket.socket.connect.
    """

    def _block_connect(*args: Any, **kwargs: Any) -> None:
        """Blocked connect function.

        Args:
            *args: Positional arguments for socket.connect.
            **kwargs: Keyword arguments for socket.connect.

        Raises:
            RuntimeError: Always raised to simulate offline isolated environment.
        """
        raise RuntimeError("Network access forbidden in offline mode")

    with mock.patch.object(socket.socket, "connect", side_effect=_block_connect):
        yield


def test_offline_mcp_sass_grounding() -> None:
    """Verify SASS instruction verification operates offline using bundled JSON."""
    res = mcp_server.check_sass_instruction("FADD", operands=["R0", "R1", "R2"])
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True


def test_offline_mcp_rdna_grounding() -> None:
    """Verify AMD RDNA instruction verification operates offline using bundled JSON."""
    res = mcp_server.check_rdna_instruction("v_add_f32", operands=["v0", "v1", "v2"])
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True


def test_offline_mcp_ptx_grounding() -> None:
    """Verify NVIDIA PTX instruction verification operates offline using bundled JSON."""
    res = mcp_server.check_ptx_instruction(
        "add",
        types=[".f32"],
        operands=["%f0", "%f1", "%f2"],
    )
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True


def test_offline_mcp_mlir_grounding() -> None:
    """Verify MLIR operation verification operates offline using bundled JSON."""
    res = mcp_server.check_mlir_op("arith.addf")
    assert res["is_valid"] is True
    assert res["op_exists"] is True


def test_offline_mcp_stablehlo_grounding() -> None:
    """Verify StableHLO operation verification operates offline using bundled JSON."""
    res = mcp_server.check_stablehlo_op("stablehlo.add")
    assert res["is_valid"] is True
    assert res["op_exists"] is True


def test_offline_mcp_hallucination_check() -> None:
    """Verify API hallucination checking operates offline using local snapshot."""
    mcp_server._SNAPSHOT_CACHE["torch"] = {
        "_snapshot_source": "local_cache",
        "categories": {
            "ACTIVATION": [
                {
                    "name": "relu",
                    "api_path": "torch.relu",
                    "kind": "function",
                    "params": [{"name": "input", "kind": "POSITIONAL_OR_KEYWORD"}],
                }
            ]
        },
    }
    # Test checking a known PyTorch API offline
    res = mcp_server.check_hallucination(
        framework="torch",
        api_path="torch.relu",
        strict_kwargs=True,
    )
    assert res["api_exists"] is True
    assert res["is_hallucinated"] is False
    assert res["snapshot_source"] == "local_cache"


def test_offline_mcp_hallucination_accepted_kwargs() -> None:
    """Verify accepted_kwargs closes unconstrained kwargs loophole offline."""
    # Mock an API that has VAR_KEYWORD but specifies accepted_kwargs
    mcp_server._SNAPSHOT_CACHE["custom_fw"] = {
        "_snapshot_source": "local_cache",
        "categories": {
            "ARRAY_API": [
                {
                    "name": "tensor_op",
                    "api_path": "custom_fw.tensor_op",
                    "kind": "function",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "kwargs", "kind": "VAR_KEYWORD"},
                    ],
                    "accepted_kwargs": ["dtype", "device", "requires_grad"],
                }
            ]
        },
    }

    # 1. Valid accepted kwargs should pass
    valid_res = mcp_server.check_hallucination(
        framework="custom_fw",
        api_path="custom_fw.tensor_op",
        kwargs=["dtype", "device"],
    )
    assert valid_res["api_exists"] is True
    assert valid_res["is_hallucinated"] is False
    assert valid_res["invalid_kwargs"] == []
    assert valid_res["has_unconstrained_kwargs"] is False

    # 2. Hallucinated kwarg not in accepted_kwargs must be rejected
    invalid_res = mcp_server.check_hallucination(
        framework="custom_fw",
        api_path="custom_fw.tensor_op",
        kwargs=["dtype", "hallucinated_axis"],
    )
    assert invalid_res["api_exists"] is True
    assert invalid_res["is_hallucinated"] is True
    assert "hallucinated_axis" in invalid_res["invalid_kwargs"]
    assert invalid_res["has_unconstrained_kwargs"] is False


def test_offline_mcp_hallucination_missing_snapshot_diagnostic() -> None:
    """Verify informative offline diagnostic message when framework snapshot is absent."""
    res = mcp_server.check_hallucination(
        framework="nonexistent_framework_xyz",
        api_path="nonexistent_framework_xyz.foo",
    )
    assert res["api_exists"] is False
    assert res["is_hallucinated"] is True
    assert "could not be verified: no local ground-truth snapshot" in res["reason"]
    assert "To ground offline" in res["reason"]


def test_offline_mlir_decoupled_operands_and_regions() -> None:
    """Verify GhostMlirRef decoupled operands, regions, and successor block validation."""
    from ml_framework_snapshots.frameworks.mlir import (
        validate_mlir_region,
        validate_mlir_successors,
    )
    from ml_framework_snapshots.models import (
        ExtendedGhostParam,
        GhostMlirRef,
        IRParameterRole,
    )

    op_param = ExtendedGhostParam(
        name="lhs",
        kind="POSITIONAL_OR_KEYWORD",
        annotation="tensor<4xf32>",
        role=IRParameterRole.OPERAND,
    )
    attr_param = ExtendedGhostParam(
        name="predicate",
        kind="KEYWORD_ONLY",
        annotation="i64",
        role=IRParameterRole.ATTRIBUTE,
    )

    mlir_ref = GhostMlirRef(
        name="CmpFOp",
        api_path="arith.cmpf",
        kind="function",
        params=[op_param, attr_param],
        operands=[op_param],
        attributes={"predicate": {"type": "i64"}},
        type_constraints={"lhs": "AnyFloat"},
    )
    assert len(mlir_ref.params) == 2
    assert mlir_ref.operands is not None
    assert len(mlir_ref.operands) == 1
    assert mlir_ref.operands[0].name == "lhs"

    # Region validation
    reg_errs_valid = validate_mlir_region("scf.for", "body", ["index"], [])
    assert len(reg_errs_valid) == 0

    reg_errs_bad_arg = validate_mlir_region("scf.for", "body", ["f32"], [])
    assert any("must be 'index' type" in e for e in reg_errs_bad_arg)

    reg_errs_empty_arg = validate_mlir_region("scf.for", "body", [], [])
    assert any("must be 'index' type" in e for e in reg_errs_empty_arg)

    reg_errs_iter_match = validate_mlir_region(
        "scf.for", "body", ["index", "f32"], ["f32"]
    )
    assert len(reg_errs_iter_match) == 0

    reg_errs_iter_mismatch = validate_mlir_region(
        "scf.for", "body", ["index", "f32"], ["i32"]
    )
    assert any("must match loop-carried iter_args" in e for e in reg_errs_iter_mismatch)

    reg_errs_for_non_body = validate_mlir_region("scf.for", "other", ["index"], ["f32"])
    assert len(reg_errs_for_non_body) == 0

    reg_errs_while_valid = validate_mlir_region("scf.while", "before", [], ["i1"])
    assert len(reg_errs_while_valid) == 0

    reg_errs_while_invalid = validate_mlir_region(
        "scf.while", "before", [], ["tensor<f32>"]
    )
    assert any("condition boolean i1" in e for e in reg_errs_while_invalid)

    reg_errs_while_empty = validate_mlir_region("scf.while", "before", [], [])
    assert any("condition boolean i1" in e for e in reg_errs_while_empty)

    reg_errs_while_non_before = validate_mlir_region(
        "scf.while", "after", ["f32"], ["f32"]
    )
    assert len(reg_errs_while_non_before) == 0

    reg_errs_other_op = validate_mlir_region("arith.addf", "body", [], [])
    assert len(reg_errs_other_op) == 0

    # Successor validation
    succ_errs_valid = validate_mlir_successors("cf.br", ["^bb1"])
    assert len(succ_errs_valid) == 0

    succ_errs_br_mismatch = validate_mlir_successors("cf.br", ["^bb1", "^bb2"])
    assert any("requires exactly 1 successor block" in e for e in succ_errs_br_mismatch)

    succ_errs_invalid_label = validate_mlir_successors("cf.br", ["invalid_label"])
    assert any("expected prefix '^'" in e for e in succ_errs_invalid_label)

    succ_errs_cond_count = validate_mlir_successors("cf.cond_br", ["^bb1"])
    assert any("requires exactly 2 successor blocks" in e for e in succ_errs_cond_count)

    succ_errs_cond_valid = validate_mlir_successors("cf.cond_br", ["^bb1", "^bb2"])
    assert len(succ_errs_cond_valid) == 0

    succ_errs_expected_count_mismatch = validate_mlir_successors(
        "custom.op", ["^bb1"], expected_count=2
    )
    assert any(
        "Successor count mismatch" in e for e in succ_errs_expected_count_mismatch
    )

    succ_errs_expected_count_match = validate_mlir_successors(
        "custom.op", ["^bb1"], expected_count=1
    )
    assert len(succ_errs_expected_count_match) == 0

    succ_errs_other_op = validate_mlir_successors("custom.op", ["^bb1"])
    assert len(succ_errs_other_op) == 0


def test_offline_stablehlo_op_validation() -> None:
    """Verify validate_stablehlo_op checking dimension numbers and regions against operand ranks."""
    from ml_framework_snapshots.frameworks.stablehlo import validate_stablehlo_op

    # Valid dot_general
    valid_dot = validate_stablehlo_op(
        "stablehlo.dot_general",
        attributes={
            "dot_dimension_numbers": {
                "lhs_batch_dimensions": [0],
                "rhs_batch_dimensions": [0],
                "lhs_contracting_dimensions": [2],
                "rhs_contracting_dimensions": [1],
            }
        },
        operand_ranks=[3, 3],
    )
    assert len(valid_dot) == 0

    # Invalid dot_general with out-of-bounds contracting dimension
    invalid_dot = validate_stablehlo_op(
        "stablehlo.dot_general",
        attributes={
            "dot_dimension_numbers": {
                "lhs_batch_dimensions": [0],
                "rhs_batch_dimensions": [0],
                "lhs_contracting_dimensions": [5],  # rank is 2
                "rhs_contracting_dimensions": [1],
            }
        },
        operand_ranks=[2, 2],
    )
    assert len(invalid_dot) > 0

    # Non-dict dot_dimension_numbers
    bad_type_dot = validate_stablehlo_op(
        "stablehlo.dot_general",
        attributes={"dot_dimension_numbers": "not_a_dict"},
    )
    assert any("must be a dictionary specification" in e for e in bad_type_dot)

    # Valid convolution
    valid_conv = validate_stablehlo_op(
        "stablehlo.convolution",
        attributes={
            "conv_dimension_numbers": {
                "input_batch_dimension": 0,
                "input_feature_dimension": 1,
                "input_spatial_dimensions": [2, 3],
                "kernel_input_feature_dimension": 1,
                "kernel_output_feature_dimension": 0,
                "kernel_spatial_dimensions": [2, 3],
                "output_batch_dimension": 0,
                "output_feature_dimension": 1,
                "output_spatial_dimensions": [2, 3],
            }
        },
    )
    assert len(valid_conv) == 0

    # Non-dict convolution
    bad_type_conv = validate_stablehlo_op(
        "stablehlo.convolution",
        attributes={"conv_dimension_numbers": "not_a_dict"},
    )
    assert any("must be a dictionary specification" in e for e in bad_type_conv)

    # Valid scatter
    valid_scatter = validate_stablehlo_op(
        "stablehlo.scatter",
        attributes={
            "scatter_dimension_numbers": {
                "update_window_dims": [1],
                "inserted_window_dims": [0],
                "scatter_dims_to_operand_dims": [0],
                "index_vector_dim": 1,
            }
        },
    )
    assert len(valid_scatter) == 0

    # Non-dict scatter
    bad_type_scatter = validate_stablehlo_op(
        "stablehlo.scatter",
        attributes={"scatter_dimension_numbers": 123},
    )
    assert any("must be a dictionary specification" in e for e in bad_type_scatter)

    # Valid gather
    valid_gather = validate_stablehlo_op(
        "stablehlo.gather",
        attributes={
            "gather_dimension_numbers": {
                "offset_dims": [1],
                "collapsed_slice_dims": [0],
                "start_index_map": [0],
                "index_vector_dim": 1,
            }
        },
    )
    assert len(valid_gather) == 0

    # Non-dict gather
    bad_type_gather = validate_stablehlo_op(
        "stablehlo.gather",
        attributes={"gather_dimension_numbers": ["not_a_dict"]},
    )
    assert any("must be a dictionary specification" in e for e in bad_type_gather)

    # Other non-dimension attribute
    other_attr_res = validate_stablehlo_op(
        "stablehlo.custom",
        attributes={"other_attr": 42},
    )
    assert len(other_attr_res) == 0

    # Valid reduce region
    valid_reduce = validate_stablehlo_op(
        "stablehlo.reduce",
        regions={
            "body": {
                "block_arguments": ["tensor<f32>", "tensor<f32>"],
                "yield_types": ["tensor<f32>"],
            }
        },
    )
    assert len(valid_reduce) == 0

    # Region with non-dict value
    bad_reg_res = validate_stablehlo_op(
        "stablehlo.reduce",
        regions={"body": "not_a_dict"},
    )
    assert any("takes 2 scalar arguments" in e for e in bad_reg_res)

    # Op without attributes or regions
    empty_res = validate_stablehlo_op("stablehlo.abs")
    assert len(empty_res) == 0


def test_offline_index_db_in_memory_fallback(monkeypatch: Any) -> None:
    """Verify init_db falls back to in-memory database if disk directory creation fails."""
    import os
    from ml_framework_snapshots.index import init_db

    def mock_makedirs_fail(path: str, exist_ok: bool = True) -> None:
        """Simulate read-only filesystem error.

        Args:
            path: Path to create.
            exist_ok: Whether existing directory is acceptable.

        Raises:
            OSError: Read-only filesystem.
        """
        raise OSError("Read-only file system")

    monkeypatch.setattr(os, "makedirs", mock_makedirs_fail)

    conn = init_db("/nonexistent_readonly_path/index.db")
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cur.fetchall()]
        assert "symbols" in tables
        assert "indexed_files" in tables
    finally:
        conn.close()


def test_offline_ghost_isa_structured_modifiers() -> None:
    """Verify GhostIsaRef handles structured modifiers and VOPD profiles."""
    from ml_framework_snapshots.models import GhostIsaRef

    isa_ref = GhostIsaRef(
        name="FADD",
        api_path="nvidia_sass.FADD",
        kind="instruction",
        instruction_modifiers=[".SAT", ".RN"],
        structured_modifiers={"rounding": ".RN", "saturation": True},
        vopd_profile={"supported": False},
    )
    assert isa_ref.structured_modifiers is not None
    assert isa_ref.structured_modifiers["rounding"] == ".RN"
    assert isa_ref.vopd_profile is not None
    assert isa_ref.vopd_profile["supported"] is False
