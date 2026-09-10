"""Tests for MLIR type hierarchy, dialect traits, and verification rules."""

import argparse
import json
from typing import Any
from unittest import mock
import pytest

from ml_framework_snapshots.frameworks import mlir
from ml_framework_snapshots.mcp_server import check_mlir_op
from ml_framework_snapshots.cli import cmd_check_mlir


def test_classify_mlir_type() -> None:
    """Test classification of MLIR types into type categories."""
    assert mlir.classify_mlir_type("tensor<4x?xf32>") == "tensor"
    assert mlir.classify_mlir_type("vector<[4]xi32>") == "vector"
    assert mlir.classify_mlir_type("memref<10x20xf32>") == "memref"
    assert mlir.classify_mlir_type("i32") == "integer"
    assert mlir.classify_mlir_type("si64") == "integer"
    assert mlir.classify_mlir_type("ui16") == "integer"
    assert mlir.classify_mlir_type("i1") == "integer"
    assert mlir.classify_mlir_type("index") == "index"
    assert mlir.classify_mlir_type("f32") == "float"
    assert mlir.classify_mlir_type("bf16") == "float"
    assert mlir.classify_mlir_type("complex<f32>") == "complex"
    assert mlir.classify_mlir_type("none") == "custom"


def test_validate_mlir_type() -> None:
    """Test syntax validation and TableGen constraint matching for MLIR types."""
    # Syntax
    assert mlir.validate_mlir_type("tensor<4xf32>") == []
    assert mlir.validate_mlir_type("vector<4xi32>") == []
    assert mlir.validate_mlir_type("memref<10xf32>") == []
    assert mlir.validate_mlir_type("i32") == []

    # Malformed types
    err_t = mlir.validate_mlir_type("tensor-bad")
    assert any("Malformed MLIR tensor type" in e for e in err_t)

    err_v = mlir.validate_mlir_type("vector-bad")
    assert any("Malformed MLIR vector type" in e for e in err_v)

    err_m = mlir.validate_mlir_type("memref-bad")
    assert any("Malformed MLIR memref type" in e for e in err_m)

    err_i = mlir.validate_mlir_type("integer_bad")
    assert any("Malformed MLIR integer type" in e for e in err_i)

    # Constraints
    assert (
        mlir.validate_mlir_type("tensor<4xf32>", constraint="RankedTensorOf<[F32]>")
        == []
    )
    err_ranked = mlir.validate_mlir_type(
        "tensor<*xf32>", constraint="RankedTensorOf<[F32]>"
    )
    assert any("RankedTensorOf" in e for e in err_ranked)

    assert mlir.validate_mlir_type("tensor<*xf32>", constraint="UnrankedTensorOf") == []
    err_unranked = mlir.validate_mlir_type(
        "tensor<4xf32>", constraint="UnrankedTensorOf"
    )
    assert any("UnrankedTensorOf" in e for e in err_unranked)

    assert mlir.validate_mlir_type("vector<4xf32>", constraint="VectorOf<[F32]>") == []
    err_vec = mlir.validate_mlir_type("tensor<4xf32>", constraint="VectorOf<[F32]>")
    assert any("VectorOf" in e for e in err_vec)

    assert mlir.validate_mlir_type("memref<4xf32>", constraint="MemRefOf<[F32]>") == []
    err_mem = mlir.validate_mlir_type("tensor<4xf32>", constraint="MemRefOf<[F32]>")
    assert any("MemRefOf" in e for e in err_mem)

    assert mlir.validate_mlir_type("i32", constraint="AnyInteger") == []
    assert mlir.validate_mlir_type("index", constraint="AnyInteger") == []
    err_int = mlir.validate_mlir_type("f32", constraint="AnyInteger")
    assert any("integer constraint" in e for e in err_int)

    assert mlir.validate_mlir_type("f32", constraint="AnyFloat") == []
    err_flt = mlir.validate_mlir_type("i32", constraint="AnyFloat")
    assert any("float constraint" in e for e in err_flt)

    # StableHLO constraints: HLO_StaticShapeTensor
    assert (
        mlir.validate_mlir_type(
            "tensor<4x8xf32>",
            constraint="HLO_StaticShapeTensorOrPerAxisQuantizedTensor",
        )
        == []
    )
    err_dynamic = mlir.validate_mlir_type(
        "tensor<?x8xf32>",
        constraint="HLO_StaticShapeTensorOrPerAxisQuantizedTensor",
    )
    assert any("static shape constraint" in e for e in err_dynamic)
    err_not_tensor = mlir.validate_mlir_type(
        "i32", constraint="HLO_StaticShapeTensorOrPerAxisQuantizedTensor"
    )
    assert any("tensor constraint" in e for e in err_not_tensor)


def test_validate_mlir_traits() -> None:
    """Test validation of MLIR dialect traits."""
    # SameOperandsAndResultType
    assert (
        mlir.validate_mlir_traits(
            "arith.addf",
            traits=["SameOperandsAndResultType"],
            operand_types=["tensor<4xf32>", "tensor<4xf32>"],
            result_types=["tensor<4xf32>"],
        )
        == []
    )
    err_same = mlir.validate_mlir_traits(
        "arith.addf",
        traits=["SameOperandsAndResultType"],
        operand_types=["tensor<4xf32>", "tensor<4xf32>"],
        result_types=["tensor<4xi32>"],
    )
    assert any("SameOperandsAndResultType" in e for e in err_same)

    # SameTypeOperands
    assert (
        mlir.validate_mlir_traits(
            "arith.cmpi",
            traits=["SameTypeOperands"],
            operand_types=["i32", "i32"],
        )
        == []
    )
    err_same_op = mlir.validate_mlir_traits(
        "arith.cmpi",
        traits=["SameTypeOperands"],
        operand_types=["i32", "i64"],
    )
    assert any("SameTypeOperands" in e for e in err_same_op)

    # AttrSizedOperandSegments
    assert (
        mlir.validate_mlir_traits(
            "test.custom",
            traits=["AttrSizedOperandSegments"],
            attributes=["operand_segment_sizes"],
        )
        == []
    )
    err_seg_missing = mlir.validate_mlir_traits(
        "test.custom",
        traits=["AttrSizedOperandSegments"],
        attributes=[],
    )
    assert any(
        "Missing required attribute 'operand_segment_sizes'" in e
        for e in err_seg_missing
    )

    err_seg_mismatch = mlir.validate_mlir_traits(
        "test.custom",
        traits=["AttrSizedOperandSegments"],
        operand_types=["i32", "i32"],
        structured_attributes={"operand_segment_sizes": [1, 2]},
    )
    assert any("AttrSizedOperandSegments mismatch" in e for e in err_seg_mismatch)

    # AttrSizedResultSegments
    assert (
        mlir.validate_mlir_traits(
            "test.custom",
            traits=["AttrSizedResultSegments"],
            attributes=["result_segment_sizes"],
        )
        == []
    )
    err_res_missing = mlir.validate_mlir_traits(
        "test.custom",
        traits=["AttrSizedResultSegments"],
        attributes=[],
    )
    assert any(
        "Missing required attribute 'result_segment_sizes'" in e
        for e in err_res_missing
    )


def test_check_mlir_op_with_traits() -> None:
    """Test check_mlir_op validating traits and type constraints."""
    # Valid arith.addf
    res_valid = check_mlir_op(
        "arith.addf",
        operands_count=2,
        operand_types=["tensor<4xf32>", "tensor<4xf32>"],
        result_types=["tensor<4xf32>"],
    )
    assert res_valid["is_valid"] is True

    # SameOperandsAndResultType violation
    res_bad = check_mlir_op(
        "arith.addf",
        operand_types=["tensor<4xf32>", "tensor<4xi32>"],
    )
    assert res_bad["is_valid"] is False
    assert any(
        "SameOperandsAndResultType" in e or "SameTypeOperands" in e
        for e in res_bad["errors"]
    )


def test_cli_check_mlir(capsys: Any, tmp_path: Any) -> None:
    """Test CLI check-mlir subcommand."""
    # 1. Valid op
    args_valid = argparse.Namespace(
        op_name="arith.addf",
        operands_count=2,
        attributes=None,
        file=None,
    )
    cmd_check_mlir(args_valid)
    assert "MLIR Operation 'arith.addf' is valid." in capsys.readouterr().out

    # 2. Invalid op
    args_invalid = argparse.Namespace(
        op_name="arith.nonexistent",
        operands_count=None,
        attributes=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_mlir(args_invalid)

    # 3. Missing arguments
    args_empty = argparse.Namespace(
        op_name=None,
        operands_count=None,
        attributes=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_mlir(args_empty)

    # 4. Valid file
    valid_file = tmp_path / "valid.mlir"
    valid_file.write_text("%0 = arith.addf(%a, %b) : f32\n", encoding="utf-8")
    args_file_valid = argparse.Namespace(
        file=str(valid_file),
        op_name=None,
        operands_count=None,
        attributes=None,
    )
    cmd_check_mlir(args_file_valid)
    assert "MLIR snippet verified compliant" in capsys.readouterr().out

    # 5. Nonexistent file
    args_file_missing = argparse.Namespace(
        file=str(tmp_path / "missing.mlir"),
        op_name=None,
        operands_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_mlir(args_file_missing)

    # 6. Invalid file
    invalid_file = tmp_path / "invalid.mlir"
    invalid_file.write_text("%0 = arith.fake_op(%a)\n", encoding="utf-8")
    args_file_invalid = argparse.Namespace(
        file=str(invalid_file),
        op_name=None,
        operands_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_mlir(args_file_invalid)


def test_mlir_collect_api(mocker: Any) -> None:
    """Test collect_api on MLIR framework module."""
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    # 1. Collect real
    refs = mlir.collect_api(SemanticTier.UTIL)
    assert len(refs) > 0
    assert any(r.name == "AddFOp" for r in refs)

    # 2. Non-util tier
    assert mlir.collect_api(SemanticTier.LAYER) == []

    # 3. Missing file
    mocker.patch("os.path.exists", return_value=False)
    assert mlir.collect_api(SemanticTier.UTIL) == []

    # 4. OSError
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", side_effect=OSError)
    assert mlir.collect_api(SemanticTier.UTIL) == []


def test_validate_mlir_traits_edge_cases() -> None:
    """Test edge cases in validate_mlir_traits."""
    # Traits empty or no types
    assert mlir.validate_mlir_traits("test.op", traits=[]) == []
    assert (
        mlir.validate_mlir_traits(
            "test.op", traits=["SameTypeOperands"], operand_types=["i32"]
        )
        == []
    )
    assert (
        mlir.validate_mlir_traits(
            "test.custom",
            traits=["AttrSizedOperandSegments"],
            attributes=["operand_segment_sizes"],
            structured_attributes={"operand_segment_sizes": "not_a_list"},
        )
        == []
    )
    assert (
        mlir.validate_mlir_traits(
            "test.custom",
            traits=["AttrSizedOperandSegments"],
            operand_types=["i32", "i32"],
            attributes=["operand_segment_sizes"],
            structured_attributes={"operand_segment_sizes": [1, 1]},
        )
        == []
    )
    assert (
        mlir.validate_mlir_traits(
            "test.op",
            traits=["AttrSizedResultSegments"],
            attributes=["result_segment_sizes"],
            structured_attributes={"result_segment_sizes": [1]},
        )
        == []
    )

    # 5. Commutative trait
    assert (
        mlir.validate_mlir_traits(
            "test.add", traits=["Commutative"], operand_types=["i32", "i32"]
        )
        == []
    )
    err_comm = mlir.validate_mlir_traits(
        "test.add", traits=["Commutative"], operand_types=["i32"]
    )
    assert any("Commutative trait requires at least 2 operands" in e for e in err_comm)

    # 6. Elementwise trait
    assert (
        mlir.validate_mlir_traits(
            "test.op",
            traits=["Elementwise"],
            operand_types=["tensor<4xf32>", "tensor<4xf32>"],
        )
        == []
    )
    err_elem = mlir.validate_mlir_traits(
        "test.op",
        traits=["Elementwise"],
        operand_types=["tensor<4xf32>", "tensor<4xi32>"],
    )
    assert any("matching element types" in e for e in err_elem)

    # 7. IsolatedFromAbove trait
    assert (
        mlir.validate_mlir_traits(
            "test.region_op",
            traits=["IsolatedFromAbove"],
            structured_attributes={"has_external_captures": False},
        )
        == []
    )
    err_iso = mlir.validate_mlir_traits(
        "test.region_op",
        traits=["IsolatedFromAbove"],
        structured_attributes={"has_external_captures": True},
    )
    assert any("forbids implicit captures" in e for e in err_iso)

    # 8. Type validation missing branches (anytensor, index)
    assert mlir.validate_mlir_type("tensor<4xf32>", constraint="anytensor") == []
    err_anytensor = mlir.validate_mlir_type("i32", constraint="anytensor")
    assert any("expected tensor type" in e for e in err_anytensor)

    assert mlir.validate_mlir_type("index", constraint="index") == []
    err_idx = mlir.validate_mlir_type("f32", constraint="index")
    assert any("expected 'index' type" in e for e in err_idx)


def test_mlir_collect_api_variants() -> None:
    """Test collect_api envelope parsing with categories, dict operations, and list."""
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    # 1. Envelope dict with non-list category
    mock_envelope = {
        "schema_version": "1.0.0",
        "target": "mlir",
        "categories": {
            "UTIL": [{"api_path": "test.op1", "summary": "op1"}],
            "metadata": "non_list_val",
        },
    }
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_envelope))
    ):
        refs = mlir.collect_api(SemanticTier.UTIL)
        assert len(refs) == 1
        assert refs[0].api_path == "test.op1"

    # 2. Dict with operations
    mock_dict = {
        "operations": [{"api_path": "test.op2", "summary": "op2"}],
    }
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_dict))):
        refs = mlir.collect_api(SemanticTier.UTIL)
        assert len(refs) == 1
        assert refs[0].api_path == "test.op2"

    # 3. Raw list
    mock_list = [{"api_path": "test.op3", "summary": "op3"}]
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_list))):
        refs = mlir.collect_api(SemanticTier.UTIL)
        assert len(refs) == 1
        assert refs[0].api_path == "test.op3"
