"""Tests for the StableHLO framework extractor."""

import json
from typing import Any
from unittest import mock

from ml_framework_snapshots.frameworks import stablehlo as stablehlo_fw
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_stablehlo_collect_real() -> None:
    """Test collecting from the real stablehlo_exhaustive.json file."""
    api = stablehlo_fw.collect_api(SemanticTier.UTIL)
    assert len(api) > 0
    # AbsOp should be present
    abs_ops = [ref for ref in api if ref.name == "AbsOp"]
    assert len(abs_ops) == 1
    assert abs_ops[0].api_path == "stablehlo.abs"
    assert len(abs_ops[0].params) == 1
    assert abs_ops[0].params[0].name == "operand"


def test_stablehlo_collect_mocked(mocker: Any) -> None:
    """Test the stablehlo API collector with mocked data.

    Args:
        mocker: pytest-mock fixture.
    """
    mocker.patch("os.path.exists", return_value=True)

    mock_json_data = [
        {
            "api_path": "stablehlo.custom_op",
            "dialect": "stablehlo",
            "class_name": "CustomOp",
            "operands": ["in0", "in1"],
            "attributes": ["attr0"],
            "description": "Custom StableHLO operation",
        },
        {
            "api_path": "stablehlo.no_class",
            "dialect": "stablehlo",
            "operands": [],
            "attributes": [],
        },
    ]

    mocker.patch(
        "builtins.open", mocker.mock_open(read_data=json.dumps(mock_json_data))
    )

    api = stablehlo_fw.collect_api(SemanticTier.UTIL)
    assert len(api) == 2
    assert api[0].name == "CustomOp"
    assert api[0].api_path == "stablehlo.custom_op"
    assert len(api[0].params) == 3
    assert api[0].params[0].name == "in0"
    assert api[0].params[0].kind == "POSITIONAL_OR_KEYWORD"
    assert api[0].params[2].name == "attr0"
    assert api[0].params[2].kind == "KEYWORD_ONLY"
    assert api[0].docstring == "Custom StableHLO operation"

    # Default class name fallback
    assert api[1].name == "UnknownOp"

    # Test rich TableGen ODS metadata with dict operands, regions, results, and traits
    rich_op = [
        {
            "api_path": "stablehlo.reduce",
            "dialect": "stablehlo",
            "class_name": "ReduceOp",
            "operands": [{"name": "inputs", "type": "Variadic<HLO_Tensor>"}],
            "attributes": [{"name": "dimensions", "type": "I64ElementsAttr"}],
            "regions": ["body"],
            "results": [{"type": "HLO_Tensor"}],
            "traits": ["Commutative", "SameOperandsAndResultType"],
            "description": "Reduce operation.",
        }
    ]
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(rich_op)))
    rich_api = stablehlo_fw.collect_api(SemanticTier.UTIL)
    assert len(rich_api) == 1
    assert rich_api[0].name == "ReduceOp"
    assert rich_api[0].returns_type == "HLO_Tensor"
    assert "Traits: Commutative" in (rich_api[0].docstring or "")
    assert "Regions: body" in (rich_api[0].docstring or "")
    assert any(p.name == "body" for p in rich_api[0].params)

    # Test wrong category
    assert stablehlo_fw.collect_api(SemanticTier.LAYER) == []

    # Test parser/OSError exception
    mocker.patch("builtins.open", side_effect=OSError)
    assert stablehlo_fw.collect_api(SemanticTier.UTIL) == []

    # Test file missing
    mocker.patch("os.path.exists", return_value=False)
    assert stablehlo_fw.collect_api(SemanticTier.UTIL) == []


def test_stablehlo_regions_and_structured_schemas() -> None:
    """Test that StableHLO operations model region signatures and structured attribute schemas."""
    refs = stablehlo_fw.collect_api(SemanticTier.UTIL)
    ref_map = {r.api_path: r for r in refs}

    assert "stablehlo.reduce" in ref_map
    reduce_op = ref_map["stablehlo.reduce"]
    assert "regions" in reduce_op.domain_metadata
    assert "body" in reduce_op.domain_metadata["regions"]
    assert "block_arguments" in reduce_op.domain_metadata["regions"]["body"]
    assert reduce_op.domain_metadata["regions"]["body"]["block_arguments"] == [
        "tensor<T>",
        "tensor<T>",
    ]

    assert "stablehlo.while" in ref_map
    while_op = ref_map["stablehlo.while"]
    assert "cond" in while_op.domain_metadata["regions"]
    assert "body" in while_op.domain_metadata["regions"]

    assert "stablehlo.sort" in ref_map
    sort_op = ref_map["stablehlo.sort"]
    assert "comparator" in sort_op.domain_metadata["regions"]
    assert sort_op.domain_metadata["regions"]["comparator"]["block_arguments"] == [
        "tensor<T>",
        "tensor<T>",
    ]

    assert "structured_attribute_schemas" in reduce_op.domain_metadata
    schemas = reduce_op.domain_metadata["structured_attribute_schemas"]
    assert "DotDimensionNumbersAttr" in schemas
    assert "ConvDimensionNumbersAttr" in schemas
    assert "ComparisonDirectionAttr" in schemas
    assert "PrecisionAttr" in schemas


def test_validate_dot_dimension_numbers() -> None:
    """Test validation of DotDimensionNumbersAttr."""
    # Valid
    valid_dot = {
        "lhs_batch_dimensions": [0],
        "rhs_batch_dimensions": [0],
        "lhs_contracting_dimensions": [2],
        "rhs_contracting_dimensions": [1],
    }
    assert (
        stablehlo_fw.validate_dot_dimension_numbers(valid_dot, lhs_rank=3, rhs_rank=3)
        == []
    )

    # Count mismatch
    bad_count = {
        "lhs_batch_dimensions": [0, 1],
        "rhs_batch_dimensions": [0],
        "lhs_contracting_dimensions": [2],
        "rhs_contracting_dimensions": [1, 2],
    }
    errs = stablehlo_fw.validate_dot_dimension_numbers(bad_count)
    assert any("Batch dimension count mismatch" in e for e in errs)
    assert any("Contracting dimension count mismatch" in e for e in errs)

    # Disjointness overlap
    overlap_lhs = {
        "lhs_batch_dimensions": [0],
        "rhs_batch_dimensions": [0],
        "lhs_contracting_dimensions": [0],
        "rhs_contracting_dimensions": [1],
    }
    errs_overlap = stablehlo_fw.validate_dot_dimension_numbers(overlap_lhs)
    assert any(
        "LHS batch and contracting dimensions must be disjoint" in e
        for e in errs_overlap
    )

    overlap_rhs = {
        "lhs_batch_dimensions": [0],
        "rhs_batch_dimensions": [1],
        "lhs_contracting_dimensions": [1],
        "rhs_contracting_dimensions": [1],
    }
    assert any(
        "RHS batch and contracting dimensions must be disjoint" in e
        for e in stablehlo_fw.validate_dot_dimension_numbers(overlap_rhs)
    )

    # Rank bounds
    out_of_bounds = {
        "lhs_batch_dimensions": [5],
        "rhs_batch_dimensions": [0],
        "lhs_contracting_dimensions": [1],
        "rhs_contracting_dimensions": [5],
    }
    errs_oob = stablehlo_fw.validate_dot_dimension_numbers(
        out_of_bounds, lhs_rank=2, rhs_rank=2
    )
    assert any("LHS dimension 5" in e for e in errs_oob)
    assert any("RHS dimension 5" in e for e in errs_oob)


def test_validate_conv_dimension_numbers() -> None:
    """Test validation of ConvDimensionNumbersAttr."""
    # Valid 2D conv
    valid_conv = {
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
    assert (
        stablehlo_fw.validate_conv_dimension_numbers(
            valid_conv, input_rank=4, kernel_rank=4, output_rank=4
        )
        == []
    )

    # Spatial count mismatch
    bad_spatial = dict(valid_conv, input_spatial_dimensions=[2])
    errs_spatial = stablehlo_fw.validate_conv_dimension_numbers(bad_spatial)
    assert any("Spatial dimensions count mismatch" in e for e in errs_spatial)

    # Overlapping input dimensions
    bad_in = dict(valid_conv, input_feature_dimension=0)
    errs_in = stablehlo_fw.validate_conv_dimension_numbers(bad_in)
    assert any(
        "Input dimensions in ConvDimensionNumbersAttr must be distinct" in e
        for e in errs_in
    )

    # Input rank out of bounds
    bad_in_rank = dict(valid_conv, input_batch_dimension=4)
    assert any(
        "exceeds input rank" in e
        for e in stablehlo_fw.validate_conv_dimension_numbers(bad_in_rank, input_rank=4)
    )

    # Overlapping kernel dimensions
    bad_k = dict(valid_conv, kernel_output_feature_dimension=1)
    assert any(
        "Kernel dimensions in ConvDimensionNumbersAttr must be distinct" in e
        for e in stablehlo_fw.validate_conv_dimension_numbers(bad_k)
    )
    bad_k_rank = dict(valid_conv, kernel_input_feature_dimension=4)
    assert any(
        "exceeds kernel rank" in e
        for e in stablehlo_fw.validate_conv_dimension_numbers(bad_k_rank, kernel_rank=4)
    )

    # Output disjointness
    bad_out = dict(valid_conv, output_feature_dimension=0)
    assert any(
        "Output dimensions in ConvDimensionNumbersAttr must be distinct" in e
        for e in stablehlo_fw.validate_conv_dimension_numbers(bad_out)
    )
    bad_out_rank = dict(valid_conv, output_batch_dimension=4)
    assert any(
        "exceeds output rank" in e
        for e in stablehlo_fw.validate_conv_dimension_numbers(
            bad_out_rank, output_rank=4
        )
    )

    # Empty dictionary
    assert stablehlo_fw.validate_conv_dimension_numbers({}) == []


def test_validate_scatter_and_gather_dimension_numbers() -> None:
    """Test Scatter and Gather dimension numbers validation."""
    # Scatter disjointness
    valid_scatter = {
        "update_window_dims": [1, 2],
        "inserted_window_dims": [0],
        "scatter_dims_to_operand_dims": [0],
        "index_vector_dim": 1,
    }
    assert stablehlo_fw.validate_scatter_dimension_numbers(valid_scatter) == []

    overlap_scatter = dict(valid_scatter, inserted_window_dims=[1])
    assert any(
        "must be disjoint" in e
        for e in stablehlo_fw.validate_scatter_dimension_numbers(overlap_scatter)
    )

    # Gather disjointness
    valid_gather = {
        "offset_dims": [1, 2],
        "collapsed_slice_dims": [0],
        "start_index_map": [0],
        "index_vector_dim": 1,
    }
    assert stablehlo_fw.validate_gather_dimension_numbers(valid_gather) == []

    overlap_gather = dict(valid_gather, collapsed_slice_dims=[1])
    assert any(
        "must be disjoint" in e
        for e in stablehlo_fw.validate_gather_dimension_numbers(overlap_gather)
    )


def test_validate_stablehlo_region() -> None:
    """Test region arguments and terminator constraints for reduce, while, and sort."""
    # Reduce
    assert (
        stablehlo_fw.validate_stablehlo_region(
            "stablehlo.reduce", "body", ["tensor<f32>", "tensor<f32>"], ["tensor<f32>"]
        )
        == []
    )
    err_reduce_odd = stablehlo_fw.validate_stablehlo_region(
        "stablehlo.reduce", "body", ["tensor<f32>"], ["tensor<f32>"]
    )
    assert any(
        "takes 2 scalar arguments per reduce operand" in e for e in err_reduce_odd
    )

    err_reduce_yield = stablehlo_fw.validate_stablehlo_region(
        "stablehlo.reduce", "body", ["tensor<f32>", "tensor<f32>"], []
    )
    assert any("terminator yield count" in e for e in err_reduce_yield)

    # While
    assert (
        stablehlo_fw.validate_stablehlo_region(
            "stablehlo.while", "cond", ["tensor<i32>"], ["tensor<i1>"]
        )
        == []
    )
    err_while_cond = stablehlo_fw.validate_stablehlo_region(
        "stablehlo.while", "cond", ["tensor<i32>"], ["tensor<i32>"]
    )
    assert any("must terminate with tensor<i1>" in e for e in err_while_cond)

    assert (
        stablehlo_fw.validate_stablehlo_region(
            "stablehlo.while", "body", ["tensor<i32>"], ["tensor<i32>"]
        )
        == []
    )
    err_while_body = stablehlo_fw.validate_stablehlo_region(
        "stablehlo.while", "body", ["tensor<i32>"], ["tensor<f32>"]
    )
    assert any("must match block arguments" in e for e in err_while_body)

    # Sort
    assert (
        stablehlo_fw.validate_stablehlo_region(
            "stablehlo.sort",
            "comparator",
            ["tensor<f32>", "tensor<f32>"],
            ["tensor<i1>"],
        )
        == []
    )
    err_sort_args = stablehlo_fw.validate_stablehlo_region(
        "stablehlo.sort", "comparator", ["tensor<f32>"], ["tensor<i1>"]
    )
    assert any("takes 2 scalar arguments" in e for e in err_sort_args)

    err_sort_yield = stablehlo_fw.validate_stablehlo_region(
        "stablehlo.sort", "comparator", ["tensor<f32>", "tensor<f32>"], ["tensor<f32>"]
    )
    assert any("must terminate with tensor<i1>" in e for e in err_sort_yield)

    # Unknown op or region
    assert (
        stablehlo_fw.validate_stablehlo_region("stablehlo.unknown", "body", [], [])
        == []
    )
    assert (
        stablehlo_fw.validate_stablehlo_region("stablehlo.while", "other", [], []) == []
    )
    assert (
        stablehlo_fw.validate_stablehlo_region("stablehlo.sort", "other", [], []) == []
    )


def test_validate_broadcast_and_binary_broadcasting() -> None:
    """Test broadcast_in_dim and binary broadcast invariant validation."""
    # broadcast_in_dim valid
    assert (
        stablehlo_fw.validate_broadcast_in_dim(
            operand_shape=[1, 3], result_shape=[2, 3], broadcast_dimensions=[0, 1]
        )
        == []
    )

    # broadcast_dimensions length mismatch
    err_len = stablehlo_fw.validate_broadcast_in_dim(
        operand_shape=[1, 3], result_shape=[2, 3], broadcast_dimensions=[0]
    )
    assert any("must match operand rank" in e for e in err_len)

    # broadcast_dimensions out of bounds
    err_oob = stablehlo_fw.validate_broadcast_in_dim(
        operand_shape=[3], result_shape=[2, 3], broadcast_dimensions=[5]
    )
    assert any("out of bounds for result rank" in e for e in err_oob)

    # broadcast dimension size mismatch
    err_mismatch = stablehlo_fw.validate_broadcast_in_dim(
        operand_shape=[4], result_shape=[2, 3], broadcast_dimensions=[1]
    )
    assert any("Broadcast dimension mismatch" in e for e in err_mismatch)

    # binary broadcast valid
    ok, shape, errs = stablehlo_fw.validate_binary_broadcast([2, 1], [3])
    assert ok is True
    assert shape == [2, 3]
    assert errs == []

    # binary broadcast dynamic
    ok_dyn, shape_dyn, _ = stablehlo_fw.validate_binary_broadcast([2, "?"], [2, 3])
    assert ok_dyn is True
    assert shape_dyn == [2, "?"]

    # binary broadcast incompatible
    bad_ok, _, bad_errs = stablehlo_fw.validate_binary_broadcast([2, 4], [2, 3])
    assert bad_ok is False
    assert any("Incompatible broadcast dimensions" in e for e in bad_errs)


def test_cli_check_stablehlo(capsys: Any, tmp_path: Any) -> None:
    """Test CLI check-stablehlo subcommand.

    Args:
        capsys: Pytest capsys fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_framework_snapshots.cli import cmd_check_stablehlo
    import argparse
    import pytest

    # 1. Valid op
    args_valid = argparse.Namespace(
        op_name="stablehlo.abs",
        operands_count=1,
        attributes=None,
        file=None,
    )
    cmd_check_stablehlo(args_valid)
    assert "StableHLO Operation 'stablehlo.abs' is valid." in capsys.readouterr().out

    # 2. Invalid op
    args_invalid = argparse.Namespace(
        op_name="stablehlo.nonexistent_op",
        operands_count=None,
        attributes=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_stablehlo(args_invalid)

    # 3. Missing arguments
    args_empty = argparse.Namespace(
        op_name=None,
        operands_count=None,
        attributes=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_stablehlo(args_empty)

    # 4. Valid file
    valid_file = tmp_path / "valid.mlir"
    valid_file.write_text(
        "%0 = stablehlo.abs(%arg0) : tensor<4xf32>\n", encoding="utf-8"
    )
    args_file_valid = argparse.Namespace(
        file=str(valid_file),
        op_name=None,
        operands_count=None,
        attributes=None,
    )
    cmd_check_stablehlo(args_file_valid)
    assert "StableHLO snippet verified compliant" in capsys.readouterr().out

    # 5. Nonexistent file
    args_file_missing = argparse.Namespace(
        file=str(tmp_path / "missing.mlir"),
        op_name=None,
        operands_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_stablehlo(args_file_missing)

    # 6. Invalid file
    invalid_file = tmp_path / "invalid.mlir"
    invalid_file.write_text("%0 = stablehlo.fake_op(%arg0)\n", encoding="utf-8")
    args_file_invalid = argparse.Namespace(
        file=str(invalid_file),
        op_name=None,
        operands_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_stablehlo(args_file_invalid)


def test_stablehlo_collect_api_variants() -> None:
    """Test collect_api envelope parsing with categories, dict operations, and list."""
    # 1. Envelope dict with non-list category
    mock_envelope = {
        "schema_version": "1.0.0",
        "target": "stablehlo",
        "categories": {
            "UTIL": [{"api_path": "stablehlo.test_op1", "name": "TestOp1"}],
            "metadata": "non_list_val",
        },
    }
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_envelope))
    ):
        refs = stablehlo_fw.collect_api(SemanticTier.UTIL)
        assert len(refs) == 1
        assert refs[0].api_path == "stablehlo.test_op1"

    # 2. Dict with operations
    mock_dict = {
        "operations": [{"api_path": "stablehlo.test_op2", "name": "TestOp2"}],
    }
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_dict))):
        refs = stablehlo_fw.collect_api(SemanticTier.UTIL)
        assert len(refs) == 1
        assert refs[0].api_path == "stablehlo.test_op2"

    # 3. Raw list
    mock_list = [{"api_path": "stablehlo.test_op3", "name": "TestOp3"}]
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_list))):
        refs = stablehlo_fw.collect_api(SemanticTier.UTIL)
        assert len(refs) == 1
        assert refs[0].api_path == "stablehlo.test_op3"
