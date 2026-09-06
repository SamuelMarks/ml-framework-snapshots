"""Tests for the StableHLO framework extractor."""

import json
from typing import Any

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
