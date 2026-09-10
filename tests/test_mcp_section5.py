"""Unit tests for Model Context Protocol (MCP) server Section 5 capabilities.

Tests version-aware MCP queries, SQLite FTS5 index lookups,
batch code block verification (check_code_block), and expanded semantic concept maps.
"""

import json
from typing import Any, Dict, Optional
from unittest import mock

from ml_framework_snapshots.mcp_server import (
    CONCEPT_ALIAS_MAP,
    check_code_block,
    check_hallucination,
    check_mlir_op,
    get_api_signature,
    get_framework_snapshot,
    handle_mcp_message,
    search_apis,
)


def test_version_aware_mcp_queries(mocker: Any) -> None:
    """Test version-aware queries across get_api_signature, search_apis, and check_hallucination.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap_v24 = {
        "categories": {
            "math": [
                {
                    "name": "op_v24",
                    "api_path": "torch.op_v24",
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
                }
            ]
        }
    }
    mock_snap_v23 = {
        "categories": {
            "math": [
                {
                    "name": "op_v23",
                    "api_path": "torch.op_v23",
                    "params": [{"name": "y", "kind": "POSITIONAL_OR_KEYWORD"}],
                }
            ]
        }
    }

    def mock_get_snap(framework: str, version: Any = None) -> Dict[str, Any]:
        """Mock framework snapshot getter returning version-specific mock snapshot.

        Args:
            framework: Framework identifier string.
            version: Optional version string.

        Returns:
            Dictionary containing mocked categories for version.
        """
        if version and version.startswith("2.4"):
            return mock_snap_v24
        return mock_snap_v23

    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        side_effect=mock_get_snap,
    )
    # Also bypass SQLite lookup so it queries snapshot
    mocker.patch(
        "ml_framework_snapshots.index.lookup_symbol",
        return_value=None,
    )
    mocker.patch(
        "ml_framework_snapshots.index.search_index",
        return_value=[],
    )

    # 1. get_api_signature with version
    sig_v24 = get_api_signature("torch", "torch.op_v24", version="2.4.0")
    assert sig_v24 is not None
    assert sig_v24["name"] == "op_v24"

    sig_v23 = get_api_signature("torch", "torch.op_v23", version="2.3.0")
    assert sig_v23 is not None
    assert sig_v23["name"] == "op_v23"

    # 2. search_apis with version
    search_v24 = search_apis("torch", "op_v24", version="2.4.0")
    assert "torch.op_v24" in search_v24

    search_v23 = search_apis("torch", "op_v23", version="2.3.0")
    assert "torch.op_v23" in search_v23

    # 3. check_hallucination with version
    res_v24 = check_hallucination("torch", "torch.op_v24", version="2.4.0")
    assert res_v24["is_hallucinated"] is False

    res_v23_fail = check_hallucination("torch", "torch.op_v24", version="2.3.0")
    assert res_v23_fail["is_hallucinated"] is True


def test_sqlite_fts5_index_integration_in_mcp(mocker: Any) -> None:
    """Test get_api_signature and search_apis delegating to local SQLite FTS5 index.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_symbol = {
        "name": "fast_op",
        "api_path": "torch.fast_op",
        "kind": "function",
        "params": [{"name": "input", "kind": "POSITIONAL_OR_KEYWORD"}],
    }
    mocker.patch(
        "ml_framework_snapshots.index.lookup_symbol",
        return_value=mock_symbol,
    )
    mocker.patch(
        "ml_framework_snapshots.index.search_index",
        return_value=[mock_symbol],
    )

    # get_api_signature returns fast SQLite result without loading snapshot
    sig = get_api_signature("torch", "torch.fast_op")
    assert sig is not None
    assert sig["api_path"] == "torch.fast_op"

    # search_apis returns fast SQLite results
    matches = search_apis("torch", "fast_op")
    assert "torch.fast_op" in matches


def test_expanded_semantic_concept_maps() -> None:
    """Test expanded CONCEPT_ALIAS_MAP covering normalization, attention, activations, quantization, and collectives."""
    required_concepts = [
        "normalization",
        "layer_norm",
        "rms_norm",
        "batch_norm",
        "group_norm",
        "attention",
        "scaled_dot_product_attention",
        "flash_attention",
        "paged_attention",
        "sliding_window_attention",
        "wgmma_mma_async",
        "activations",
        "gelu",
        "silu",
        "swiglu",
        "relu",
        "quick_gelu",
        "quantization",
        "quantize",
        "dequantize",
        "fp8_e4m3",
        "fp8_e5m2",
        "scaled_fp8_quant",
        "mx_fp4",
        "mxfp8",
        "collective",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "memcpy",
        "gather",
        "scatter",
    ]

    for concept in required_concepts:
        assert (
            concept in CONCEPT_ALIAS_MAP
        ), f"Concept '{concept}' missing from CONCEPT_ALIAS_MAP"
        aliases = CONCEPT_ALIAS_MAP[concept]
        assert "torch" in aliases
        assert len(aliases["torch"]) > 0

    # Test searching for concepts via search_apis
    assert any("layer_norm" in a for a in search_apis("torch", "normalization"))
    assert any(
        "scaled_dot_product_attention" in a for a in search_apis("torch", "attention")
    )
    assert any("gelu" in a for a in search_apis("torch", "activations"))
    assert any("all_reduce" in a for a in search_apis("torch", "collective"))
    assert any("gather" in a for a in search_apis("torch", "gather"))
    assert any("scatter" in a for a in search_apis("torch", "scatter"))

    # Test load_concept_map and custom ontology
    from ml_framework_snapshots.mcp_server import load_concept_map

    custom_map = {"custom_op": {"torch": ["torch.my_custom_op"]}}
    res_custom = search_apis("torch", "custom_op", custom_concept_map=custom_map)
    assert "torch.my_custom_op" in res_custom

    assert load_concept_map("/nonexistent_path_to_ontology.json") == {}

    # Test load_concept_map with non-dict JSON (e.g. list)
    with mock.patch("builtins.open", mock.mock_open(read_data="[1, 2, 3]")):
        with mock.patch("os.path.exists", return_value=True):
            assert load_concept_map("/custom_list.json") == {}

    # Test load_concept_map with invalid JSON (raises Exception)
    with mock.patch("builtins.open", mock.mock_open(read_data="{invalid_json")):
        with mock.patch("os.path.exists", return_value=True):
            assert load_concept_map("/bad.json") == {}

    # Test search_apis with custom_concept_map as file path string
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(custom_map))):
        with mock.patch("os.path.exists", return_value=True):
            res_str = search_apis(
                "torch", "custom_op", custom_concept_map="/custom.json"
            )
            assert "torch.my_custom_op" in res_str


def test_check_code_block_python(mocker: Any) -> None:
    """Test check_code_block batch verifying Python code with valid and hallucinated calls.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_torch = {
        "categories": {
            "math": [
                {
                    "name": "sum",
                    "api_path": "torch.sum",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "dim", "kind": "KEYWORD_ONLY"},
                        {"name": "keepdim", "kind": "KEYWORD_ONLY"},
                    ],
                },
                {
                    "name": "add",
                    "api_path": "torch.add",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
            ]
        }
    }

    def mock_get_snap(framework: str, version: Any = None) -> Dict[str, Any]:
        """Mock framework snapshot getter returning mock torch snapshot.

        Args:
            framework: Framework identifier string.
            version: Optional version string.

        Returns:
            Dictionary containing mocked categories for torch framework.
        """
        if framework == "torch":
            return mock_torch
        return {"categories": {}}

    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        side_effect=mock_get_snap,
    )
    mocker.patch("ml_framework_snapshots.index.lookup_symbol", return_value=None)
    mocker.patch("ml_framework_snapshots.index.search_index", return_value=[])

    code_snippet = """
import torch

def my_func(x):
    y = torch.add(x, 2)
    # Hallucinated kwarg 'axis' instead of 'dim'
    z = torch.sum(y, axis=0)
    # Nonexistent API
    w = torch.non_existent_fake_op(z)
    return w
"""
    res = check_code_block(code_snippet, framework="torch")
    assert res["total_analyzed"] == 3
    assert res["hallucinations_detected"] == 2
    assert res["is_valid"] is False

    targets = [f["target"] for f in res["findings"]]
    assert "torch.sum" in targets
    assert "torch.non_existent_fake_op" in targets

    # Valid snippet
    valid_code = """
import torch

y = torch.add(x, 2)
z = torch.sum(y, dim=0)
"""
    res_valid = check_code_block(valid_code, framework="torch")
    assert res_valid["is_valid"] is True
    assert res_valid["hallucinations_detected"] == 0


def test_check_code_block_assembly_and_ir() -> None:
    """Test check_code_block batch verifying SASS, RDNA, and MLIR assembly blocks."""
    # 1. MLIR block
    mlir_code = """
// MLIR dialect code
%0 = arith.muli %a, %b : i32
%1 = completely.non_existent_op %0
"""
    res_mlir = check_code_block(mlir_code, framework="mlir")
    assert res_mlir["total_analyzed"] == 2
    assert res_mlir["hallucinations_detected"] == 1
    assert res_mlir["findings"][0]["target"] == "completely.non_existent_op"

    # 2. NVIDIA SASS block
    sass_code = """
// SASS instructions
FFMA R0, R1, R2, R3;
NON_EXISTENT_SASS_OP R0, R1;
"""
    res_sass = check_code_block(sass_code, framework="nvidia_sass")
    assert res_sass["total_analyzed"] == 2
    assert res_sass["hallucinations_detected"] == 1
    assert res_sass["findings"][0]["target"] == "NON_EXISTENT_SASS_OP"

    # 3. AMD RDNA block
    rdna_code = """
// RDNA instructions
v_fma_f32 v0, v1, v2, v3
v_non_existent_rdna_op v0, v1
"""
    res_rdna = check_code_block(rdna_code, framework="amd_rdna")
    assert res_rdna["total_analyzed"] == 2
    assert res_rdna["hallucinations_detected"] == 1
    assert res_rdna["findings"][0]["target"] == "v_non_existent_rdna_op"


def test_handle_mcp_message_check_code_block() -> None:
    """Test handle_mcp_message dispatching check_code_block."""
    msg = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {
            "name": "check_code_block",
            "arguments": {
                "code": "v_non_existent_rdna_op v0, v1",
                "framework": "amd_rdna",
            },
        },
    }
    resp = handle_mcp_message(msg)
    assert "result" in resp
    assert '"hallucinations_detected": 1' in resp["result"]["content"][0]["text"]
    assert '"is_valid": false' in resp["result"]["content"][0]["text"]


def test_mcp_server_disk_and_index_fallbacks(mocker: Any, tmp_path: Any) -> None:
    """Test get_framework_snapshot and get_api_signature edge cases and fallbacks.

    Args:
        mocker: Pytest mocker fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    # 1. Candidate dir with file that does not start with clean_fw (branch 58->68)
    fake_snapshots_dir = tmp_path / "snapshots"
    fake_snapshots_dir.mkdir()
    (fake_snapshots_dir / "unrelated_file.json").write_text("{}", encoding="utf-8")
    (fake_snapshots_dir / "not_json.txt").write_text("text", encoding="utf-8")

    mocker.patch(
        "ml_framework_snapshots.index.get_cache_dir",
        return_value=str(tmp_path),
    )
    # Clear cache for custom_fw
    from ml_framework_snapshots.mcp_server import _SNAPSHOT_CACHE

    _SNAPSHOT_CACHE.pop("custom_fw_none", None)
    snap = get_framework_snapshot("custom_fw", version=None)
    assert snap == {"categories": {}}

    # When version IS provided but file in dir does not match (branch 58->68)
    _SNAPSHOT_CACHE.pop("custom_fw_99.0.0", None)
    snap_v = get_framework_snapshot("custom_fw", version="99.0.0")
    assert snap_v == {"categories": {}}

    # 2. get_api_signature when lookup_symbol raises exception (lines 115-116)
    mocker.patch(
        "ml_framework_snapshots.index.lookup_symbol",
        side_effect=RuntimeError("Index lookup error"),
    )
    mock_data = {
        "categories": {
            "all": [{"api_path": "custom_fw.fn", "name": "fn", "kind": "function"}]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_data,
    )
    sig = get_api_signature("custom_fw", "custom_fw.fn")
    assert sig is not None
    assert sig["api_path"] == "custom_fw.fn"


def test_search_apis_edge_cases(mocker: Any) -> None:
    """Test search_apis branch coverage for aliases, FTS5 results, and snapshot scanning.

    Args:
        mocker: Pytest mocker fixture.
    """
    # 1. CONCEPT_ALIAS_MAP has key but no matching entries for framework (branch 428->432)
    mocker.patch.dict(
        CONCEPT_ALIAS_MAP,
        {"some_query": {"other_fw": ["other.op"]}},
        clear=False,
    )
    # 2. search_index raises an Exception (lines 449-450)
    mocker.patch(
        "ml_framework_snapshots.index.search_index",
        side_effect=RuntimeError("search failed"),
    )
    mock_snap = {
        "categories": {
            "all": [
                {"api_path": "torch.some_query_op", "name": "some_query_op"},
                {
                    "api_path": "torch.some_query_op",
                    "name": "some_query_op",
                },  # duplicate
                {
                    "api_path": "",
                    "name": "",
                    "mnemonic": "",
                },  # empty path covers 464->466
                {"api_path": "torch.other_op", "name": "other_op"},
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )
    # limit=10 ensures len(matches) < limit (branch 467->469)
    res = search_apis("torch", "some_query", limit=10)
    assert "torch.some_query_op" in res

    # 3. search_index returns results with duplicate or empty path (branches 445->439, 447->452)
    mocker.patch(
        "ml_framework_snapshots.index.search_index",
        return_value=[{"api_path": None, "name": None, "mnemonic": None}],
    )
    res2 = search_apis("torch", "some_query", limit=10)
    assert "torch.some_query_op" in res2


def test_check_hallucination_dtypes_and_ranks_edge_cases(mocker: Any) -> None:
    """Test check_hallucination handling dict dtypes/ranks, length mismatches, and rank comparisons.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "all": [
                {
                    "api_path": "torch.test_op",
                    "name": "test_op",
                    "params": [
                        {"name": "x", "dtypes": ["float32"], "rank": 2},
                        {"name": "y", "dtypes": ["float32"], "rank": ">=2"},
                        {"name": "z", "dtypes": None, "rank": "unknown_rank_format"},
                        {"name": "unconstrained", "dtypes": None, "rank": None},
                    ],
                },
                {
                    "api_path": "torch.linalg.inv",
                    "name": "inv",
                    "params": [
                        {"name": "a", "dtypes": None, "rank": ">=2"},
                    ],
                },
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )
    mocker.patch(
        "ml_framework_snapshots.index.lookup_symbol",
        return_value=None,
    )

    # 1. Dict arg_dtypes (line 611), list arg_dtypes exceeding params (branch 615->614)
    # kwarg_ranks (line 620), dict arg_ranks (line 622), list arg_ranks exceeding params (branch 626->625)
    # kwarg_values containing dtype keyword when already passed (branch 639->630) and not passed (line 640)
    # exact rank match (branch 686->663), min rank match (branch 688->663)
    # non-int non->= rank (branch 686->663)
    # unconstrained param without allowed dtypes (branch 673->679)
    res_valid = check_hallucination(
        framework="torch",
        api_path="torch.test_op",
        kwargs=["x", "y", "z", "unconstrained"],
        kwarg_dtypes={"unconstrained": "float32", "dtype": "float32"},
        arg_dtypes=[
            "float32",
            "float32",
            "float32",
            "extra_ignored_dtype",
            "fifth_excess_dtype",
        ],
        kwarg_ranks={"z": 2, "unconstrained": 1},
        arg_ranks=[2, 3, 1, 99, 100],
        kwarg_values={"dtype": "float32", "unpassed_kw": "float32"},
    )
    assert not res_valid.get("is_hallucinated")

    # Test check_rdna_instruction with modifiers (lines 979-980)
    from ml_framework_snapshots.mcp_server import check_rdna_instruction

    res_rdna_mods = check_rdna_instruction("v_add_f32", modifiers=["dpp"])
    assert res_rdna_mods is not None

    # 2. Test dict arg_dtypes and dict arg_ranks directly
    res_dict_args = check_hallucination(
        framework="torch",
        api_path="torch.test_op",
        kwargs=["x", "y"],
        arg_dtypes={"x": "float32", "y": "float32"},
        arg_ranks={"x": 2, "y": 2},
    )
    assert not res_dict_args.get("is_hallucinated")

    # 3. is_float_complex_api fallback allowed dtypes (line 672) when parameter 'a' is validated
    res_inv_valid = check_hallucination(
        framework="torch",
        api_path="torch.linalg.inv",
        kwargs=["a"],
        kwarg_dtypes={"a": "float32"},
    )
    assert not res_inv_valid.get("is_hallucinated")

    # 4. is_float_complex_api with unexpected int dtype passed (line 697)
    res_inv_int = check_hallucination(
        framework="torch",
        api_path="torch.linalg.inv",
        kwargs=["a"],
        kwarg_dtypes={"other": "int64"},
    )
    assert res_inv_int.get("is_hallucinated")
    assert "Floating-point or complex dtype required" in res_inv_int.get("reason", "")


def test_check_mlir_op_and_stablehlo_additional_branches(mocker: Any) -> None:
    """Test check_mlir_op and check_stablehlo_op for traits, broadcast_in_dim, and regions.

    Args:
        mocker: Pytest mocker fixture.
    """
    # 1. Existing SameOperandsAndResultType and Elementwise traits (branches 1126->1128, 1128->1143)
    mock_mlir_snap = {
        "categories": {
            "all": [
                {
                    "api_path": "arith.muli",
                    "name": "muli",
                    "operands": [
                        {"name": "lhs", "type": "i32"},
                        {"name": "rhs", "type": "i32"},
                    ],
                    "attributes": [],
                    "traits": ["SameOperandsAndResultType", "Elementwise"],
                },
                {
                    "api_path": "stablehlo.add",
                    "name": "add",
                    "operands": [
                        {"name": "lhs", "type": "tensor<4xf32>"},
                        {"name": "rhs", "type": "tensor<4xf32>"},
                    ],
                    "attributes": [],
                    "traits": [],
                },
                {
                    "api_path": "stablehlo.subtract",
                    "name": "subtract",
                    "operands": [
                        {"name": "lhs", "type": "tensor<4xf32>"},
                        {"name": "rhs", "type": "tensor<4xf32>"},
                    ],
                    "attributes": [],
                    "traits": ["SameOperandsAndResultType", "Elementwise"],
                },
                {
                    "api_path": "stablehlo.broadcast_in_dim",
                    "name": "broadcast_in_dim",
                    "operands": [{"name": "operand", "type": "tensor<4xf32>"}],
                    "attributes": [{"name": "broadcast_dimensions", "type": "array"}],
                    "traits": [],
                },
                {
                    "api_path": "stablehlo.reduce",
                    "name": "reduce",
                    "operands": [
                        {"name": "inputs", "type": "tensor<4xf32>"},
                        {"name": "init_values", "type": "tensor<f32>"},
                    ],
                    "attributes": [{"name": "dimensions", "type": "array"}],
                    "traits": [],
                },
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_mlir_snap,
    )
    res_existing_traits = check_mlir_op(
        op_name="arith.muli",
        operands_count=2,
        operand_types=["i32", "i32"],
        result_types=["i32"],
    )
    assert res_existing_traits["is_valid"]

    # 2. stablehlo.add sets traits (lines 1137-1140)
    res_shlo_add = check_mlir_op(
        op_name="stablehlo.add",
        operands_count=2,
        operand_types=["tensor<4xf32>", "tensor<4xf32>"],
        result_types=["tensor<4xf32>"],
    )
    assert res_shlo_add["is_valid"]

    # stablehlo.subtract already has traits (branches 1137->1139, 1139->1143)
    res_shlo_sub = check_mlir_op(
        op_name="stablehlo.subtract",
        operands_count=2,
        operand_types=["tensor<4xf32>", "tensor<4xf32>"],
        result_types=["tensor<4xf32>"],
    )
    assert res_shlo_sub["is_valid"]

    # 3. broadcast_dimensions validation (lines 1294-1298) and non-list / missing shape fallbacks (1294->1200, 1297->1200)
    res_bcast = check_mlir_op(
        op_name="stablehlo.broadcast_in_dim",
        structured_attributes={
            "broadcast_dimensions": [0],
            "operand_shape": [4],
            "result_shape": [4, 8],
        },
    )
    assert res_bcast["is_valid"]

    # Non-list broadcast_dimensions (branch 1294->1200)
    res_bcast_str = check_mlir_op(
        op_name="stablehlo.broadcast_in_dim",
        structured_attributes={"broadcast_dimensions": "not_a_list"},
    )
    assert res_bcast_str["is_valid"]

    # Missing operand_shape (branch 1297->1200)
    res_bcast_noshape = check_mlir_op(
        op_name="stablehlo.broadcast_in_dim",
        structured_attributes={"broadcast_dimensions": [0], "operand_shape": None},
    )
    assert res_bcast_noshape["is_valid"]

    # 4. region validation (lines 1304-1312) and non-dict region spec fallback (branch 1305->1304)
    res_region = check_mlir_op(
        op_name="stablehlo.reduce",
        regions={
            "body": {
                "block_arguments": ["tensor<f32>", "tensor<f32>"],
                "yield_types": ["tensor<f32>"],
            },
            "invalid_spec": "not_a_dict",  # branch 1305->1304
        },
    )
    assert res_region["is_valid"]


def test_validate_source_code_and_jsonrpc_additional_branches(mocker: Any) -> None:
    """Test check_code_block framework inference, AST edge cases, and JSON-RPC tool calls.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch(
        "ml_framework_snapshots.mcp_server.check_hallucination",
        return_value={"is_hallucinated": False},
    )
    mocker.patch(
        "ml_framework_snapshots.mcp_server.check_mlir_op",
        return_value={"is_valid": True},
    )
    mocker.patch(
        "ml_framework_snapshots.mcp_server.check_sass_instruction",
        return_value={"is_valid": True},
    )
    mocker.patch(
        "ml_framework_snapshots.mcp_server.check_rdna_instruction",
        return_value={"is_valid": True},
    )
    mocker.patch(
        "ml_framework_snapshots.mcp_server.check_stablehlo_op",
        return_value={"is_valid": True},
    )

    # 1. Framework auto-inference in check_code_block (lines 1424-1438)
    # and AST edge cases: non-Name call target (1402->1404), call without parts (1404->1414),
    # non-constant kwarg and **kwargs (1409->1408), unknown framework (1438)
    py_code = """
x = 1 + 2
val = 10
extra_kwargs = {}
torch.abs(x, alpha=1, **extra_kwargs)
jax.numpy.abs(x)
tf.math.abs(x)
np.abs(x)
nn.Linear(10, 20)
F.relu(x)
stablehlo.constant()
custom_unknown_module.some_fn(x)
(lambda: None)()
"""
    res_py = check_code_block(py_code, framework=None)
    assert res_py["total_analyzed"] >= 6

    # Explicit framework (branch 1423->1440)
    res_explicit_fw = check_code_block("torch.abs(x)", framework="torch")
    assert res_explicit_fw["total_analyzed"] == 1

    # Unrecognized library call alone (line 1438)
    res_unrec = check_code_block("unrecognized_lib.call()", framework=None)
    assert res_unrec["total_analyzed"] == 0

    # 2. Code block with no function calls (branch 1419->1463)
    res_no_calls = check_code_block("x = 10\ny = 20\n", framework=None)
    assert res_no_calls["total_analyzed"] == 0

    # 3. Line-by-line validation covering valid branches and non-matching lines:
    # 1497->1506 (valid MLIR op), 1522->1531 (valid SASS), 1535->1465 (skipped line/label), 1541->1550 (valid RDNA)
    mixed_code = """
// Just a comment line
my_label:
%0 = arith.muli %a, %b : i32
FADD R0, R1, R2;
v_add_f32 v0, v1, v2
"""
    res_mixed = check_code_block(mixed_code, framework="mixed")
    assert res_mixed["total_analyzed"] >= 3
    assert res_mixed["is_valid"]

    # 4. JSON-RPC check_stablehlo_op dispatch (lines 1964-1973)
    req = {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {
            "name": "check_stablehlo_op",
            "arguments": {
                "op_name": "stablehlo.add",
                "operands_count": 2,
                "operand_types": ["tensor<2xf32>", "tensor<2xf32>"],
                "result_types": ["tensor<2xf32>"],
            },
        },
    }
    resp = handle_mcp_message(req)
    assert resp["id"] == 42
    assert '"is_valid": true' in resp["result"]["content"][0]["text"]


def test_shape_and_dtype_grounding(mocker: Any) -> None:
    """Test shape broadcasting, matmul contracting dimensions, and specialized dtype validations.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "math": [
                {
                    "name": "add",
                    "api_path": "torch.add",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
                {
                    "name": "mm",
                    "api_path": "torch.mm",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD", "rank": 2},
                        {"name": "mat2", "kind": "POSITIONAL_OR_KEYWORD", "rank": 2},
                    ],
                },
                {
                    "name": "matmul",
                    "api_path": "torch.matmul",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
                {
                    "name": "bitwise_and",
                    "api_path": "torch.bitwise_and",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
                {
                    "name": "quantize_per_tensor",
                    "api_path": "torch.quantize_per_tensor",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "scale", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "zero_point", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "dtype", "kind": "KEYWORD_ONLY"},
                    ],
                },
            ]
        }
    }
    from ml_framework_snapshots.index import lookup_symbol

    orig_get_framework_snapshot = get_framework_snapshot
    orig_lookup_symbol = lookup_symbol

    def mock_get_snap(framework: str, version: Any = None) -> Dict[str, Any]:
        """Return mock snapshot for torch, otherwise delegate to original getter.

        Args:
            framework: Target framework identifier.
            version: Optional version string.

        Returns:
            Framework snapshot dictionary.
        """
        if framework == "torch":
            return mock_snap
        return orig_get_framework_snapshot(framework, version)

    def mock_lookup(
        framework: str, api_path: str, version: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Mock lookup_symbol returning None for torch.

        Args:
            framework: Target framework identifier.
            api_path: Target API path.
            version: Optional version string.

        Returns:
            Optional symbol dictionary.
        """
        if framework == "torch":
            return None
        return orig_lookup_symbol(framework, api_path, version=version)

    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        side_effect=mock_get_snap,
    )
    mocker.patch(
        "ml_framework_snapshots.index.lookup_symbol",
        side_effect=mock_lookup,
    )

    # 1. Elementwise broadcasting
    res_add_valid = check_hallucination(
        "torch", "torch.add", arg_shapes=[[2, 3], [1, 3]]
    )
    assert res_add_valid["is_hallucinated"] is False

    res_add_invalid = check_hallucination(
        "torch", "torch.add", arg_shapes=[[2, 3], [3, 2]]
    )
    assert res_add_invalid["is_hallucinated"] is True
    assert "Broadcasting error" in res_add_invalid["reason"]

    # 2. Matmul strict 2D and dimension mismatch
    res_mm_valid = check_hallucination("torch", "torch.mm", arg_shapes=[[2, 3], [3, 4]])
    assert res_mm_valid["is_hallucinated"] is False

    res_mm_3d = check_hallucination(
        "torch", "torch.mm", arg_shapes=[[2, 3, 4], [2, 4, 5]]
    )
    assert res_mm_3d["is_hallucinated"] is True
    assert "Strict 2D" in res_mm_3d["reason"]

    res_matmul_k_err = check_hallucination(
        "torch", "torch.matmul", arg_shapes=[[2, 3], [4, 5]]
    )
    assert res_matmul_k_err["is_hallucinated"] is True
    assert "contracting dimension mismatch" in res_matmul_k_err["reason"]

    # 3. Logical/bitwise float dtype rejection
    res_bitwise = check_hallucination(
        "torch", "torch.bitwise_and", arg_dtypes=["float32", "float32"]
    )
    assert res_bitwise["is_hallucinated"] is True
    assert "require integer or boolean" in res_bitwise["reason"]

    # 4. Quantization low-precision constraint
    res_quant = check_hallucination(
        "torch", "torch.quantize_per_tensor", kwarg_values={"dtype": "float32"}
    )
    assert res_quant["is_hallucinated"] is True
    assert "not a valid quantization dtype" in res_quant["reason"]


def test_stablehlo_rank_and_conv_validation() -> None:
    """Test check_mlir_op with inferred operand ranks for DotDimensionNumbersAttr and ConvDimensionNumbersAttr."""
    # DotDimensionNumbersAttr with contracting dim exceeding rank
    dot_attrs = {
        "dot_dimension_numbers": {
            "lhs_batch_dimensions": [],
            "rhs_batch_dimensions": [],
            "lhs_contracting_dimensions": [5],
            "rhs_contracting_dimensions": [0],
        }
    }
    res_dot = check_mlir_op(
        "stablehlo.dot_general",
        operand_types=["tensor<2x3xf32>", "tensor<3x4xf32>"],
        structured_attributes=dot_attrs,
    )
    assert res_dot["is_valid"] is False
    assert any("is out of bounds for rank 2" in err for err in res_dot["errors"])

    # ConvDimensionNumbersAttr with spatial dimension mismatch
    conv_attrs = {
        "dimension_numbers": {
            "input_batch_dimension": 0,
            "input_feature_dimension": 1,
            "input_spatial_dimensions": [2, 3],
            "kernel_input_feature_dimension": 1,
            "kernel_output_feature_dimension": 0,
            "kernel_spatial_dimensions": [2, 3, 4],
            "output_batch_dimension": 0,
            "output_feature_dimension": 1,
            "output_spatial_dimensions": [2, 3],
        }
    }
    res_conv = check_mlir_op(
        "stablehlo.convolution",
        operand_types=["tensor<1x3x224x224xf32>", "tensor<64x3x7x7x7xf32>"],
        structured_attributes=conv_attrs,
    )
    assert res_conv["is_valid"] is False
    assert any("Spatial dimensions count mismatch" in err for err in res_conv["errors"])


def test_explain_anti_pattern_tool() -> None:
    """Test explain_anti_pattern tool with known cross-framework patterns and unknown fallbacks."""
    from ml_framework_snapshots.mcp_server import explain_anti_pattern

    # 1. PyTorch axis -> dim
    res_axis = explain_anti_pattern("torch", "torch.sum", "axis")
    assert res_axis["is_known_anti_pattern"] is True
    assert res_axis["canonical_argument"] == "dim"
    assert "dim" in res_axis["explanation"]

    # 2. PyTorch keepdims -> keepdim
    res_keepdims = explain_anti_pattern("torch", "torch.sum", "keepdims")
    assert res_keepdims["is_known_anti_pattern"] is True
    assert res_keepdims["canonical_argument"] == "keepdim"

    # 3. JAX dim -> axis
    res_jax = explain_anti_pattern("jax", "jax.numpy.sum", "dim")
    assert res_jax["is_known_anti_pattern"] is True
    assert res_jax["canonical_argument"] == "axis"

    # 4. TensorFlow keepdim -> keepdims
    res_tf = explain_anti_pattern("tensorflow", "tf.reduce_sum", "keepdim")
    assert res_tf["is_known_anti_pattern"] is True
    assert res_tf["canonical_argument"] == "keepdims"

    # PyTorch inplace migration
    res_inp = explain_anti_pattern("torch", "torch.relu", "inplace")
    assert res_inp["is_known_anti_pattern"] is True
    assert "underscore" in res_inp["explanation"]

    # JAX PRNG key and rng migration
    res_key = explain_anti_pattern("jax", "jax.random.normal", "key")
    assert res_key["is_known_anti_pattern"] is True
    assert "PRNGKey" in res_key["explanation"]

    res_rng = explain_anti_pattern("jax", "jax.random.normal", "rng")
    assert res_rng["is_known_anti_pattern"] is True
    assert res_rng["canonical_argument"] == "key"

    # 5. Unknown argument fallback
    res_unknown = explain_anti_pattern("torch", "torch.sum", "unrecognized_kwarg")
    assert res_unknown["is_known_anti_pattern"] is False
    assert res_unknown["canonical_argument"] is None
    assert "not recognized" in res_unknown["explanation"]

    # 6. MCP message dispatch
    msg = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "explain_anti_pattern",
            "arguments": {
                "framework": "torch",
                "api_path": "torch.sum",
                "hallucinated_argument": "axis",
            },
        },
    }
    resp = handle_mcp_message(msg)
    assert resp["id"] == 101
    assert "result" in resp
    assert '"canonical_argument": "dim"' in resp["result"]["content"][0]["text"]


def test_mcp_edge_coverage_branches(mocker: Any) -> None:
    """Test MCP server edge branches for hallucination checks, anti-patterns, and MLIR ops.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "math": [
                {
                    "name": "add",
                    "api_path": "torch.add",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
                {
                    "name": "bitwise_and",
                    "api_path": "torch.bitwise_and",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
                {
                    "name": "quantize_per_tensor",
                    "api_path": "torch.quantize_per_tensor",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "scale", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "zero_point", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "dtype", "kind": "KEYWORD_ONLY"},
                    ],
                },
            ]
        }
    }
    from ml_framework_snapshots.index import lookup_symbol

    orig_get_framework_snapshot = get_framework_snapshot
    orig_lookup_symbol = lookup_symbol

    def mock_get_snap(framework: str, version: Any = None) -> Dict[str, Any]:
        """Return mock snapshot for torch, otherwise delegate to original getter.

        Args:
            framework: Target framework identifier.
            version: Optional version string.

        Returns:
            Framework snapshot dictionary.
        """
        if framework == "torch":
            return mock_snap
        return orig_get_framework_snapshot(framework, version)

    def mock_lookup(
        framework: str, api_path: str, version: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Mock lookup_symbol returning None for torch.

        Args:
            framework: Target framework identifier.
            api_path: Target API path.
            version: Optional version string.

        Returns:
            Optional symbol dictionary.
        """
        if framework == "torch":
            return None
        return orig_lookup_symbol(framework, api_path, version=version)

    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        side_effect=mock_get_snap,
    )
    mocker.patch(
        "ml_framework_snapshots.index.lookup_symbol",
        side_effect=mock_lookup,
    )
    from ml_framework_snapshots.mcp_server import explain_anti_pattern

    # 1. explain_anti_pattern with unknown framework and API (sig is None)
    res_none_sig = explain_anti_pattern("nonexistent_fw", "nonexistent.op", "bad_arg")
    assert res_none_sig["is_known_anti_pattern"] is False
    assert res_none_sig["canonical_params"] == []

    # 2. check_hallucination: logical/bitwise op with non-float dtype followed by float dtype (714->713)
    res_dup_dtype = check_hallucination(
        framework="torch",
        api_path="torch.bitwise_and",
        kwarg_dtypes={"input": "int32", "other": "float32"},
    )
    assert res_dup_dtype["is_hallucinated"] is True
    assert "Dtype 'float32' is not supported" in res_dup_dtype["reason"]

    # 3. check_hallucination: quantization op without "dtype" and with valid "dtype" (735->741)
    res_quant_no_dt = check_hallucination(
        framework="torch",
        api_path="torch.quantize_per_tensor",
        kwarg_dtypes={"scale": "float32"},
    )
    assert res_quant_no_dt["api_exists"] is True
    res_quant_valid_dt = check_hallucination(
        framework="torch",
        api_path="torch.quantize_per_tensor",
        kwarg_dtypes={"dtype": "torch.qint8"},
    )
    assert res_quant_valid_dt["is_hallucinated"] is False

    # 4. check_hallucination: kwarg_shapes and extra arg_shapes
    res_shapes = check_hallucination(
        framework="torch",
        api_path="torch.add",
        kwarg_shapes={"input": [2, 2]},
        arg_shapes=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
    )
    assert res_shapes["is_hallucinated"] is False

    # 5. check_mlir_op: dot with non-tensor operand types
    res_dot_nontensor = check_mlir_op(
        op_name="stablehlo.dot_general",
        structured_attributes={
            "dot_dimension_numbers": {
                "lhs_batch_dimensions": [0],
                "rhs_batch_dimensions": [0],
                "lhs_contracting_dimensions": [1],
                "rhs_contracting_dimensions": [1],
            }
        },
        operand_types=["i32", "i32"],
    )
    assert res_dot_nontensor["is_valid"] is True

    # 6. check_mlir_op: conv with non-tensor operand types and valid result_types
    conv_attr = {
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
    res_conv_tensor_res = check_mlir_op(
        op_name="stablehlo.convolution",
        structured_attributes={"dimension_numbers": conv_attr},
        operand_types=["i32", "i32"],
        result_types=["tensor<1x16x32x32xf32>"],
    )
    assert res_conv_tensor_res["is_valid"] is True

    # 7. check_mlir_op: conv with non-tensor result_types
    res_conv_nontensor_res = check_mlir_op(
        op_name="stablehlo.convolution",
        structured_attributes={"dimension_numbers": conv_attr},
        operand_types=["tensor<1x16x32x32xf32>", "tensor<16x16x3x3xf32>"],
        result_types=["i32"],
    )
    assert res_conv_nontensor_res["is_valid"] is True


def test_get_framework_snapshot_extraction_exception(mocker: Any) -> None:
    """Test get_framework_snapshot fallback to empty dict when runtime extraction raises exception.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.mcp_server import (
        _SNAPSHOT_CACHE,
        get_framework_snapshot,
    )

    _SNAPSHOT_CACHE.clear()
    mock_extract = mocker.MagicMock(side_effect=RuntimeError("extraction failure"))
    mocker.patch("ml_framework_snapshots.mcp_server.extract_snapshot", mock_extract)
    mocker.patch(
        "ml_framework_snapshots.mcp_server.is_offline_mode", return_value=False
    )

    res = get_framework_snapshot("torch")
    assert res == {"categories": {}}


def test_check_sass_fp4_microscopic_scaling_architecture_gating() -> None:
    """Test check_sass_instruction architecture gating for FP4/FP6/microscopic scaling mnemonics."""
    from ml_framework_snapshots.mcp_server import check_sass_instruction

    # sm_90 target architecture should fail for FP4 instruction
    res_sm90 = check_sass_instruction("MMA_SCALE_FP4", sm_arch="sm_90")
    assert res_sm90["is_valid"] is False
    assert any(
        "Microscopic scaling and FP4/FP6 instructions" in err
        for err in res_sm90["errors"]
    )

    # sm_100 target architecture should succeed
    res_sm100 = check_sass_instruction("MMA_SCALE_FP4", sm_arch="sm_100")
    assert res_sm100["is_valid"] is True


def test_translate_concept_arguments_direct_schema_key() -> None:
    """Test translate_concept_arguments when concept name directly matches a parameter translation schema key."""
    from ml_framework_snapshots.mcp_server import translate_concept_arguments

    res = translate_concept_arguments("matmul", "torch", "numpy", {"dim": 0})
    assert res["concept"] == "matmul"
    assert res["source_framework"] == "torch"
    assert res["target_framework"] == "numpy"

    res_dot = translate_concept_arguments("dot", "torch", "numpy", {"dim": 0})
    assert res_dot["concept"] == "dot"
    assert res_dot["source_framework"] == "torch"
    assert res_dot["target_framework"] == "numpy"
