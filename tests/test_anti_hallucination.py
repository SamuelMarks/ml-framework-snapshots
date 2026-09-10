"""Anti-Hallucination Grounding Test Suite.

Verifies that compliance checking, snapshot diffing, and MCP tools correctly
flag 100% of synthetic LLM hallucinations:
    1. Argument name discrepancies (dim vs axis, keepdim vs keepdims)
    2. Non-existent hallucinated kwargs (e.g. cross_entropy(..., axis=...))
    3. Deprecated/discarded parameters across versions
    4. Invalid SASS operands and nonexistent hardware instructions
    5. Invalid StableHLO ops and illegal operand counts
    6. LLM prompt context export and .pyi stub compilation validation
    7. Model Context Protocol (MCP) JSON-RPC tool server handlers
"""

import io
import json
import os
from typing import Any, List, Tuple
from unittest.mock import patch
import pytest

from ml_framework_snapshots.compliance import score_compliance
from ml_framework_snapshots.diff import diff_snapshots
from ml_framework_snapshots.export import export_llm_prompt_context
from ml_framework_snapshots.mcp_server import (
    get_api_signature,
    search_apis,
    check_hallucination,
    check_sass_instruction,
    check_rdna_instruction,
    check_mlir_op,
    handle_mcp_message,
    run_mcp_server,
)
from ml_framework_snapshots.stubs import validate_pyi_stub
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind


def test_hallucination_argument_name_discrepancy() -> None:
    """Verify that compliance checking flags argument name discrepancies (dim vs axis)."""
    ref_snapshot = {
        "categories": {
            "math": [
                {
                    "name": "sum",
                    "api_path": "torch.sum",
                    "kind": "function",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "keepdim", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                }
            ]
        }
    }

    # Hallucinated target using NumPy-style axis and keepdims instead of torch's dim and keepdim
    hallucinated_target = [
        GhostRef(
            name="sum",
            api_path="torch.sum",
            kind="function",
            params=[
                GhostParam(name="input", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
                GhostParam(name="axis", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
                GhostParam(name="keepdims", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            ],
        )
    ]

    report = score_compliance(ref_snapshot, hallucinated_target)
    mismatched_paths = [m["api_path"] for m in report.get("mismatched", [])]
    assert "torch.sum" in mismatched_paths
    assert report.get("score_percentage", 100.0) == 0.0


def test_hallucination_nonexistent_kwargs() -> None:
    """Verify that compliance and MCP tool flag non-existent kwargs hallucinated by LLMs."""
    ref_snapshot = {
        "categories": {
            "loss": [
                {
                    "name": "cross_entropy",
                    "api_path": "torch.nn.functional.cross_entropy",
                    "kind": "function",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "target", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "weight", "kind": "KEYWORD_ONLY", "default": "None"},
                    ],
                }
            ]
        }
    }

    hallucinated_cross_entropy = [
        GhostRef(
            name="cross_entropy",
            api_path="torch.nn.functional.cross_entropy",
            kind="function",
            params=[
                GhostParam(name="input", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
                GhostParam(name="target", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
                # Hallucinated 'axis' kwarg that doesn't exist in PyTorch's cross_entropy
                GhostParam(name="axis", kind=ParameterKind.KEYWORD_ONLY),
            ],
        )
    ]

    report = score_compliance(ref_snapshot, hallucinated_cross_entropy)
    mismatched_paths = [m["api_path"] for m in report.get("mismatched", [])]
    assert "torch.nn.functional.cross_entropy" in mismatched_paths

    # Test MCP check_hallucination helper
    mcp_check = check_hallucination(
        framework="mock_torch",
        api_path="torch.nn.functional.cross_entropy",
        kwargs=["input", "target", "axis"],
    )
    # When snapshot is not cached or api not found
    assert mcp_check["is_hallucinated"] is True


def test_hallucination_deprecated_parameter_diff() -> None:
    """Verify that diff_snapshots flags removed/deprecated parameters as breaking changes."""
    old_snapshot = {
        "categories": {
            "ops": [
                {
                    "name": "legacy_op",
                    "api_path": "framework.legacy_op",
                    "kind": "function",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
                        {
                            "name": "deprecated_flag",
                            "kind": "KEYWORD_ONLY",
                            "default": "True",
                        },
                    ],
                }
            ]
        }
    }

    new_snapshot = {
        "categories": {
            "ops": [
                {
                    "name": "legacy_op",
                    "api_path": "framework.legacy_op",
                    "kind": "function",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                }
            ]
        }
    }

    result = diff_snapshots(old_snapshot, new_snapshot)
    assert "framework.legacy_op" in result.signature_changed
    assert "framework.legacy_op" in result.breaking_signature_changed


def test_hallucination_invalid_sass_instructions() -> None:
    """Verify that invalid SASS operands and nonexistent instructions are detected."""
    sass_snapshot = {
        "categories": {
            "isa": [
                {
                    "name": "FADD",
                    "api_path": "nvidia_sass.inst.FADD",
                    "kind": "function",
                    "params": [
                        {"name": "op0", "annotation": "R", "kind": "POSITIONAL_ONLY"},
                        {"name": "op1", "annotation": "R", "kind": "POSITIONAL_ONLY"},
                        {"name": "op2", "annotation": "R", "kind": "POSITIONAL_ONLY"},
                    ],
                }
            ]
        }
    }

    # Nonexistent SASS instruction hallucinated by LLM (e.g. FADD_QUAD)
    hallucinated_insts = [
        GhostRef(
            name="FADD_QUAD",
            api_path="nvidia_sass.inst.FADD_QUAD",
            kind="function",
            params=[
                GhostParam(
                    name="op0", annotation="R", kind=ParameterKind.POSITIONAL_ONLY
                )
            ],
        ),
        # Real instruction but with hallucinated 5 operands (illegal for FADD)
        GhostRef(
            name="FADD",
            api_path="nvidia_sass.inst.FADD",
            kind="function",
            params=[
                GhostParam(
                    name=f"op{i}", annotation="R", kind=ParameterKind.POSITIONAL_ONLY
                )
                for i in range(5)
            ],
        ),
    ]

    report = score_compliance(sass_snapshot, hallucinated_insts)
    mismatched_paths = [m["api_path"] for m in report.get("mismatched", [])]
    assert "nvidia_sass.inst.FADD" in mismatched_paths

    # Verify diff_snapshots flags the added hallucinated instruction and changed signature
    hallucinated_snap = {
        "categories": {
            "isa": [
                {
                    "name": "FADD_QUAD",
                    "api_path": "nvidia_sass.inst.FADD_QUAD",
                    "kind": "function",
                    "params": [],
                },
                {
                    "name": "FADD",
                    "api_path": "nvidia_sass.inst.FADD",
                    "kind": "function",
                    "params": [
                        {"name": f"op{i}", "kind": "POSITIONAL_ONLY"} for i in range(5)
                    ],
                },
            ]
        }
    }
    diff_res = diff_snapshots(sass_snapshot, hallucinated_snap)
    assert "nvidia_sass.inst.FADD_QUAD" in diff_res.added
    assert "nvidia_sass.inst.FADD" in diff_res.signature_changed


def test_hallucination_invalid_stablehlo_ops() -> None:
    """Verify that illegal StableHLO ops and wrong operand counts are detected."""
    stablehlo_snapshot = {
        "categories": {
            "dialect": [
                {
                    "name": "add",
                    "api_path": "stablehlo.add",
                    "kind": "function",
                    "params": [
                        {
                            "name": "lhs",
                            "annotation": "tensor",
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                        {
                            "name": "rhs",
                            "annotation": "tensor",
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                    ],
                }
            ]
        }
    }

    hallucinated_ops = [
        # Hallucinated op: stablehlo.fused_attention (does not exist in stablehlo spec)
        GhostRef(
            name="fused_attention",
            api_path="stablehlo.fused_attention",
            kind="function",
            params=[GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
        ),
        # add with 3 operands (illegal for binary op)
        GhostRef(
            name="add",
            api_path="stablehlo.add",
            kind="function",
            params=[
                GhostParam(name="lhs", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
                GhostParam(name="rhs", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
                GhostParam(name="extra", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            ],
        ),
    ]

    report = score_compliance(stablehlo_snapshot, hallucinated_ops)
    mismatched_paths = [m["api_path"] for m in report.get("mismatched", [])]
    assert "stablehlo.add" in mismatched_paths

    # Verify diff_snapshots flags the hallucinated op and illegal operand count
    hallucinated_snap = {
        "categories": {
            "dialect": [
                {
                    "name": "fused_attention",
                    "api_path": "stablehlo.fused_attention",
                    "kind": "function",
                    "params": [],
                },
                {
                    "name": "add",
                    "api_path": "stablehlo.add",
                    "kind": "function",
                    "params": [
                        {"name": "lhs", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "rhs", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "extra", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
            ]
        }
    }
    diff_res = diff_snapshots(stablehlo_snapshot, hallucinated_snap)
    assert "stablehlo.fused_attention" in diff_res.added
    assert "stablehlo.add" in diff_res.signature_changed


def test_export_llm_prompt_context() -> None:
    """Test exporting GhostRef list to compact LLM prompt context."""
    ref = GhostRef(
        name="relu",
        api_path="torch.nn.functional.relu",
        kind="function",
        params=[
            GhostParam(
                name="input",
                annotation="Tensor",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                description="Input tensor",
            ),
            GhostParam(
                name="inplace",
                annotation="bool",
                default="False",
                kind=ParameterKind.KEYWORD_ONLY,
                description="Inplace operation flag",
            ),
        ],
        returns_type="Tensor",
        raises=["ValueError"],
        environment_tags=["cpu", "cuda"],
        docstring="Applies the rectified linear unit function element-wise.\nMore detailed explanation here.",
    )

    context = export_llm_prompt_context([ref])
    assert "### `torch.nn.functional.relu`" in context
    assert "relu(input: Tensor, inplace: bool = False) -> Tensor" in context
    assert "- **Raises**: ValueError" in context
    assert "- **Environments**: cpu, cuda" in context
    assert "- **Summary**: Applies the rectified linear unit function" in context


def test_validate_pyi_stub() -> None:
    """Test validating syntactically valid and invalid .pyi stub strings."""
    valid_stub = """from typing import Any, Optional

def relu(input: Any, inplace: bool = False) -> Any: ...

class Linear:
    def __init__(self, in_features: int, out_features: int) -> None: ...
"""
    assert validate_pyi_stub(valid_stub) is True

    invalid_stub = """def bad_syntax(::: invalid: ..."""
    with pytest.raises(SyntaxError):
        validate_pyi_stub(invalid_stub)


def test_mcp_server_protocol(mocker: Any) -> None:
    """Test Model Context Protocol (MCP) tool list and tool call request handling.

    Args:
        mocker: Pytest mocker fixture.
    """
    # Test tools/list
    list_req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    list_resp = handle_mcp_message(list_req)
    assert list_resp["id"] == 1
    tools = list_resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "get_api_signature" in tool_names
    assert "search_apis" in tool_names
    assert "check_hallucination" in tool_names

    # Mock snapshot cache
    mock_snap = {
        "categories": {
            "math": [
                {
                    "name": "add",
                    "api_path": "torch.add",
                    "aliases": ["torch.Tensor.add"],
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                }
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
    mocker.patch(
        "ml_framework_snapshots.index.search_index",
        return_value=[],
    )

    # Test get_api_signature
    sig = get_api_signature("torch", "torch.add")
    assert sig is not None
    assert sig["name"] == "add"

    # Alias lookup
    sig_alias = get_api_signature("torch", "torch.Tensor.add")
    assert sig_alias is not None
    assert sig_alias["name"] == "add"

    # Non-existent
    assert get_api_signature("torch", "torch.nonexistent") is None

    # Test search_apis
    search_res = search_apis("torch", "add")
    assert "torch.add" in search_res

    # Test check_hallucination on valid call
    valid_check = check_hallucination("torch", "torch.add", kwargs=["input", "other"])
    assert valid_check["is_hallucinated"] is False

    # Test check_hallucination on hallucinated kwarg
    invalid_check = check_hallucination("torch", "torch.add", kwargs=["input", "axis"])
    assert invalid_check["is_hallucinated"] is True
    assert "axis" in invalid_check["invalid_kwargs"]

    # Test tools/call JSON-RPC messages
    call_sig_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_api_signature",
            "arguments": {"framework": "torch", "api_path": "torch.add"},
        },
    }
    call_sig_resp = handle_mcp_message(call_sig_req)
    assert call_sig_resp["id"] == 2
    assert "torch.add" in call_sig_resp["result"]["content"][0]["text"]

    call_search_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "search_apis",
            "arguments": {"framework": "torch", "query": "add"},
        },
    }
    call_search_resp = handle_mcp_message(call_search_req)
    assert call_search_resp["id"] == 3
    assert "torch.add" in call_search_resp["result"]["content"][0]["text"]

    call_check_req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "check_hallucination",
            "arguments": {
                "framework": "torch",
                "api_path": "torch.add",
                "kwargs": ["axis"],
            },
        },
    }
    call_check_resp = handle_mcp_message(call_check_req)
    assert call_check_resp["id"] == 4
    assert "axis" in call_check_resp["result"]["content"][0]["text"]

    # Test unknown tool
    unknown_tool_req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "nonexistent_tool"},
    }
    unknown_tool_resp = handle_mcp_message(unknown_tool_req)
    assert "error" in unknown_tool_resp

    # Test unknown method
    unknown_method_req = {"jsonrpc": "2.0", "id": 6, "method": "unknown/method"}
    unknown_method_resp = handle_mcp_message(unknown_method_req)
    assert "error" in unknown_method_resp


def test_run_mcp_server_stdio() -> None:
    """Test run_mcp_server reading from string input stream and writing responses."""
    in_data = json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/list"}) + "\n\n"
    in_stream = io.StringIO(in_data)
    out_stream = io.StringIO()

    run_mcp_server(input_stream=in_stream, output_stream=out_stream)
    output = out_stream.getvalue().strip()
    assert output
    resp = json.loads(output)
    assert resp["id"] == 10
    assert "tools" in resp["result"]


def test_mcp_server_branches(mocker: Any) -> None:
    """Test get_framework_snapshot uncached extractor, search limit, and varkwargs in check_hallucination.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch(
        "ml_framework_snapshots.mcp_server.extract_snapshot",
        return_value={
            "categories": {
                "math": [
                    {
                        "name": "op1",
                        "api_path": "test.op1",
                        "params": [
                            {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
                            {"name": "kwargs", "kind": "VAR_KEYWORD"},
                        ],
                    },
                    {
                        "name": "op2",
                        "api_path": "test.op2",
                        "params": [],
                    },
                ]
            }
        },
    )

    from ml_framework_snapshots.mcp_server import (
        _SNAPSHOT_CACHE,
        get_framework_snapshot,
    )

    _SNAPSHOT_CACHE.clear()

    snap = get_framework_snapshot("torch")
    assert "math" in snap["categories"]

    # Search with limit reached
    matches = search_apis("torch", "op", limit=1)
    assert len(matches) == 1

    # Fuzzy search fallback
    fuzzy_matches = search_apis("torch", "test.op", limit=2)
    assert len(fuzzy_matches) >= 1

    # Search with no matches (completely distant query)
    assert search_apis("torch", "zzzzzz_totally_unknown_12345") == []

    # Check hallucination with VAR_KEYWORD: strict_kwargs=True by default flags unconstrained kwargs
    var_check_strict = check_hallucination(
        "torch", "test.op1", kwargs=["any_arg", "another_arg"]
    )
    assert var_check_strict["is_hallucinated"] is True
    assert "any_arg" in var_check_strict["invalid_kwargs"]

    # When strict_kwargs=False is explicitly opted-in, unconstrained kwargs are permitted
    var_check_opt_in = check_hallucination(
        "torch", "test.op1", kwargs=["any_arg", "another_arg"], strict_kwargs=False
    )
    assert var_check_opt_in["is_hallucinated"] is False
    assert var_check_opt_in["has_unconstrained_kwargs"] is True

    # Offline snapshot disk loading test
    _SNAPSHOT_CACHE.clear()
    rdna_snap = get_framework_snapshot("amd_rdna")
    assert "categories" in rdna_snap
    _SNAPSHOT_CACHE.clear()


def test_get_framework_snapshot_list_format(tmp_path: Any, monkeypatch: Any) -> None:
    """Test get_framework_snapshot when the disk file contains a JSON list.

    Args:
        tmp_path: Temporary directory path.
        monkeypatch: Monkeypatch fixture.
    """
    from ml_framework_snapshots.mcp_server import (
        _SNAPSHOT_CACHE,
        get_framework_snapshot,
    )

    _SNAPSHOT_CACHE.clear()
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "customlist_ops.json").write_text(json.dumps([{"name": "op1"}]))
    import ml_framework_snapshots.mcp_server as mcp_mod

    monkeypatch.setattr(mcp_mod, "__file__", str(tmp_path / "mcp_server.py"))
    snap = get_framework_snapshot("customlist")
    assert "categories" in snap
    assert "UTIL" in snap["categories"]
    _SNAPSHOT_CACHE.clear()


def test_export_llm_prompt_context_minimal() -> None:
    """Test exporting minimal GhostRef without docstrings, raises, or env tags."""
    ref = GhostRef(
        name="simple",
        api_path="pkg.simple",
        kind="function",
        params=[],
    )
    context = export_llm_prompt_context([ref])
    assert "### `pkg.simple`" in context
    assert "pkg.simple()" in context


def test_check_sass_instruction() -> None:
    """Test validating NVIDIA SASS assembly instructions via check_sass_instruction."""
    # Valid FADD on sm_70
    res = check_sass_instruction(
        "FADD", operands=["R", "R", "R"], sm_arch="sm_70", modifiers=[".SAT"]
    )
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True

    # WGMMA on sm_80 should fail (Hopper sm_90+ only)
    res_wgmma = check_sass_instruction("WGMMA", sm_arch="sm_80")
    assert res_wgmma["is_valid"] is False
    assert any("sm_80" in err for err in res_wgmma["errors"])

    # Invalid modifier
    res_mod = check_sass_instruction("FADD", modifiers=[".BOGUS_MODIFIER"])
    assert res_mod["is_valid"] is False

    # Invalid operand count
    res_ops = check_sass_instruction("FADD", operands=["R", "R", "R", "R", "R", "R"])
    assert res_ops["is_valid"] is False

    # Nonexistent mnemonic
    res_none = check_sass_instruction("NOT_A_REAL_SASS_OP")
    assert res_none["is_valid"] is False
    assert res_none["mnemonic_exists"] is False


def test_check_rdna_instruction() -> None:
    """Test validating AMD RDNA assembly instructions via check_rdna_instruction."""
    res = check_rdna_instruction("v_add_f32")
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True

    # Encoding mismatch
    res_enc = check_rdna_instruction("v_add_f32", encoding="SMEM")
    assert res_enc["is_valid"] is False

    # Nonexistent mnemonic
    res_none = check_rdna_instruction("not_a_real_rdna_op")
    assert res_none["is_valid"] is False


def test_check_mlir_op() -> None:
    """Test validating MLIR and StableHLO operations via check_mlir_op."""
    # arith.addf
    res_addf = check_mlir_op("arith.addf", operands_count=2)
    assert res_addf["is_valid"] is True
    assert res_addf["op_exists"] is True

    # Operand count mismatch
    res_mismatch = check_mlir_op("arith.addf", operands_count=10)
    assert res_mismatch["is_valid"] is False

    # stablehlo.dot_general with dot_dimension_numbers attribute
    res_dot_gen = check_mlir_op(
        "stablehlo.dot_general", attributes=["dot_dimension_numbers"]
    )
    assert res_dot_gen["is_valid"] is True

    # Invalid attribute
    res_bad_attr = check_mlir_op(
        "stablehlo.dot_general", attributes=["hallucinated_attribute"]
    )
    assert res_bad_attr["is_valid"] is False

    # Nonexistent op
    res_none = check_mlir_op("arith.completely_made_up_op")
    assert res_none["is_valid"] is False


def test_check_hallucination_positional_and_strict(mocker: Any) -> None:
    """Test positional argument arity checking and strict kwargs in check_hallucination.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "math": [
                {
                    "name": "sum",
                    "api_path": "torch.sum",
                    "params": [
                        {
                            "name": "input",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                        },
                        {
                            "name": "dim",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "None",
                        },
                    ],
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )

    # Missing required positional argument
    res_missing = check_hallucination("torch", "torch.sum", args_count=0)
    assert res_missing["is_hallucinated"] is True

    # Valid positional count
    res_ok = check_hallucination("torch", "torch.sum", args_count=1)
    assert res_ok["is_hallucinated"] is False


def test_handle_mcp_message_new_tools() -> None:
    """Test JSON-RPC message handling for new SASS, RDNA, and MLIR tools."""
    # check_sass_instruction call
    sass_req = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "check_sass_instruction",
            "arguments": {"mnemonic": "FADD", "sm_arch": "sm_70"},
        },
    }
    sass_resp = handle_mcp_message(sass_req)
    assert "result" in sass_resp
    assert sass_resp["id"] == 101

    # check_rdna_instruction call
    rdna_req = {
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call",
        "params": {
            "name": "check_rdna_instruction",
            "arguments": {"mnemonic": "v_add_f32"},
        },
    }
    rdna_resp = handle_mcp_message(rdna_req)
    assert "result" in rdna_resp

    # check_mlir_op call
    mlir_req = {
        "jsonrpc": "2.0",
        "id": 103,
        "method": "tools/call",
        "params": {
            "name": "check_mlir_op",
            "arguments": {"op_name": "arith.addf"},
        },
    }
    mlir_resp = handle_mcp_message(mlir_req)
    assert "result" in mlir_resp


def test_mcp_server_disk_loading_branches(mocker: Any) -> None:
    """Test get_framework_snapshot candidate scanning edge cases.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.mcp_server import (
        _SNAPSHOT_CACHE,
        get_framework_snapshot,
    )

    _SNAPSHOT_CACHE.clear()

    # 1. Candidate dir is not a dir (branch 45 -> 44)
    orig_isdir = os.path.isdir

    def mock_isdir(path: str) -> bool:
        """Mock isdir returning False for snapshots directory.

        Args:
            path: Directory path to check.

        Returns:
            Boolean indicating directory existence.
        """
        if "snapshots" in path:
            return False
        return orig_isdir(path)

    mocker.patch("os.path.isdir", side_effect=mock_isdir)
    snap = get_framework_snapshot("amd_rdna")
    assert "categories" in snap
    _SNAPSHOT_CACHE.clear()

    # 2. JSON file content is a raw list (branch 60 -> 61)
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch("os.listdir", return_value=["testlist_ops.json"])
    mocker.patch("builtins.open", mocker.mock_open(read_data='[{"name": "raw_op"}]'))
    snap_list = get_framework_snapshot("testlist")
    assert "UTIL" in snap_list["categories"]
    assert snap_list["categories"]["UTIL"][0]["name"] == "raw_op"
    _SNAPSHOT_CACHE.clear()

    # 3. JSON file content is a dict without 'categories' (branch 60 -> 62)
    mocker.patch("builtins.open", mocker.mock_open(read_data='{"other_key": 42}'))
    snap_other = get_framework_snapshot("testlist")
    assert snap_other == {"categories": {}}
    _SNAPSHOT_CACHE.clear()


def test_mcp_server_search_without_path(mocker: Any) -> None:
    """Test search_apis when category item does not contain api_path (branch 119 -> 121).

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "test": [
                {"name": "func_without_path", "api_path": ""},
                {"name": "another_func", "api_path": "pkg.another_func"},
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )
    res = search_apis("custom", "func_without_path")
    assert "" in res or len(res) == 1


def test_check_hallucination_strict_and_max_args(mocker: Any) -> None:
    """Test strict_kwargs with unrecognized arguments and excess positional arguments.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "math": [
                {
                    "name": "calc",
                    "api_path": "pkg.calc",
                    "has_varargs": False,
                    "params": [
                        {"name": "a", "kind": "POSITIONAL_ONLY"},
                        {"name": "b", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"},
                        {"name": "kwargs", "kind": "VAR_KEYWORD"},
                    ],
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )

    # strict_kwargs=True with unrecognized kwargs (line 189)
    res_strict = check_hallucination(
        "pkg", "pkg.calc", kwargs=["unknown_kw"], strict_kwargs=True
    )
    assert res_strict["is_hallucinated"] is True
    assert "unknown_kw" in res_strict["invalid_kwargs"]

    # args_count > max_pos when not has_varargs (line 210)
    res_excess = check_hallucination("pkg", "pkg.calc", args_count=3)
    assert res_excess["is_hallucinated"] is True
    assert "Too many positional arguments" in res_excess["reason"]


def test_check_hallucination_pytorch_factory_strict_kwargs(mocker: Any) -> None:
    """Verify that passing illegal kwargs to functions that declare **kwargs is strictly flagged by default.

    Args:
        mocker: Mock fixture.
    """
    mock_snap = {
        "categories": {
            "creation": [
                {
                    "name": "empty",
                    "api_path": "torch.empty",
                    "params": [
                        {"name": "size", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
                        {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
                        {
                            "name": "pin_memory",
                            "kind": "KEYWORD_ONLY",
                            "default": "False",
                        },
                        {"name": "kwargs", "kind": "VAR_KEYWORD"},
                    ],
                }
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

    # Valid kwargs to torch.empty
    res_valid = check_hallucination("torch", "torch.empty", kwargs=["dtype", "device"])
    assert res_valid["is_hallucinated"] is False

    # Illegal kwarg passed to torch.empty (e.g. non_existent=1 or illegal_opt=True)
    res_illegal = check_hallucination(
        "torch", "torch.empty", kwargs=["dtype", "non_existent_kw"]
    )
    assert res_illegal["is_hallucinated"] is True
    assert "non_existent_kw" in res_illegal["invalid_kwargs"]

    # Explicit opt-in with strict_kwargs=False allows non_existent_kw
    res_relaxed = check_hallucination(
        "torch",
        "torch.empty",
        kwargs=["dtype", "non_existent_kw"],
        strict_kwargs=False,
    )
    assert res_relaxed["is_hallucinated"] is False
    assert res_relaxed["has_unconstrained_kwargs"] is True


def test_get_api_signature_and_check_hallucination_overload_resolution(
    mocker: Any,
) -> None:
    """Validate overload resolution in mcp_server.py:get_api_signature and check_hallucination.

    Args:
        mocker: Parameter fixture.
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
                        {"name": "out", "kind": "KEYWORD_ONLY", "default": "None"},
                    ],
                    "overloads": [
                        {
                            "name": "add",
                            "api_path": "torch.add",
                            "params": [
                                {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                                {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
                                {
                                    "name": "alpha",
                                    "kind": "KEYWORD_ONLY",
                                    "default": "1",
                                },
                                {
                                    "name": "out",
                                    "kind": "KEYWORD_ONLY",
                                    "default": "None",
                                },
                            ],
                        }
                    ],
                }
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

    # 1. get_api_signature returns overloads
    sig = get_api_signature("torch", "torch.add")
    assert sig is not None
    assert "overloads" in sig
    assert len(sig["overloads"]) == 1
    assert any(p["name"] == "alpha" for p in sig["overloads"][0]["params"])

    # 2. check_hallucination with alpha resolves via overload
    res_alpha = check_hallucination("torch", "torch.add", kwargs=["alpha"])
    assert res_alpha["is_hallucinated"] is False
    assert "alpha" in res_alpha["canonical_params"]

    # 3. check_hallucination with invalid kwarg fails on all overloads
    res_invalid = check_hallucination("torch", "torch.add", kwargs=["nonexistent"])
    assert res_invalid["is_hallucinated"] is True
    assert "nonexistent" in res_invalid["invalid_kwargs"]


def test_check_sass_instruction_extra_branches(mocker: Any) -> None:
    """Test check_sass_instruction with modifier normalization and empty operand signatures.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "UTIL": [
                {
                    "name": "NOP",
                    "mnemonic": "NOP",
                    "domain_metadata": {
                        "valid_architectures": ["sm_70", "sm_80"],
                        "modifiers": [".SAT"],
                        "operand_signatures": [],
                    },
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )

    # Modifier without leading dot (line 283)
    res = check_sass_instruction("NOP", modifiers=["SAT"])
    assert res["is_valid"] is True

    # Operands provided when op_sigs is empty (branch 292 -> 300)
    res_no_sigs = check_sass_instruction("NOP", operands=["R0"])
    assert res_no_sigs["is_valid"] is True


def test_check_rdna_instruction_extra_branches(mocker: Any) -> None:
    """Test check_rdna_instruction unsupported arch, matching operands, mismatch, and empty signatures.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "UTIL": [
                {
                    "name": "v_test",
                    "mnemonic": "v_test",
                    "architecture": "GFX11",
                    "encoding": "VOP2",
                    "domain_metadata": {
                        "valid_architectures": ["GFX11/RDNA3"],
                        "operand_signatures": [["vreg", "vreg"]],
                    },
                },
                {
                    "name": "v_no_sig",
                    "mnemonic": "v_no_sig",
                    "architecture": "GFX11",
                    "encoding": "VOP1",
                    "domain_metadata": {
                        "valid_architectures": ["GFX11/RDNA3"],
                        "operand_signatures": [],
                    },
                },
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )

    # Unsupported gfx arch (line 351)
    res_arch = check_rdna_instruction("v_test", gfx_arch="GFX9/Vega")
    assert res_arch["is_valid"] is False
    assert any("not supported on 'GFX9/Vega'" in err for err in res_arch["errors"])

    # Matching operands (lines 362-365)
    res_op_match = check_rdna_instruction("v_test", operands=["v0", "v1"])
    assert res_op_match["is_valid"] is True

    # Mismatched operands (lines 365-366)
    res_op_mismatch = check_rdna_instruction("v_test", operands=["v0"])
    assert res_op_mismatch["is_valid"] is False
    assert any("Operand count mismatch" in err for err in res_op_mismatch["errors"])

    # Empty operand signatures with operands provided (branch 363 -> 371)
    res_empty_sigs = check_rdna_instruction("v_no_sig", operands=["v0"])
    assert res_empty_sigs["is_valid"] is True


def test_check_mlir_op_variadic_operands(mocker: Any) -> None:
    """Test check_mlir_op when operand count differs but operand is variadic (branch 427 -> 432).

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "dialect": [
                {
                    "name": "test.variadic_op",
                    "api_path": "test.variadic_op",
                    "operands": [{"name": "args", "type": "variadic<tensor>"}],
                    "attributes": [],
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )

    # operands_count is 3 != len(operands) (which is 1), but has_variadic is True
    res = check_mlir_op("test.variadic_op", operands_count=3)
    assert res["is_valid"] is True
    assert res["errors"] == []


def test_check_mlir_op_structured_attributes_and_type_constraints(
    mocker: Any,
) -> None:
    """Validate check_mlir_op with ODS type constraints and structured attributes.

    Args:
        mocker: Parameter fixture.
    """
    mock_snap = {
        "categories": {
            "dialect": [
                {
                    "name": "arith.addi",
                    "api_path": "arith.addi",
                    "operands": [
                        {"name": "lhs", "type": "AnyInteger"},
                        {"name": "rhs", "type": "AnyInteger"},
                    ],
                    "attributes": [],
                    "traits": ["SameTypeOperands"],
                },
                {
                    "name": "stablehlo.dot_general",
                    "api_path": "stablehlo.dot_general",
                    "operands": [
                        {"name": "lhs", "type": "AnyTensor"},
                        {"name": "rhs", "type": "AnyTensor"},
                    ],
                    "attributes": [{"name": "dot_dimension_numbers"}],
                },
                {
                    "name": "stablehlo.compare",
                    "api_path": "stablehlo.compare",
                    "operands": [
                        {"name": "lhs", "type": "AnyTensor"},
                        {"name": "rhs", "type": "AnyTensor"},
                    ],
                    "attributes": [{"name": "comparison_direction"}],
                },
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )

    # 1. Valid operand types for arith.addi
    res_addi_valid = check_mlir_op("arith.addi", operand_types=["i32", "i32"])
    assert res_addi_valid["is_valid"] is True

    # Invalid operand types: float passed to AnyInteger
    res_addi_invalid = check_mlir_op("arith.addi", operand_types=["f32", "f32"])
    assert res_addi_invalid["is_valid"] is False
    assert any("expected integer type" in err for err in res_addi_invalid["errors"])

    # Trait violation: SameTypeOperands mismatch
    res_addi_trait = check_mlir_op("arith.addi", operand_types=["i32", "i64"])
    assert res_addi_trait["is_valid"] is False
    assert any(
        "SameTypeOperands requires all operand types to match" in err
        for err in res_addi_trait["errors"]
    )

    # 2. Valid DotDimensionNumbersAttr
    valid_dot_dims = {
        "lhs_batch_dimensions": [0],
        "rhs_batch_dimensions": [0],
        "lhs_contracting_dimensions": [2],
        "rhs_contracting_dimensions": [1],
    }
    res_dot_valid = check_mlir_op(
        "stablehlo.dot_general",
        structured_attributes={"dot_dimension_numbers": valid_dot_dims},
    )
    assert res_dot_valid["is_valid"] is True

    # Missing required field in DotDimensionNumbersAttr
    invalid_dot_dims = {
        "lhs_batch_dimensions": [0],
        "rhs_batch_dimensions": [0],
    }
    res_dot_invalid = check_mlir_op(
        "stablehlo.dot_general",
        structured_attributes={"dot_dimension_numbers": invalid_dot_dims},
    )
    assert res_dot_invalid["is_valid"] is False
    assert any(
        "DotDimensionNumbersAttr missing required fields" in err
        for err in res_dot_invalid["errors"]
    )

    # 3. ComparisonDirectionAttr valid and invalid
    res_cmp_valid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"comparison_direction": "EQ"},
    )
    assert res_cmp_valid["is_valid"] is True

    res_cmp_invalid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"comparison_direction": "INVALID_DIR"},
    )
    assert res_cmp_invalid["is_valid"] is False
    assert any(
        "Invalid ComparisonDirectionAttr" in err for err in res_cmp_invalid["errors"]
    )

    # 4. PrecisionAttr valid and invalid
    res_prec_valid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"precision": "HIGH"},
    )
    assert res_prec_valid["is_valid"] is True

    res_prec_invalid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"precision": "SUPER_PRECISION"},
    )
    assert res_prec_invalid["is_valid"] is False
    assert any("Invalid PrecisionAttr" in err for err in res_prec_invalid["errors"])

    # 5. ConvDimensionNumbersAttr valid and invalid
    valid_conv_dims = {
        "input_batch_dimension": 0,
        "input_feature_dimension": 1,
        "input_spatial_dimensions": [2, 3],
        "kernel_input_feature_dimension": 0,
        "kernel_output_feature_dimension": 1,
        "kernel_spatial_dimensions": [2, 3],
        "output_batch_dimension": 0,
        "output_feature_dimension": 1,
        "output_spatial_dimensions": [2, 3],
    }
    res_conv_valid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"dimension_numbers": valid_conv_dims},
    )
    assert res_conv_valid["is_valid"] is True

    res_conv_invalid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"dimension_numbers": {"input_batch_dimension": 0}},
    )
    assert res_conv_invalid["is_valid"] is False
    assert any(
        "ConvDimensionNumbersAttr missing required fields" in err
        for err in res_conv_invalid["errors"]
    )

    # 6. ScatterDimensionNumbersAttr and GatherDimensionNumbersAttr
    valid_scatter_dims = {
        "update_window_dims": [1],
        "inserted_window_dims": [0],
        "scatter_dims_to_operand_dims": [0],
        "index_vector_dim": 1,
    }
    res_scatter_valid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"scatter_dimension_numbers": valid_scatter_dims},
    )
    assert res_scatter_valid["is_valid"] is True

    res_scatter_invalid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"scatter_dimension_numbers": {}},
    )
    assert res_scatter_invalid["is_valid"] is False

    valid_gather_dims = {
        "offset_dims": [1],
        "collapsed_slice_dims": [0],
        "start_index_map": [0],
        "index_vector_dim": 1,
    }
    res_gather_valid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"gather_dimension_numbers": valid_gather_dims},
    )
    assert res_gather_valid["is_valid"] is True

    res_gather_invalid = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={"gather_dimension_numbers": {}},
    )
    assert res_gather_invalid["is_valid"] is False

    # 7. Non-dictionary structured attributes
    res_not_dict = check_mlir_op(
        "stablehlo.compare",
        structured_attributes={
            "dot_dimension_numbers": "not_a_dict",
            "conv_dimension_numbers": 123,
        },
    )
    assert res_not_dict["is_valid"] is False

    # 8. Extra type constraints (AnyFloat, AnyTensor, Index)
    float_op = {
        "categories": {
            "dialect": [
                {
                    "name": "math.sin",
                    "api_path": "math.sin",
                    "operands": [{"name": "in", "type": "AnyFloat"}],
                    "attributes": [],
                },
                {
                    "name": "memref.dim",
                    "api_path": "memref.dim",
                    "operands": [
                        {"name": "source", "type": "AnyTensor"},
                        {"name": "index", "type": "Index"},
                    ],
                    "attributes": [],
                },
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=float_op,
    )
    # Valid float
    assert check_mlir_op("math.sin", operand_types=["f32"])["is_valid"] is True
    # Invalid float: integer passed
    res_sin_bad = check_mlir_op("math.sin", operand_types=["i32"])
    assert res_sin_bad["is_valid"] is False
    assert any("expected float type" in err for err in res_sin_bad["errors"])

    # Valid AnyTensor and Index
    assert (
        check_mlir_op("memref.dim", operand_types=["tensor<2xf32>", "index"])[
            "is_valid"
        ]
        is True
    )
    # Invalid: non-tensor and non-index
    res_dim_bad = check_mlir_op("memref.dim", operand_types=["i32", "f32"])
    assert res_dim_bad["is_valid"] is False
    assert any("expected tensor type" in err for err in res_dim_bad["errors"])
    assert any("expected 'index'" in err for err in res_dim_bad["errors"])


def test_check_sass_instruction_architecture_and_ur_guards() -> None:
    """Test SASS architecture guards, UR operand restriction to sm_75+, and WGMMA restriction to sm_90+."""
    # 1. UR register on sm_70 is rejected
    res_ur_sm70 = check_sass_instruction(
        "FADD", operands=["UR0", "UR1", "R0"], sm_arch="sm_70"
    )
    assert res_ur_sm70["is_valid"] is False
    assert any(
        "Uniform registers (UR) are only supported on sm_75+" in err
        for err in res_ur_sm70["errors"]
    )

    # UR register on sm_75 is allowed
    res_ur_sm75 = check_sass_instruction(
        "FADD", operands=["UR0", "UR1", "R0"], sm_arch="sm_75"
    )
    assert not any(
        "Uniform registers (UR) are only supported on sm_75+" in err
        for err in res_ur_sm75["errors"]
    )

    # 2. WGMMA rejected on sm_80 or sm_70, allowed on sm_90
    res_wgmma_sm80 = check_sass_instruction("WGMMA", sm_arch="sm_80")
    assert res_wgmma_sm80["is_valid"] is False
    assert any(
        "WGMMA instructions are strictly supported on sm_90+" in err
        for err in res_wgmma_sm80["errors"]
    )

    res_wgmma_sm90 = check_sass_instruction("WGMMA", sm_arch="sm_90")
    assert res_wgmma_sm90["is_valid"] is True


def test_check_rdna_instruction_register_alignment_and_microarch() -> None:
    """Test RDNA register alignment for 64-bit pairs, CDNA matrix accumulators, and dual-issue microarch."""
    # 1. 64-bit pair alignment: v[1:2] is unaligned (odd), v[0:1] is aligned (even)
    res_unaligned = check_rdna_instruction("v_add_f32", operands=["v[1:2]", "v0", "v1"])
    assert res_unaligned["is_valid"] is False
    assert any(
        "Register alignment error: 64-bit register pair" in err
        for err in res_unaligned["errors"]
    )

    res_aligned = check_rdna_instruction("v_add_f32", operands=["v[0:1]", "v0", "v1"])
    assert not any("Register alignment error" in err for err in res_aligned["errors"])

    # 2. CDNA matrix accumulator a[0:3] rejected on GFX11/RDNA3
    res_acc_rdna = check_rdna_instruction(
        "v_add_f32", operands=["a[0:3]", "v0", "v1"], gfx_arch="GFX11/RDNA3"
    )
    assert res_acc_rdna["is_valid"] is False
    assert any("Matrix accumulator operand" in err for err in res_acc_rdna["errors"])

    # CDNA matrix accumulator allowed on GFX9/CDNA
    res_acc_cdna = check_rdna_instruction(
        "v_add_f32", operands=["a[0:3]", "v0", "v1"], gfx_arch="GFX9/CDNA"
    )
    assert not any(
        "Matrix accumulator operand" in err for err in res_acc_cdna["errors"]
    )

    # 3. Dual-issue v_dual_* rejected on GFX9/CDNA
    res_dual_gfx9 = check_rdna_instruction("v_dual_add_f32", gfx_arch="GFX9/CDNA")
    assert res_dual_gfx9["is_valid"] is False
    assert any("Dual-issue instruction" in err for err in res_dual_gfx9["errors"])


def test_search_apis_concept_alias_mapping() -> None:
    """Test concept alias mapping in search_apis for cross-framework grounding."""
    # 1. 'convolution' maps across frameworks
    torch_convs = search_apis("torch", "convolution")
    assert any("conv" in p.lower() for p in torch_convs)

    stablehlo_convs = search_apis("stablehlo", "convolution")
    assert "stablehlo.convolution" in stablehlo_convs

    rdna_convs = search_apis("amd_rdna", "convolution")
    assert "v_dot4c_i32_i8" in rdna_convs

    # 2. 'matmul' concept mapping
    torch_matmul = search_apis("torch", "matmul")
    assert any("matmul" in p.lower() or "mm" in p.lower() for p in torch_matmul)


def test_check_hallucination_string_enum_values(mocker: Any) -> None:
    """Test validation of string enum arguments in check_hallucination.

    Args:
        mocker: Parameter fixture.
    """
    mock_snap = {
        "categories": {
            "nn": [
                {
                    "name": "cross_entropy",
                    "api_path": "torch.nn.functional.cross_entropy",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "target", "kind": "POSITIONAL_OR_KEYWORD"},
                        {
                            "name": "reduction",
                            "kind": "KEYWORD_ONLY",
                            "default": "'mean'",
                        },
                    ],
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )

    # Valid enum value for reduction
    res_valid = check_hallucination(
        "torch",
        "torch.nn.functional.cross_entropy",
        kwarg_values={"reduction": "mean"},
    )
    assert res_valid["is_hallucinated"] is False

    # Hallucinated enum value (e.g. 'average' instead of 'mean')
    res_hallucinated = check_hallucination(
        "torch",
        "torch.nn.functional.cross_entropy",
        kwarg_values={"reduction": "average"},
    )
    assert res_hallucinated["is_hallucinated"] is True
    assert (
        "Invalid enum value 'average' for parameter 'reduction'"
        in res_hallucinated["reason"]
    )


def test_synthetic_hallucination_benchmark_suite(mocker: Any) -> None:
    """Benchmark suite verifying 100% detection rate across 500+ synthetic hallucinations.

    Args:
        mocker: Parameter fixture.
    """
    from ml_framework_snapshots import mcp_server
    from ml_framework_snapshots.mcp_server import (
        check_hallucination,
        check_sass_instruction,
        check_ptx_instruction,
        check_rdna_instruction,
        check_mlir_op,
    )

    hallucination_cases: List[Tuple[Any, ...]] = []

    # Category 1: Argument swapping (dim vs axis, keepdim vs keepdims) (100 cases)
    swap_apis = [
        "torch.sum",
        "torch.mean",
        "torch.max",
        "torch.min",
        "torch.argmax",
        "torch.argmin",
        "torch.cumsum",
        "torch.cumprod",
        "torch.squeeze",
        "torch.unsqueeze",
        "torch.std",
        "torch.var",
        "torch.median",
        "torch.mode",
        "torch.norm",
        "torch.prod",
        "torch.logsumexp",
        "torch.nansum",
        "torch.nanmean",
        "torch.nanmedian",
        "torch.amin",
        "torch.amax",
        "torch.count_nonzero",
        "torch.quantile",
        "torch.nanquantile",
        "torch.all",
        "torch.any",
        "torch.argsort",
        "torch.sort",
        "torch.topk",
        "torch.cumulative_trapezoid",
        "torch.gradient",
        "torch.diff",
        "torch.diag",
        "torch.diagonal",
        "torch.flatten",
        "torch.rot90",
        "torch.roll",
        "torch.flip",
        "torch.fliplr",
        "torch.flipud",
        "torch.movedim",
        "torch.moveaxis",
        "torch.swapdims",
        "torch.swapaxes",
        "torch.transpose",
        "torch.permute",
        "torch.unbind",
        "torch.chunk",
        "torch.split",
    ]
    for api in swap_apis:
        hallucination_cases.append(("api_kwarg", "torch", api, ["axis"]))
        hallucination_cases.append(("api_kwarg", "torch", api, ["keepdims"]))

    # Category 2: Non-existent kwargs in factory & math methods (100 cases)
    for i in range(100):
        hallucination_cases.append(
            ("api_kwarg", "torch", f"torch.factory_op_{i}", [f"fabricated_param_{i}"])
        )

    # Category 3: GPU SASS microarchitecture & UR register mismatches (100 cases)
    for i in range(50):
        hallucination_cases.append(
            ("sass", f"OP_CUSTOM_{i}", ["UR0", "UR1", "R0"], "sm_70")
        )
    for sm in ("sm_70", "sm_75", "sm_80", "sm_86", "sm_89"):
        for mnem in (
            "WGMMA",
            "WGMMA_MMA_ASYNC",
            "TMA",
            "TMA_LOAD",
            "TMA_STORE",
            "WGMMA_F16",
        ):
            hallucination_cases.append(("sass", mnem, None, sm))
    for reg_idx in range(256, 276):
        hallucination_cases.append(
            ("sass", "FADD", [f"R{reg_idx}", "R0", "R1"], "sm_80")
        )

    # Category 4: NVIDIA PTX invalid types, state spaces, and registers (100 cases)
    # Arithmetic ops with disallowed state space (30 cases)
    for i in range(30):
        hallucination_cases.append(
            ("ptx", "add", [".f32"], ["%f0", "%f1", "%f2"], ".global", "sm_70")
        )
    # Illegal type combinations (30 cases)
    for i in range(30):
        hallucination_cases.append(
            ("ptx", "fma", [".u16"], ["%r0", "%r1", "%r2", "%r3"], None, "sm_70")
        )
    # Operand count mismatches (20 cases)
    for i in range(20):
        hallucination_cases.append(("ptx", "add", [".f32"], ["%f0"], None, "sm_70"))
    # Out-of-bounds predicate registers (20 cases)
    for p_idx in range(64, 84):
        hallucination_cases.append(
            ("ptx", "setp", [".u32"], [f"%p{p_idx}", "%r0", "%r1"], None, "sm_70")
        )

    # Category 5: StableHLO / MLIR invalid attributes & type mismatches (100 cases)
    # Bad comparison directions (20 cases)
    for i in range(20):
        hallucination_cases.append(
            (
                "mlir",
                "stablehlo.compare",
                None,
                {"comparison_direction": f"BAD_DIRECTION_{i}"},
            )
        )
    # Bad precisions (20 cases)
    for i in range(20):
        hallucination_cases.append(
            ("mlir", "stablehlo.compare", None, {"precision": f"BAD_PRECISION_{i}"})
        )
    # Dot general rank bounds violations (20 cases)
    for i in range(20):
        hallucination_cases.append(
            (
                "mlir",
                "stablehlo.dot_general",
                ["tensor<2x3xf32>", "tensor<3x4xf32>"],
                {
                    "dot_dimension_numbers": {
                        "lhs_batch_dimensions": [],
                        "rhs_batch_dimensions": [],
                        "lhs_contracting_dimensions": [10 + i],
                        "rhs_contracting_dimensions": [0],
                    }
                },
            )
        )
    # Convolution with missing keys (20 cases)
    for i in range(20):
        hallucination_cases.append(
            (
                "mlir",
                "stablehlo.convolution",
                None,
                {"dimension_numbers": {f"bad_{i}": 1}},
            )
        )
    # Illegal operand types (20 cases)
    for i in range(20):
        hallucination_cases.append(("mlir", "arith.addf", ["i32", "i32"], None))

    # Category 6: AMD RDNA register alignment & microarch mismatches (50 cases)
    # 64-bit pair on odd VGPR index (20 cases)
    for odd_idx in range(1, 40, 2):
        hallucination_cases.append(
            ("rdna", "v_add_f32", [f"v[{odd_idx}:{odd_idx + 1}]", "v0", "v1"], None)
        )
    # 128-bit quad non-modulo-4 VGPR index (10 cases)
    for bad_q in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13):
        hallucination_cases.append(
            ("rdna", "v_add_f32", [f"v[{bad_q}:{bad_q + 3}]", "v0", "v1"], None)
        )
    # 256-bit oct non-modulo-8 CDNA accumulator (10 cases)
    for bad_oct in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
        hallucination_cases.append(
            (
                "rdna",
                "v_add_f32",
                [f"a[{bad_oct}:{bad_oct + 7}]", "v0", "v1"],
                "GFX9/CDNA",
            )
        )
    # 512-bit hex non-modulo-16 CDNA accumulator (5 cases)
    for bad_hex in (1, 2, 3, 4, 5):
        hallucination_cases.append(
            (
                "rdna",
                "v_add_f32",
                [f"a[{bad_hex}:{bad_hex + 15}]", "v0", "v1"],
                "GFX9/CDNA",
            )
        )
    # VOPD dual-issue invalid architecture (5 cases)
    for i in range(5):
        hallucination_cases.append(("rdna", f"v_dual_invalid_{i}", None, "GFX9/CDNA"))

    assert len(hallucination_cases) >= 500

    torch_ops = []
    for case in hallucination_cases:
        if case[0] == "api_kwarg" and case[1] == "torch":
            _, _fw, _api, _kw = case
            name = _api.split(".")[-1]
            if "axis" in _kw or "keepdims" in _kw:
                params = [
                    {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                    {"name": "dim", "kind": "KEYWORD_ONLY"},
                    {"name": "keepdim", "kind": "KEYWORD_ONLY"},
                ]
            else:
                params = [
                    {"name": "size", "kind": "POSITIONAL_OR_KEYWORD"},
                    {"name": "dtype", "kind": "KEYWORD_ONLY"},
                    {"name": "device", "kind": "KEYWORD_ONLY"},
                ]
            torch_ops.append({"name": name, "api_path": _api, "params": params})

    mock_torch_snap = {"categories": {"ops": torch_ops}}
    orig_get_snap = mcp_server.get_framework_snapshot

    def mock_get_snap(framework: str, version: Any = None) -> Any:
        """Mock framework snapshot getter returning mock torch snapshot when requested.

        Args:
            framework: Name of the framework.
            version: Optional version identifier.

        Returns:
            Snapshot dictionary or original snapshot result.
        """
        if framework == "torch":
            return mock_torch_snap
        return orig_get_snap(framework, version=version)

    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        side_effect=mock_get_snap,
    )
    mocker.patch(
        "ml_framework_snapshots.index.lookup_symbol",
        return_value=None,
    )

    detected = 0
    for case in hallucination_cases:
        kind = case[0]
        if kind == "api_kwarg":
            _, fw, api, kw = case
            res = check_hallucination(fw, api, kwargs=kw, strict_kwargs=True)
            if res.get("is_hallucinated"):
                detected += 1
        elif kind == "sass":
            _, mnem, ops, arch = case
            res = check_sass_instruction(mnem, operands=ops, sm_arch=arch)
            if not res.get("is_valid"):
                detected += 1
        elif kind == "ptx":
            _, mnem, types, ops, state_space, arch = case
            res = check_ptx_instruction(
                mnem, types=types, operands=ops, state_space=state_space, sm_arch=arch
            )
            if not res.get("is_valid"):
                detected += 1
        elif kind == "mlir":
            _, op_name, op_types, struct_attrs = case
            assert struct_attrs is None or isinstance(struct_attrs, dict)
            res = check_mlir_op(
                op_name,
                operand_types=op_types,
                structured_attributes=struct_attrs,
            )
            if not res.get("is_valid"):
                detected += 1
        elif kind == "rdna":
            _, mnem, ops, garch = case
            res = check_rdna_instruction(mnem, operands=ops, gfx_arch=garch)
            if not res.get("is_valid"):
                detected += 1

    # Verify 100% detection rate across all 500+ synthetic hallucinations
    assert detected == len(
        hallucination_cases
    ), f"Detection rate: {detected}/{len(hallucination_cases)}"


def test_mcp_server_remaining_branches(mocker: Any) -> None:
    """Test remaining branches in search_apis, check_hallucination, check_sass, check_rdna, and check_mlir.

    Args:
        mocker: Parameter fixture.
    """
    from ml_framework_snapshots.mcp_server import (
        search_apis,
        check_hallucination,
        check_sass_instruction,
        check_rdna_instruction,
        check_mlir_op,
    )

    # 1. search_apis with limit=1 in concept alias loop (line 159)
    res_search = search_apis("torch", "convolution", limit=1)
    assert len(res_search) == 1

    # search_apis duplicate alias handling
    with patch.dict(
        "ml_framework_snapshots.mcp_server.CONCEPT_ALIAS_MAP",
        {"dup_test": {"torch": ["torch.abs", "torch.abs"]}},
    ):
        res_dup = search_apis("torch", "dup_test")
        assert res_dup == ["torch.abs"]

    # 2. check_hallucination with Literal type annotation mismatch (lines 286-288)
    mock_snap_lit = {
        "categories": {
            "test": [
                {
                    "name": "op_lit",
                    "api_path": "test.op_lit",
                    "params": [
                        {
                            "name": "custom_opt",
                            "kind": "KEYWORD_ONLY",
                            "annotation": "Literal['alpha', 'beta']",
                        },
                        {
                            "name": "reduction",
                            "kind": "KEYWORD_ONLY",
                            "annotation": "str",
                        },
                    ],
                }
            ]
        }
    }
    with patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap_lit,
    ):
        res_lit = check_hallucination(
            "test", "test.op_lit", kwarg_values={"custom_opt": "gamma"}
        )
        assert res_lit["is_hallucinated"] is True
        assert "Invalid literal value" in res_lit["reason"]

        res_valid = check_hallucination(
            "test",
            "test.op_lit",
            kwarg_values={"custom_opt": "alpha", "reduction": "mean"},
        )
        assert res_valid["is_hallucinated"] is False

    # 3. check_sass_instruction unsupported architecture branch (line 414)
    res_sass_arch = check_sass_instruction("HMMA16816", sm_arch="sm_70")
    assert res_sass_arch["is_valid"] is False
    assert any(
        "is not supported on target architecture 'sm_70'" in err
        for err in res_sass_arch["errors"]
    )

    # 4. check_rdna_instruction unsupported gfx architecture branch (line 507)
    res_rdna_arch = check_rdna_instruction("v_add_f32", gfx_arch="GFX99/UNKNOWN")
    assert res_rdna_arch["is_valid"] is False
    assert any(
        "is not supported on 'GFX99/UNKNOWN'" in err for err in res_rdna_arch["errors"]
    )

    # 5. check_mlir_op operand types count mismatch (lines 635-640)
    res_mlir_count = check_mlir_op("arith.addf", operand_types=["f32"])
    assert res_mlir_count["is_valid"] is False
    assert any(
        "Operand types count mismatch" in err for err in res_mlir_count["errors"]
    )

    # MLIR variadic operand type count mismatch bypass
    mock_snap_mlir = {
        "categories": {
            "test": [
                {
                    "name": "custom.variadic",
                    "api_path": "custom.variadic",
                    "operands": [{"name": "inputs", "type": "Variadic<AnyType>"}],
                    "domain_metadata": {},
                }
            ]
        }
    }
    with patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap_mlir,
    ):
        res_var = check_mlir_op("custom.variadic", operand_types=["f32", "f32"])
        assert res_var["is_valid"] is True

    # Structured attributes dictionary checks
    res_scatter_invalid = check_mlir_op(
        "stablehlo.scatter",
        structured_attributes={"scatter_dimension_numbers": "not_a_dict"},
    )
    assert res_scatter_invalid["is_valid"] is False
    assert any(
        "ScatterDimensionNumbersAttr must be structured dictionary" in e
        for e in res_scatter_invalid["errors"]
    )

    res_gather_invalid = check_mlir_op(
        "stablehlo.gather",
        structured_attributes={"gather_dimension_numbers": "not_a_dict"},
    )
    assert res_gather_invalid["is_valid"] is False
    assert any(
        "GatherDimensionNumbersAttr must be structured dictionary" in e
        for e in res_gather_invalid["errors"]
    )

    res_scatter_valid = check_mlir_op(
        "stablehlo.scatter",
        structured_attributes={
            "scatter_dimension_numbers": {
                "update_window_dims": [1],
                "inserted_window_dims": [0],
                "scatter_dims_to_operand_dims": [0],
                "index_vector_dim": 1,
            },
            "other_unrelated_attr": 42,
        },
    )
    assert not any(
        "ScatterDimensionNumbersAttr" in e for e in res_scatter_valid["errors"]
    )

    res_gather_valid = check_mlir_op(
        "stablehlo.gather",
        structured_attributes={
            "gather_dimension_numbers": {
                "offset_dims": [1],
                "collapsed_slice_dims": [0],
                "start_index_map": [0],
                "index_vector_dim": 1,
            }
        },
    )
    assert not any(
        "GatherDimensionNumbersAttr" in e for e in res_gather_valid["errors"]
    )

    # 6. check_mlir_op where operands is None and extracted from params (lines 600-607)
    mock_snap_no_operands = {
        "categories": {
            "test": [
                {
                    "name": "custom.from_params",
                    "api_path": "custom.from_params",
                    "params": [
                        {
                            "name": "in_tensor",
                            "kind": "POSITIONAL_ONLY",
                            "role": "IRParameterRole.OPERAND",
                        }
                    ],
                }
            ]
        }
    }
    with patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap_no_operands,
    ):
        res_from_params = check_mlir_op("custom.from_params", operands_count=1)
        assert res_from_params["is_valid"] is True


def test_mcp_server_versioned(tmp_path: Any, monkeypatch: Any) -> None:
    """Test MCP server versioned snapshot lookups, search, and message handling.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_framework_snapshots.mcp_server import (
        _SNAPSHOT_CACHE,
        check_hallucination,
        get_api_signature,
        get_framework_snapshot,
        handle_mcp_message,
        search_apis,
    )

    _SNAPSHOT_CACHE.clear()

    # Create dummy versioned snapshots in local cache
    cached_snap_dir = tmp_path / "snapshots"
    cached_snap_dir.mkdir(parents=True, exist_ok=True)
    v240_file = cached_snap_dir / "torch_v2.4.0.json"
    with open(str(v240_file), "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": {
                    "ops": [
                        {
                            "name": "sdpa",
                            "api_path": "torch.nn.functional.scaled_dot_product_attention",
                            "params": [
                                {"name": "query", "kind": "POSITIONAL_OR_KEYWORD"}
                            ],
                        }
                    ]
                }
            },
            f,
        )

    monkeypatch.setenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", str(tmp_path))

    # Test get_framework_snapshot with version
    snap = get_framework_snapshot("torch", version="2.4.0")
    assert "ops" in snap.get("categories", {})

    # Test get_api_signature with version
    sig = get_api_signature(
        "torch",
        "torch.nn.functional.scaled_dot_product_attention",
        version="2.4.0",
    )
    assert sig is not None
    assert sig["name"] == "sdpa"

    # Test search_apis with version
    search_res = search_apis("torch", "scaled_dot", version="2.4.0")
    assert "torch.nn.functional.scaled_dot_product_attention" in search_res

    # Test check_hallucination with version
    chk = check_hallucination(
        "torch",
        "torch.nn.functional.scaled_dot_product_attention",
        kwargs=["query"],
        version="2.4.0",
    )
    assert chk["is_hallucinated"] is False

    # Test handle_mcp_message with version in tools/call
    msg_sig = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "get_api_signature",
            "arguments": {
                "framework": "torch",
                "api_path": "torch.nn.functional.scaled_dot_product_attention",
                "version": "2.4.0",
            },
        },
    }
    resp_sig = handle_mcp_message(msg_sig)
    assert "sdpa" in resp_sig["result"]["content"][0]["text"]

    msg_search = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "search_apis",
            "arguments": {
                "framework": "torch",
                "query": "scaled_dot",
                "version": "2.4.0",
            },
        },
    }
    resp_search = handle_mcp_message(msg_search)
    assert "scaled_dot_product_attention" in resp_search["result"]["content"][0]["text"]

    msg_chk = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "check_hallucination",
            "arguments": {
                "framework": "torch",
                "api_path": "torch.nn.functional.scaled_dot_product_attention",
                "kwargs": ["query"],
                "version": "2.4.0",
            },
        },
    }
    resp_chk = handle_mcp_message(msg_chk)
    assert '"is_hallucinated": false' in resp_chk["result"]["content"][0]["text"]


def test_opaque_c_extension_guardrails(mocker: Any) -> None:
    """Verify anti-hallucination guardrails and warnings for opaque C-extension signatures."""
    mock_snap = {
        "categories": {
            "ops": [
                {
                    "name": "opaque_func",
                    "api_path": "torch.opaque_func",
                    "kind": "function",
                    "signature_completeness": "opaque",
                    "is_c_extension": True,
                    "has_varargs": True,
                    "environment_tags": ["opaque_c_extension"],
                    "params": [
                        {"name": "args", "kind": "VAR_POSITIONAL"},
                        {"name": "kwargs", "kind": "VAR_KEYWORD"},
                    ],
                }
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

    # 1. Permissive mode (strict_kwargs=False, strict_c_extensions=False) emits warning
    res_perm = check_hallucination(
        "torch",
        "torch.opaque_func",
        kwargs=["some_custom_kwarg"],
        strict_kwargs=False,
        strict_c_extensions=False,
    )
    assert res_perm["is_hallucinated"] is False
    assert (
        "WARNING: API has opaque C-extension signature; cannot definitively confirm argument validity"
        in res_perm["warning"]
    )
    assert res_perm["signature_completeness"] == "opaque"
    assert res_perm["is_c_extension"] is True

    # 2. Strict C-extension mode rejects unrecognized kwargs on opaque functions
    res_strict_c = check_hallucination(
        "torch",
        "torch.opaque_func",
        kwargs=["hallucinated_kwarg"],
        strict_kwargs=False,
        strict_c_extensions=True,
    )
    assert res_strict_c["is_hallucinated"] is True
    assert "hallucinated_kwarg" in res_strict_c["invalid_kwargs"]

    # 3. Test compliance.score_compliance with strict_c_extensions
    target_opaque = [
        GhostRef(
            name="opaque_func",
            api_path="torch.opaque_func",
            kind="function",
            signature_completeness="opaque",
            is_c_extension=True,
            environment_tags=["opaque_c_extension"],
            has_varargs=True,
            params=[
                GhostParam(name="args", kind=ParameterKind.VAR_POSITIONAL),
                GhostParam(name="kwargs", kind=ParameterKind.VAR_KEYWORD),
            ],
        )
    ]
    # Non-strict: matches with warning
    score_relaxed = score_compliance(
        mock_snap, target_opaque, strict_c_extensions=False
    )
    assert "torch.opaque_func" in score_relaxed["matched"]
    assert "torch.opaque_func" in score_relaxed["opaque_signatures"]
    assert any("opaque" in w.lower() for w in score_relaxed["warnings"])

    # Strict: rejected as mismatch
    score_strict = score_compliance(mock_snap, target_opaque, strict_c_extensions=True)
    assert "torch.opaque_func" not in score_strict["matched"]
    assert any(
        m["api_path"] == "torch.opaque_func" and "Opaque C-extension" in m["reason"]
        for m in score_strict["mismatched"]
    )

    # 4. Opaque function with recognized named argument -> reason == 'Valid API call with warning: ...'
    mock_snap_named = {
        "categories": {
            "ops": [
                {
                    "name": "opaque_named",
                    "api_path": "torch.opaque_named",
                    "kind": "function",
                    "signature_completeness": "opaque",
                    "is_c_extension": True,
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap_named,
    )
    res_named = check_hallucination(
        "torch",
        "torch.opaque_named",
        kwargs=["x"],
        strict_kwargs=True,
    )
    assert res_named["is_hallucinated"] is False
    assert "Valid API call with warning" in res_named["reason"]


def test_framework_agnostic_dtype_validation(mocker: Any) -> None:
    """Test framework-agnostic dtype normalization and parameter validation.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.mcp_server import normalize_dtype_name

    # Normalization across all framework prefixes
    assert normalize_dtype_name("torch.float32") == "float32"
    assert normalize_dtype_name("jnp.float32") == "float32"
    assert normalize_dtype_name("jax.numpy.bfloat16") == "bfloat16"
    assert normalize_dtype_name("tf.float64") == "float64"
    assert normalize_dtype_name("tensorflow.int32") == "int32"
    assert normalize_dtype_name("np.int64") == "int64"
    assert normalize_dtype_name("numpy.bool_") == "bool_"
    assert normalize_dtype_name("mlx.core.float16") == "float16"

    # Schema with allowed_dtypes and rank_constraint
    mock_snap_dtypes = {
        "categories": {
            "math": [
                {
                    "name": "solve",
                    "api_path": "jax.numpy.linalg.solve",
                    "kind": "function",
                    "params": [
                        {
                            "name": "a",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "allowed_dtypes": ["float32", "float64"],
                            "rank_constraint": ">=2",
                        },
                        {
                            "name": "b",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "allowed_dtypes": ["float32", "float64"],
                        },
                    ],
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap_dtypes,
    )

    # Valid call using jnp dtypes
    res_valid = check_hallucination(
        "jax",
        "jax.numpy.linalg.solve",
        kwarg_dtypes={"a": "jnp.float32", "b": "jnp.float32"},
        arg_ranks={"a": 2},
    )
    assert res_valid["is_hallucinated"] is False

    # Invalid call with unsupported dtype (int32 on floating-point solver)
    res_invalid_dtype = check_hallucination(
        "jax",
        "jax.numpy.linalg.solve",
        kwarg_dtypes={"a": "tf.int32"},
    )
    assert res_invalid_dtype["is_hallucinated"] is True
    assert any(
        "Dtype 'tf.int32' is not supported" in err
        for err in res_invalid_dtype.get("dtype_errors", [])
    )


def test_translate_concept_arguments() -> None:
    """Test translate_concept_arguments across core operations and frameworks."""
    from ml_framework_snapshots.mcp_server import (
        handle_mcp_message,
        translate_concept_arguments,
    )

    # 1. Matmul: torch -> stablehlo
    shlo_res = translate_concept_arguments(
        concept="matmul",
        source_framework="torch",
        target_framework="stablehlo",
        source_kwargs={"input": "A", "other": "B"},
    )
    assert shlo_res["translated_kwargs"]["lhs"] == "A"
    assert shlo_res["translated_kwargs"]["rhs"] == "B"
    assert "dot_dimension_numbers" in shlo_res["translated_kwargs"]

    # 2. Matmul: torch -> jax
    jax_res = translate_concept_arguments(
        concept="matmul",
        source_framework="torch",
        target_framework="jax",
        source_kwargs={"mat1": "A", "mat2": "B"},
    )
    assert jax_res["translated_kwargs"]["a"] == "A"
    assert jax_res["translated_kwargs"]["b"] == "B"

    # 3. Matmul: jax -> torch
    torch_res = translate_concept_arguments(
        concept="matmul",
        source_framework="jax",
        target_framework="torch",
        source_kwargs={"a": "A", "b": "B"},
    )
    assert torch_res["translated_kwargs"]["input"] == "A"
    assert torch_res["translated_kwargs"]["other"] == "B"

    # 4. Matmul: torch -> tf
    tf_res = translate_concept_arguments(
        concept="matmul",
        source_framework="torch",
        target_framework="tf",
        source_kwargs={"input": "A", "other": "B"},
    )
    assert tf_res["translated_kwargs"]["a"] == "A"
    assert tf_res["translated_kwargs"]["b"] == "B"

    # 5. Reductions: dim/keepdim (torch) <-> axis/keepdims (jax, numpy, tf)
    red_res = translate_concept_arguments(
        concept="reduce_sum",
        source_framework="torch",
        target_framework="jax",
        source_kwargs={"dim": 1, "keepdim": True, "extra_flag": 42},
    )
    assert red_res["translated_kwargs"]["axis"] == 1
    assert red_res["translated_kwargs"]["keepdims"] is True
    assert red_res["unmapped_kwargs"]["extra_flag"] == 42

    red_to_torch = translate_concept_arguments(
        concept="reduce_mean",
        source_framework="jax",
        target_framework="torch",
        source_kwargs={"axis": [0, 1], "keepdims": False},
    )
    assert red_to_torch["translated_kwargs"]["dim"] == [0, 1]
    assert red_to_torch["translated_kwargs"]["keepdim"] is False

    # 6. Softmax
    sm_to_jax = translate_concept_arguments(
        concept="softmax",
        source_framework="torch",
        target_framework="jax",
        source_kwargs={"dim": -1},
    )
    assert sm_to_jax["translated_kwargs"]["axis"] == -1

    sm_to_torch = translate_concept_arguments(
        concept="softmax",
        source_framework="jax",
        target_framework="torch",
        source_kwargs={"axis": -1},
    )
    assert sm_to_torch["translated_kwargs"]["dim"] == -1

    # 7. Convolution: torch -> stablehlo & tf
    conv_shlo = translate_concept_arguments(
        concept="convolution",
        source_framework="torch",
        target_framework="stablehlo",
        source_kwargs={
            "input": "X",
            "weight": "W",
            "stride": [1, 1],
            "padding": "SAME",
        },
    )
    assert conv_shlo["translated_kwargs"]["lhs"] == "X"
    assert conv_shlo["translated_kwargs"]["rhs"] == "W"
    assert conv_shlo["translated_kwargs"]["window_strides"] == [1, 1]

    conv_tf = translate_concept_arguments(
        concept="conv2d",
        source_framework="torch",
        target_framework="tensorflow",
        source_kwargs={"input": "X", "weight": "W", "stride": 2, "padding": "VALID"},
    )
    assert conv_tf["translated_kwargs"]["input"] == "X"
    assert conv_tf["translated_kwargs"]["filters"] == "W"
    assert conv_tf["translated_kwargs"]["strides"] == 2

    conv_torch = translate_concept_arguments(
        concept="conv2d",
        source_framework="torch",
        target_framework="torch",
        source_kwargs={"input": "X", "weight": "W", "stride": 1, "padding": 0},
    )
    assert conv_torch["translated_kwargs"]["input"] == "X"
    assert conv_torch["translated_kwargs"]["weight"] == "W"

    conv_jax = translate_concept_arguments(
        concept="conv2d",
        source_framework="torch",
        target_framework="jax",
        source_kwargs={"input": "X", "weight": "W", "stride": 1, "padding": 0},
    )
    assert conv_jax["translated_kwargs"]["lhs"] == "X"
    assert conv_jax["translated_kwargs"]["rhs"] == "W"

    # 7b. Normalization: torch -> jax, tf, stablehlo
    norm_jax = translate_concept_arguments(
        concept="normalization",
        source_framework="torch",
        target_framework="jax",
        source_kwargs={"weight": "gamma", "bias": "beta", "eps": 1e-5},
    )
    assert norm_jax["translated_kwargs"]["scale"] == "gamma"
    assert norm_jax["translated_kwargs"]["bias"] == "beta"
    assert norm_jax["translated_kwargs"]["epsilon"] == 1e-5

    norm_shlo = translate_concept_arguments(
        concept="layer_norm",
        source_framework="torch",
        target_framework="stablehlo",
        source_kwargs={"weight": "gamma", "bias": "beta", "eps": 1e-5},
    )
    assert norm_shlo["translated_kwargs"]["scale"] == "gamma"
    assert norm_shlo["translated_kwargs"]["offset"] == "beta"
    assert norm_shlo["translated_kwargs"]["epsilon"] == 1e-5

    # 8. Test MCP JSON-RPC protocol message
    mcp_msg = {
        "jsonrpc": "2.0",
        "id": 88,
        "method": "tools/call",
        "params": {
            "name": "translate_concept_arguments",
            "arguments": {
                "concept": "matmul",
                "source_framework": "torch",
                "target_framework": "stablehlo",
                "source_kwargs": {"input": "A", "other": "B"},
            },
        },
    }
    resp = handle_mcp_message(mcp_msg)
    assert resp["id"] == 88
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["translated_kwargs"]["lhs"] == "A"

    # 9. Full branch coverage for unmapped/identity/empty branches
    empty_matmul = translate_concept_arguments("matmul", "torch", "custom_fw", {})
    assert empty_matmul["translated_kwargs"] == {}

    custom_red = translate_concept_arguments(
        "reduce_sum",
        "custom_fw",
        "custom_fw",
        {"dim": 0, "axis": 1, "keepdim": True, "keepdims": False},
    )
    assert custom_red["translated_kwargs"]["dim"] == 0
    assert custom_red["translated_kwargs"]["axis"] == 1
    assert custom_red["translated_kwargs"]["keepdim"] is True
    assert custom_red["translated_kwargs"]["keepdims"] is False

    custom_sm = translate_concept_arguments(
        "softmax", "custom_fw", "custom_fw", {"dim": 0, "axis": 1}
    )
    assert custom_sm["translated_kwargs"]["dim"] == 0
    assert custom_sm["translated_kwargs"]["axis"] == 1

    empty_conv = translate_concept_arguments("conv2d", "torch", "custom_fw", {})
    assert empty_conv["translated_kwargs"] == {}

    # Partial matmul and conv operands for 100% branch coverage
    partial_mm = translate_concept_arguments("matmul", "torch", "torch", {"other": "B"})
    assert "input" not in partial_mm["translated_kwargs"]
    assert partial_mm["translated_kwargs"]["other"] == "B"

    partial_mm2 = translate_concept_arguments(
        "matmul", "torch", "torch", {"input": "A"}
    )
    assert partial_mm2["translated_kwargs"]["input"] == "A"
    assert "other" not in partial_mm2["translated_kwargs"]

    partial_conv = translate_concept_arguments(
        "conv2d", "torch", "torch", {"input": "X"}
    )
    assert partial_conv["translated_kwargs"]["input"] == "X"
    assert "weight" not in partial_conv["translated_kwargs"]

    partial_conv_no_inp = translate_concept_arguments(
        "conv2d", "torch", "torch", {"weight": "W"}
    )
    assert "input" not in partial_conv_no_inp["translated_kwargs"]
    assert partial_conv_no_inp["translated_kwargs"]["weight"] == "W"

    unknown_conv = translate_concept_arguments(
        "conv2d", "torch", "custom_fw", {"input": "X"}
    )
    assert unknown_conv["translated_kwargs"] == {}
    assert unknown_conv["unmapped_kwargs"]["input"] == "X"

    # Completely unrecognized concept falling through all branches
    unknown_concept = translate_concept_arguments(
        "unrecognized_op", "torch", "jax", {"foo": "bar"}
    )
    assert unknown_concept["translated_kwargs"] == {}
    assert unknown_concept["unmapped_kwargs"]["foo"] == "bar"

    # SASS and RDNA modifier branch tests
    from ml_framework_snapshots.mcp_server import (
        check_sass_instruction,
        check_rdna_instruction,
    )

    assert check_sass_instruction("FADD.FTZ", modifiers=[".FTZ"])["is_valid"] is True
    assert (
        check_rdna_instruction("v_add_f32_e32", modifiers=["_e32"])["is_valid"] is True
    )
    assert (
        check_rdna_instruction("v_add_f32_e64", modifiers=["_e64"])["is_valid"] is True
    )
    # Test line 1023: append extracted suffix when modifiers is already provided but does not include suffix
    res_comb = check_rdna_instruction("v_add_f32_e32", modifiers=["_e64"])
    assert res_comb["mnemonic_exists"] is True


def test_check_hallucination_parameter_allowed_values_error(mocker: Any) -> None:
    """Test check_hallucination flags invalid values for parameters with allowed_values.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_snap = {
        "categories": {
            "math": [
                {
                    "name": "custom_norm",
                    "api_path": "torch.custom_norm",
                    "params": [
                        {
                            "name": "mode",
                            "kind": "KEYWORD_ONLY",
                            "allowed_values": ["fast", "precise"],
                        }
                    ],
                }
            ]
        }
    }
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )
    res = check_hallucination(
        "torch",
        "torch.custom_norm",
        kwarg_values={"mode": "invalid_mode"},
    )
    assert res["is_hallucinated"] is True
    assert "Invalid enum value 'invalid_mode' for parameter 'mode'" in res["reason"]

    res_valid = check_hallucination(
        "torch",
        "torch.custom_norm",
        kwarg_values={"mode": "fast"},
    )
    assert res_valid["is_hallucinated"] is False


def test_hallucination_accepted_kwargs_bounded_vs_unconstrained(
    mocker: Any,
) -> None:
    """Verify that bounded accepted_kwargs prevent false-positive hallucination errors and reject illegal kwargs.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_entry = {
        "name": "CustomLayer",
        "api_path": "custom.CustomLayer",
        "kind": "class",
        "params": [
            {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "kwargs", "kind": "VAR_KEYWORD"},
        ],
        "accepted_kwargs": ["dropout_rate", "activation", "use_bias"],
    }
    mock_snap = {"categories": {"layers": [mock_entry]}}
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    )
    mocker.patch(
        "ml_framework_snapshots.mcp_server.get_api_signature",
        return_value=mock_entry,
    )

    # 1. Valid accepted kwarg should NOT be flagged as hallucinated
    valid_res = check_hallucination(
        "custom",
        "custom.CustomLayer",
        kwargs=["dropout_rate", "activation"],
    )
    assert valid_res["is_hallucinated"] is False
    assert len(valid_res["invalid_kwargs"]) == 0
    assert valid_res["has_unconstrained_kwargs"] is False

    # 2. Illegal kwarg not in accepted_kwargs SHOULD be flagged
    invalid_res = check_hallucination(
        "custom",
        "custom.CustomLayer",
        kwargs=["dropout_rate", "hallucinated_param"],
    )
    assert invalid_res["is_hallucinated"] is True
    assert invalid_res["invalid_kwargs"] == ["hallucinated_param"]
