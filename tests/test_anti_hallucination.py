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
    """Test Model Context Protocol (MCP) tool list and tool call request handling."""
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
    """Test get_framework_snapshot uncached extractor, search limit, and varkwargs in check_hallucination."""
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
    """Test positional argument arity checking and strict kwargs in check_hallucination."""
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
    """Test get_framework_snapshot candidate scanning edge cases."""
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
    """Test search_apis when category item does not contain api_path (branch 119 -> 121)."""
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
    """Test strict_kwargs with unrecognized arguments and excess positional arguments."""
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
    """Test check_sass_instruction with modifier normalization and empty operand signatures."""
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
    """Test check_rdna_instruction unsupported arch, matching operands, mismatch, and empty signatures."""
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
    """Test check_mlir_op when operand count differs but operand is variadic (branch 427 -> 432)."""
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


def test_synthetic_hallucination_benchmark_suite() -> None:
    """Benchmark suite verifying 100% detection rate across 100+ synthetic hallucinations."""
    from ml_framework_snapshots.mcp_server import (
        check_hallucination,
        check_sass_instruction,
        check_rdna_instruction,
        check_mlir_op,
    )

    hallucination_cases: List[Tuple[Any, ...]] = []

    # Category 1: Argument swapping (dim vs axis, keepdim vs keepdims) (20 cases)
    swap_ops = [
        ("torch", "torch.sum", ["axis"]),
        ("torch", "torch.mean", ["axis"]),
        ("torch", "torch.max", ["axis"]),
        ("torch", "torch.min", ["axis"]),
        ("torch", "torch.argmax", ["axis"]),
        ("torch", "torch.argmin", ["axis"]),
        ("torch", "torch.cumsum", ["axis"]),
        ("torch", "torch.cumprod", ["axis"]),
        ("torch", "torch.squeeze", ["axis"]),
        ("torch", "torch.unsqueeze", ["axis"]),
        ("torch", "torch.sum", ["keepdims"]),
        ("torch", "torch.mean", ["keepdims"]),
        ("torch", "torch.max", ["keepdims"]),
        ("torch", "torch.min", ["keepdims"]),
        ("torch", "torch.std", ["axis"]),
        ("torch", "torch.var", ["axis"]),
        ("torch", "torch.median", ["axis"]),
        ("torch", "torch.mode", ["axis"]),
        ("torch", "torch.norm", ["axis"]),
        ("torch", "torch.prod", ["axis"]),
    ]
    for fw, api, kw in swap_ops:
        hallucination_cases.append(("api_kwarg", fw, api, kw))

    # Category 2: Non-existent kwargs in factory & math methods (25 cases)
    fake_kwargs = [
        ("torch", "torch.empty", ["non_existent_kw"]),
        ("torch", "torch.zeros", ["invented_flag"]),
        ("torch", "torch.ones", ["fake_allocator"]),
        ("torch", "torch.randn", ["bogus_entropy"]),
        ("torch", "torch.full", ["extra_unused_opt"]),
        ("torch", "torch.arange", ["unsupported_step_kw"]),
        ("torch", "torch.linspace", ["endpoint_illegal"]),
        ("torch", "torch.logspace", ["imaginary_base_kw"]),
        ("torch", "torch.eye", ["column_order_fake"]),
        ("torch", "torch.tensor", ["nonexistent_copy"]),
        ("torch", "torch.as_tensor", ["fake_borrow_kw"]),
        ("torch", "torch.from_numpy", ["zero_copy_kw"]),
        ("torch", "torch.bernoulli", ["fake_p_kw"]),
        ("torch", "torch.multinomial", ["fake_seed_kw"]),
        ("torch", "torch.normal", ["variance_instead_of_std"]),
        ("torch", "torch.poisson", ["rate_kw_typo"]),
        ("torch", "torch.rand", ["random_state_in_torch"]),
        ("torch", "torch.randint", ["exclusive_bound_kw"]),
        ("torch", "torch.randperm", ["algorithm_kw"]),
        ("torch", "torch.empty_like", ["unsupported_kwarg"]),
        ("torch", "torch.zeros_like", ["unsupported_kwarg"]),
        ("torch", "torch.ones_like", ["unsupported_kwarg"]),
        ("torch", "torch.full_like", ["unsupported_kwarg"]),
        ("torch", "torch.cat", ["axis_in_cat"]),
        ("torch", "torch.stack", ["axis_in_stack"]),
    ]
    for fw, api, kw in fake_kwargs:
        hallucination_cases.append(("api_kwarg", fw, api, kw))

    # Category 3: GPU SASS microarchitecture & UR register mismatches (25 cases)
    sass_cases = [
        ("FADD", ["UR0", "UR1", "R0"], "sm_70"),
        ("FSUB", ["UR0", "UR1", "R0"], "sm_70"),
        ("FMUL", ["UR0", "UR1", "R0"], "sm_70"),
        ("FFMA", ["UR0", "UR1", "R0", "R0"], "sm_70"),
        ("IADD3", ["UR0", "UR1", "UR2"], "sm_70"),
        ("IMUL", ["UR0", "UR1"], "sm_70"),
        ("LEA", ["UR0", "UR1", "UR2"], "sm_70"),
        ("LOP3", ["UR0", "UR1", "UR2", "UR3"], "sm_70"),
        ("WGMMA", None, "sm_70"),
        ("WGMMA", None, "sm_75"),
        ("WGMMA", None, "sm_80"),
        ("WGMMA", None, "sm_86"),
        ("WGMMA", None, "sm_89"),
        ("WGMMA_MMA_ASYNC_F16", None, "sm_80"),
        ("WGMMA_MMA_ASYNC_BF16", None, "sm_80"),
        ("WGMMA_MMA_ASYNC_TF32", None, "sm_86"),
        ("WGMMA_MMA_ASYNC_E4M3", None, "sm_89"),
        ("WGMMA_MMA_ASYNC_E5M2", None, "sm_80"),
        ("WGMMA_MMA_ASYNC_FP4", None, "sm_90"),
        ("WGMMA_MMA_ASYNC_FP6", None, "sm_90"),
        ("MMA_SCALE_FP4", None, "sm_90"),
        ("MMA_SCALE_FP6", None, "sm_90"),
        ("MXFP8_MMA", None, "sm_90"),
        ("TMA", None, "sm_70"),
        ("TMA_LOAD", None, "sm_80"),
    ]
    for mnem, ops, arch in sass_cases:
        hallucination_cases.append(("sass", mnem, ops, arch))

    # Category 4: StableHLO / MLIR invalid attributes & type mismatches (20 cases)
    mlir_cases = [
        ("stablehlo.compare", None, {"comparison_direction": "BAD_DIRECTION"}),
        ("stablehlo.compare", None, {"comparison_direction": "EQUALS"}),
        ("stablehlo.compare", None, {"comparison_direction": "LESS_THAN"}),
        ("stablehlo.compare", None, {"precision": "LOW"}),
        ("stablehlo.compare", None, {"precision": "MEDIUM"}),
        ("stablehlo.dot_general", None, {"dot_dimension_numbers": {}}),
        (
            "stablehlo.dot_general",
            None,
            {"dot_dimension_numbers": {"lhs_batch_dimensions": [0]}},
        ),
        ("stablehlo.dot_general", None, {"dot_dimension_numbers": "not_a_dict"}),
        ("stablehlo.convolution", None, {"dimension_numbers": {}}),
        ("stablehlo.convolution", None, {"dimension_numbers": "invalid_string"}),
        ("stablehlo.scatter", None, {"scatter_dimension_numbers": {}}),
        ("stablehlo.gather", None, {"gather_dimension_numbers": {}}),
        ("stablehlo.non_existent_op_1", None, None),
        ("stablehlo.non_existent_op_2", None, None),
        ("stablehlo.non_existent_op_3", None, None),
        ("arith.addf", ["i32", "i32"], None),
        ("arith.addf", ["int", "int"], None),
        ("arith.subf", ["i64", "i64"], None),
        ("math.sin", ["i32"], None),
        ("math.cos", ["i64"], None),
    ]
    for op_name, op_types, struct_attrs in mlir_cases:
        hallucination_cases.append(("mlir", op_name, op_types, struct_attrs))

    # Category 5: AMD RDNA register alignment & microarch mismatches (15 cases)
    rdna_cases = [
        ("v_add_f32", ["v[1:2]", "v0", "v1"], None),
        ("v_add_f32", ["v[3:4]", "v0", "v1"], None),
        ("v_add_f32", ["v[5:6]", "v0", "v1"], None),
        ("v_add_f32", ["s[1:2]", "s0", "s1"], None),
        ("v_add_f32", ["s[3:4]", "s0", "s1"], None),
        ("v_add_f32", ["s[5:6]", "s0", "s1"], None),
        ("v_add_f32", ["a[0:3]", "v0", "v1"], "GFX10/RDNA1"),
        ("v_add_f32", ["a[0:3]", "v0", "v1"], "GFX10.3/RDNA2"),
        ("v_add_f32", ["a[0:3]", "v0", "v1"], "GFX11/RDNA3"),
        ("v_add_f32", ["a[0:3]", "v0", "v1"], "GFX12/RDNA4"),
        ("v_dual_add_f32", None, "GFX9/CDNA"),
        ("v_dual_fmac_f32", None, "GFX9/CDNA"),
        ("v_dual_mul_f32", None, "GFX10/RDNA1"),
        ("v_dual_sub_f32", None, "GFX10.3/RDNA2"),
        ("v_non_existent_rdna_op", None, None),
    ]
    for mnem, ops, garch in rdna_cases:
        hallucination_cases.append(("rdna", mnem, ops, garch))

    assert len(hallucination_cases) >= 105

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

    # Verify 100% detection rate across all 105+ synthetic hallucinations
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
