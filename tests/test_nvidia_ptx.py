"""Tests for the NVIDIA PTX ISA framework extractor and validator."""

import json
from unittest import mock
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, SemanticTier
from ml_framework_snapshots.frameworks import nvidia_ptx
from ml_framework_snapshots import mcp_server


def test_nvidia_ptx_collect_api_layer() -> None:
    """Test that collect_api returns an empty list for non-UTIL categories."""
    assert nvidia_ptx.collect_api(SemanticTier.LAYER) == []


def test_nvidia_ptx_collect_api_util() -> None:
    """Test that collect_api returns valid GhostRef objects for PTX instructions."""
    refs = nvidia_ptx.collect_api(SemanticTier.UTIL)
    assert len(refs) > 0

    for ref in refs:
        assert isinstance(ref, GhostRef)
        assert ref.kind == "function"
        assert ref.api_path.startswith("nvidia_ptx.inst.")
        assert ref.name in ref.api_path
        assert isinstance(ref.docstring, str)
        assert len(ref.docstring) > 0

        for param in ref.params:
            assert isinstance(param, GhostParam)
            assert param.kind == "POSITIONAL_ONLY"
            assert param.annotation == "PTXOperand"


def test_validate_ptx_type() -> None:
    """Test validating legal and illegal PTX data types."""
    assert nvidia_ptx.validate_ptx_type(".f32") == []
    assert nvidia_ptx.validate_ptx_type("u64") == []
    assert nvidia_ptx.validate_ptx_type(".pred") == []

    errs = nvidia_ptx.validate_ptx_type(".invalid_type")
    assert len(errs) == 1
    assert "Invalid PTX data type" in errs[0]


def test_validate_ptx_state_space() -> None:
    """Test validating legal and illegal PTX memory state spaces."""
    assert nvidia_ptx.validate_ptx_state_space(".global") == []
    assert nvidia_ptx.validate_ptx_state_space("shared") == []
    assert nvidia_ptx.validate_ptx_state_space(".const") == []

    errs = nvidia_ptx.validate_ptx_state_space(".unknown_space")
    assert len(errs) == 1
    assert "Invalid PTX state space" in errs[0]


def test_validate_ptx_register() -> None:
    """Test validating registers, memory dereferences, and special registers."""
    # Special registers
    assert nvidia_ptx.validate_ptx_register("%tid.x") == []
    assert nvidia_ptx.validate_ptx_register("%laneid") == []
    assert nvidia_ptx.validate_ptx_register("%smid") == []
    assert nvidia_ptx.validate_ptx_register("%gridid") == []
    assert nvidia_ptx.validate_ptx_register("%dynamic_smem_size") == []
    assert nvidia_ptx.validate_ptx_register("%clock64") == []

    # Predicate registers
    assert nvidia_ptx.validate_ptx_register("%p0") == []
    assert nvidia_ptx.validate_ptx_register("%p8") == []
    assert nvidia_ptx.validate_ptx_register("%pt") == []
    assert nvidia_ptx.validate_ptx_register("!%pt") == []
    assert nvidia_ptx.validate_ptx_register("%p7") == []
    err_p_oor = nvidia_ptx.validate_ptx_register("%p64")
    assert any("out of bounds" in e for e in err_p_oor)

    # Virtual registers and immediates
    assert nvidia_ptx.validate_ptx_register("%r0") == []
    assert nvidia_ptx.validate_ptx_register("%fd3") == []
    assert nvidia_ptx.validate_ptx_register("42") == []

    # Memory dereferences
    assert nvidia_ptx.validate_ptx_register("[%rd0]") == []
    assert nvidia_ptx.validate_ptx_register("[%r1 + 8]") == []
    err_malformed_mem = nvidia_ptx.validate_ptx_register("[]")
    assert any("Malformed PTX memory dereference" in e for e in err_malformed_mem)


def test_validate_ptx_instruction() -> None:
    """Test complete PTX instruction verification against ISA metadata."""
    # Unknown mnemonic
    err_unknown = nvidia_ptx.validate_ptx_instruction("nonexistent_op")
    assert any("Unknown PTX instruction" in e for e in err_unknown)

    # Valid add
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "add",
            types=[".f32"],
            operands=["%f0", "%f1", "%f2"],
            sm_arch="sm_70",
        )
        == []
    )

    # State space on add (invalid)
    err_space = nvidia_ptx.validate_ptx_instruction("add", state_space=".global")
    assert any("not valid for instruction" in e for e in err_space)

    # Valid ld with state space
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "ld",
            types=[".b32"],
            operands=["%r0", "[%rd1]"],
            state_space=".global",
            sm_arch="sm_50",
        )
        == []
    )
    err_disallowed_space = nvidia_ptx.validate_ptx_instruction(
        "ld",
        state_space=".sreg",
    )
    assert any("not valid for instruction" in e for e in err_disallowed_space)

    # Unsupported type on fma
    err_type = nvidia_ptx.validate_ptx_instruction(
        "fma", types=[".u16"], operands=["%r0", "%r1", "%r2", "%r3"]
    )
    assert any("is not supported for PTX instruction" in e for e in err_type)

    # Operands count mismatch
    err_ops = nvidia_ptx.validate_ptx_instruction("add", operands=["%f0", "%f1"])
    assert any("expects 3 operands, got 2" in e for e in err_ops)

    # SM requirement check (wgmma.mma_async requires sm_90)
    err_sm = nvidia_ptx.validate_ptx_instruction("wgmma.mma_async", sm_arch="sm_80")
    assert any("requires at least sm_90" in e for e in err_sm)


def test_load_exhaustive_ptx_missing_file() -> None:
    """Test graceful handling when ptx JSON dump file does not exist."""
    with mock.patch("os.path.exists", return_value=False):
        assert nvidia_ptx._load_exhaustive_ptx() == []


def test_load_exhaustive_ptx_dict_envelope() -> None:
    """Test loading PTX instructions from dictionary envelopes."""
    mock_envelope = {
        "schema_version": "1.0.0",
        "target": "nvidia_ptx",
        "categories": {
            "arithmetic": [{"mnemonic": "mock_add"}],
            "memory": [{"mnemonic": "mock_ld"}],
            "non_list_cat": "invalid",
        },
    }
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_envelope))
    ):
        with mock.patch("os.path.exists", return_value=True):
            ops = nvidia_ptx._load_exhaustive_ptx()
            assert len(ops) == 2
            assert ops[0]["mnemonic"] == "mock_add"

    mock_inst_dict = {
        "instructions": [{"mnemonic": "mock_op"}],
    }
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_inst_dict))
    ):
        with mock.patch("os.path.exists", return_value=True):
            ops = nvidia_ptx._load_exhaustive_ptx()
            assert len(ops) == 1
            assert ops[0]["mnemonic"] == "mock_op"

    mock_raw_list = [{"mnemonic": "mock_raw_op"}]
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_raw_list))
    ):
        with mock.patch("os.path.exists", return_value=True):
            ops = nvidia_ptx._load_exhaustive_ptx()
            assert len(ops) == 1
            assert ops[0]["mnemonic"] == "mock_raw_op"


def test_check_ptx_instruction_mcp() -> None:
    """Test check_ptx_instruction via MCP server endpoint and call_tool dispatch."""
    res_valid = mcp_server.check_ptx_instruction(
        "add", types=[".f32"], operands=["%f0", "%f1", "%f2"]
    )
    assert res_valid["is_valid"] is True
    assert res_valid["mnemonic_exists"] is True

    res_invalid = mcp_server.check_ptx_instruction("fake_ptx_op")
    assert res_invalid["is_valid"] is False
    assert res_invalid["mnemonic_exists"] is False

    # Test call_tool dispatch
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "check_ptx_instruction",
            "arguments": {
                "mnemonic": "add",
                "types": [".f32"],
                "operands": ["%f0", "%f1", "%f2"],
            },
        },
    }
    resp = mcp_server.handle_mcp_message(msg)
    assert "result" in resp
    assert "content" in resp["result"]


def test_validate_ptx_scope() -> None:
    """Test validating legal and illegal memory scope qualifiers."""
    assert nvidia_ptx.validate_ptx_scope(".gpu") == []
    assert nvidia_ptx.validate_ptx_scope("cta") == []
    assert nvidia_ptx.validate_ptx_scope(".sys") == []
    assert nvidia_ptx.validate_ptx_scope(".cluster") == []

    errs = nvidia_ptx.validate_ptx_scope(".unknown_scope")
    assert len(errs) == 1
    assert "Invalid PTX memory scope" in errs[0]


def test_validate_ptx_vector_width() -> None:
    """Test validating legal and illegal vector width qualifiers."""
    assert nvidia_ptx.validate_ptx_vector_width(".v2") == []
    assert nvidia_ptx.validate_ptx_vector_width("v4") == []

    errs = nvidia_ptx.validate_ptx_vector_width(".v8")
    assert len(errs) == 1
    assert "Invalid PTX vector width" in errs[0]


def test_validate_ptx_instruction_scopes_and_vectors() -> None:
    """Test scope, vector width, and advanced PTX 7.x/8.x instructions."""
    # Scope on atomic operation (valid)
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "atom",
            types=[".u32"],
            operands=["%r0", "[%rd1]", "%r2"],
            state_space=".global",
            scope=".gpu",
        )
        == []
    )

    # Scope on arithmetic add (invalid)
    err_scope = nvidia_ptx.validate_ptx_instruction("add", scope=".cta")
    assert any(
        "only memory and barrier instructions accept scopes" in e for e in err_scope
    )

    # Vector width on add (valid .v2)
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "add",
            types=[".f32"],
            operands=["%f0", "%f1", "%f2"],
            vector_width=".v2",
        )
        == []
    )

    # Vector width on div (unsupported)
    err_vw = nvidia_ptx.validate_ptx_instruction(
        "div",
        types=[".f32"],
        operands=["%f0", "%f1", "%f2"],
        vector_width=".v2",
    )
    assert any("Vector width '.v2' is not supported" in e for e in err_vw)

    # Vector width not in allowed_widths (fma only supports .v2)
    err_vw2 = nvidia_ptx.validate_ptx_instruction(
        "fma",
        types=[".f16"],
        operands=["%f0", "%f1", "%f2", "%f3"],
        vector_width=".v4",
    )
    assert any("Vector width '.v4' is not supported" in e for e in err_vw2)

    # Asynchronous copy (cp.async requires sm_80)
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "cp.async",
            operands=["[%rd0]", "[%rd1]", "16"],
            sm_arch="sm_80",
        )
        == []
    )
    err_cp_sm = nvidia_ptx.validate_ptx_instruction("cp.async", sm_arch="sm_70")
    assert any("requires at least sm_80" in e for e in err_cp_sm)

    # Mbarrier operations
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "mbarrier.init",
            types=[".b64"],
            operands=["[%rd0]", "32"],
            state_space=".shared",
            sm_arch="sm_80",
        )
        == []
    )

    # Matrix operations: ldmatrix (sm_75)
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "ldmatrix",
            types=[".b16"],
            operands=["%r0", "[%rd1]"],
            vector_width=".v4",
            sm_arch="sm_75",
        )
        == []
    )

    # Transcendental tanh (sm_75)
    assert (
        nvidia_ptx.validate_ptx_instruction(
            "tanh",
            types=[".f32"],
            operands=["%f0", "%f1"],
            sm_arch="sm_75",
        )
        == []
    )

    # MCP check_ptx_instruction with scope and vector_width
    res_mcp = mcp_server.check_ptx_instruction(
        "atom",
        types=[".u32"],
        operands=["%r0", "[%rd1]", "%r2"],
        state_space=".global",
        scope=".gpu",
    )
    assert res_mcp["is_valid"] is True
