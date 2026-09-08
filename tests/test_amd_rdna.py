"""Tests for the AMD RDNA framework extractor."""

import json
from typing import Any
from unittest import mock
from ml_switcheroo_ir.schema.ghost import SemanticTier, GhostRef, GhostParam
from ml_framework_snapshots.frameworks import amd_rdna


def test_amd_rdna_collect_api_layer() -> None:
    """Test that collect_api returns an empty list for non-UTIL categories."""
    assert amd_rdna.collect_api(SemanticTier.LAYER) == []


@mock.patch("ml_framework_snapshots.frameworks.amd_rdna._load_exhaustive_rdna")
def test_amd_rdna_collect_api_util(mock_load: mock.MagicMock) -> None:
    """Test that collect_api returns valid GhostRef objects for RDNA instructions.

    Args:
        mock_load: Mock for _load_exhaustive_rdna.
    """
    # Mock data to test all branches:
    # 1. No modifiers
    # 2. Multiple signatures where one is shorter
    # 3. Condition code (VCC/SCC) and execution mask (EXEC)
    mock_load.return_value = [
        {
            "mnemonic": "v_nop",
            "modifiers": [],
            "operands": [["vGPR"], ["vGPR", "vGPR"]],  # Longer, will update max_sig
            "description": "No op",
        },
        {
            "mnemonic": "v_add_f32",
            "modifiers": ["_e32", "_e64"],
            "operands": [
                ["vGPR", "vGPR", "vGPR"],
                ["vGPR", "vGPR"],  # Shorter, tests branch
            ],
        },
        {
            "mnemonic": "v_cmp_eq_f32",
            "modifiers": ["_e32"],
            "operands": [
                ["VCC", "VGPR", "VGPR"],
                ["SCC", "SGPR", "SGPR"],
                ["EXEC", "simm16"],
            ],
            "description": "Comparison instruction.",
            "encoding": "VOPC",
            "architecture": "GFX11/RDNA3",
        },
        {
            "mnemonic": "s_cbranch_execz",
            "modifiers": [],
            "operands": [["EXEC", "simm16"]],
            "description": "Branch if exec zero.",
            "encoding": "SOPP",
            "architecture": "GFX10+",
        },
    ]

    refs = amd_rdna.collect_api(SemanticTier.UTIL)

    assert len(refs) == 4, "Expected to find 4 RDNA instructions."

    # Check structure of the refs
    for ref in refs:
        assert isinstance(ref, GhostRef)
        assert ref.kind == "function"
        assert ref.api_path.startswith("amd_rdna.inst.")
        assert ref.name in ref.api_path

        # Docstring should be present
        assert isinstance(ref.docstring, str)
        assert len(ref.docstring) > 0

        # Params should be structural positional_only
        for param in ref.params:
            assert isinstance(param, GhostParam)
            assert param.kind == "POSITIONAL_ONLY"
            assert param.name.startswith("op")
            assert param.annotation is not None

    nop = next(r for r in refs if r.name == "v_nop")
    assert len(nop.params) == 2  # max_sig length

    add = next(r for r in refs if r.name == "v_add_f32")
    assert len(add.params) == 3

    cmp_inst = next(r for r in refs if r.name == "v_cmp_eq_f32")
    assert cmp_inst.params[0].standardized_name == "condition"
    assert "rocm" in (cmp_inst.environment_tags or [])
    assert "GFX11/RDNA3" in (cmp_inst.environment_tags or [])


@mock.patch("ml_framework_snapshots.frameworks.amd_rdna._load_exhaustive_rdna")
def test_amd_rdna_specific_instruction(mock_load: mock.MagicMock) -> None:
    """Test a specific instruction to verify it was extracted.

    Args:
        mock_load: Mock for _load_exhaustive_rdna.
    """
    mock_load.return_value = [
        {
            "mnemonic": "v_add_f32",
            "modifiers": ["_e32"],
            "operands": [["vGPR", "vGPR", "vGPR"]],
            "description": "AMD RDNA v_add_f32 instruction.",
        }
    ]
    refs = amd_rdna.collect_api(SemanticTier.UTIL)

    add_refs = [r for r in refs if "v_add" in r.name]
    assert len(add_refs) > 0, "Expected to find at least one v_add instruction."
    add_inst = add_refs[0]

    assert "AMD RDNA v_add" in (add_inst.docstring or "")
    assert len(add_inst.params) == 3
    assert add_inst.params[0].standardized_name == "dst"
    assert len(add_inst.overloads) == 1


def test_amd_rdna_load_exhaustive() -> None:
    """Test that the exhaustive JSON dump can be loaded."""
    data = amd_rdna._load_exhaustive_rdna()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "mnemonic" in data[0]


def test_resolve_gfx_architectures() -> None:
    """Test resolving GFX architecture specifications to target lists."""
    assert amd_rdna.resolve_gfx_architectures(None) == [
        "GFX10/RDNA1",
        "GFX10.3/RDNA2",
        "GFX11/RDNA3",
        "GFX11.5",
        "GFX12/RDNA4",
    ]
    assert amd_rdna.resolve_gfx_architectures("GFX9/CDNA") == ["GFX9/CDNA"]
    assert amd_rdna.resolve_gfx_architectures("GFX10/RDNA1") == ["GFX10/RDNA1"]
    assert amd_rdna.resolve_gfx_architectures("GFX10.3/RDNA2") == ["GFX10.3/RDNA2"]
    assert amd_rdna.resolve_gfx_architectures("GFX11/RDNA3") == ["GFX11/RDNA3"]
    assert amd_rdna.resolve_gfx_architectures("GFX11.5") == ["GFX11.5"]
    assert amd_rdna.resolve_gfx_architectures("GFX12/RDNA4") == ["GFX12/RDNA4"]
    assert amd_rdna.resolve_gfx_architectures("GFX11_TARGETS") == [
        "GFX11/RDNA3",
        "GFX11.5",
        "GFX12/RDNA4",
    ]
    assert amd_rdna.resolve_gfx_architectures("GFX12_TARGETS") == ["GFX12/RDNA4"]
    assert amd_rdna.resolve_gfx_architectures("GFX10+") == [
        "GFX10/RDNA1",
        "GFX10.3/RDNA2",
        "GFX11/RDNA3",
        "GFX11.5",
        "GFX12/RDNA4",
    ]


def test_validate_rdna_register() -> None:
    """Test register alignment, bounds, and CDNA accumulator rules."""
    # 1. 64-bit pair even starting alignment
    assert amd_rdna.validate_rdna_register("v[0:1]") == []
    assert amd_rdna.validate_rdna_register("s[2:3]") == []
    err_pair_odd = amd_rdna.validate_rdna_register("v[1:2]")
    assert any("must start on an even register index" in e for e in err_pair_odd)

    err_s_pair_odd = amd_rdna.validate_rdna_register("s[3:4]")
    assert any("must start on an even register index" in e for e in err_s_pair_odd)

    # 2. 128-bit quad modulo-4 starting alignment
    assert amd_rdna.validate_rdna_register("v[0:3]") == []
    assert amd_rdna.validate_rdna_register("s[4:7]") == []
    assert amd_rdna.validate_rdna_register("a[0:3]", gfx_arch="GFX9/CDNA") == []
    err_quad_mod = amd_rdna.validate_rdna_register("v[2:5]")
    assert any("must start on a modulo-4 register index" in e for e in err_quad_mod)

    # 3. Out of range register indices
    err_v_oor = amd_rdna.validate_rdna_register("v256")
    assert any("out of range" in e for e in err_v_oor)
    err_s_oor = amd_rdna.validate_rdna_register("s106")
    assert any("out of range" in e for e in err_s_oor)
    err_a_oor = amd_rdna.validate_rdna_register("a256", gfx_arch="GFX9/CDNA")
    assert any("out of range" in e for e in err_a_oor)
    err_slice_oor = amd_rdna.validate_rdna_register("v[256:257]")
    assert any("out of range" in e for e in err_slice_oor)
    err_span = amd_rdna.validate_rdna_register("v[0:2]")
    assert any("Invalid register span" in e for e in err_span)

    # 4. CDNA matrix accumulators on non-CDNA architectures
    assert amd_rdna.validate_rdna_register("a0", gfx_arch="GFX9/CDNA") == []
    err_a_rdna = amd_rdna.validate_rdna_register("a0", gfx_arch="GFX11/RDNA3")
    assert any("only supported on GFX9/CDNA" in e for e in err_a_rdna)

    # 5. Memory wrappers
    assert amd_rdna.validate_rdna_register("[v0, s[0:1]]") == []

    # 6. Valid individual 32-bit registers
    assert amd_rdna.validate_rdna_register("v0") == []
    assert amd_rdna.validate_rdna_register("s0") == []
    assert amd_rdna.validate_rdna_register("imm32") == []


def test_validate_rdna_constant_bus() -> None:
    """Test constant bus limitation on pre-RDNA architectures."""
    # Pre-RDNA (GFX9/CDNA) vector instructions cannot read > 1 SGPR/literal
    err_cbus = amd_rdna.validate_rdna_constant_bus(
        ["v0", "s0", "s1"], encoding="VOP2", gfx_arch="GFX9/CDNA"
    )
    assert any("Constant bus limitation violation" in e for e in err_cbus)

    err_cbus_imm = amd_rdna.validate_rdna_constant_bus(
        ["v0", "s0", "42"], encoding="VOP2", gfx_arch="GFX9/CDNA"
    )
    assert any("Constant bus limitation violation" in e for e in err_cbus_imm)

    # Valid: 1 SGPR + 1 VGPR
    assert (
        amd_rdna.validate_rdna_constant_bus(
            ["v0", "s0", "v1"], encoding="VOP2", gfx_arch="GFX9/CDNA"
        )
        == []
    )

    # VOPC test (starts from index 2)
    err_vopc = amd_rdna.validate_rdna_constant_bus(
        ["vcc", "v0", "s0", "s1"], encoding="VOPC", gfx_arch="GFX9/CDNA"
    )
    assert any("Constant bus limitation violation" in e for e in err_vopc)

    # On RDNA3 (GFX11/RDNA3), constant bus limitation does not apply
    assert (
        amd_rdna.validate_rdna_constant_bus(
            ["v0", "s0", "s1"], encoding="VOP2", gfx_arch="GFX11/RDNA3"
        )
        == []
    )

    # Empty or single operand
    assert amd_rdna.validate_rdna_constant_bus([]) == []
    assert amd_rdna.validate_rdna_constant_bus(["v0"]) == []


def test_validate_rdna_modifiers() -> None:
    """Test source and output modifier legality checks."""
    # Valid source modifiers on VOP3/VOP3P
    assert amd_rdna.validate_rdna_modifiers(["-src", "clamp"], encoding="VOP3") == []
    assert (
        amd_rdna.validate_rdna_modifiers(["op_sel", "neg_lo"], encoding="VOP3P") == []
    )

    # Invalid source modifier on SOP1
    err_sop = amd_rdna.validate_rdna_modifiers(["clamp"], encoding="SOP1")
    assert any(
        "Source modifier 'clamp' is only valid on VOP3/VOP3P" in e for e in err_sop
    )

    # Valid output modifier on VOP3
    assert amd_rdna.validate_rdna_modifiers(["omod:2"], encoding="VOP3") == []

    # Invalid output modifier on VOP2
    err_vop2 = amd_rdna.validate_rdna_modifiers(["omod:4"], encoding="VOP2")
    assert any("Output modifier 'omod:4' is only valid on VOP3" in e for e in err_vop2)


def test_validate_rdna_microarchitecture() -> None:
    """Test microarchitecture and wave-size execution constraints."""
    # Dual-issue on GFX11/GFX12
    assert (
        amd_rdna.validate_rdna_microarchitecture(
            "v_dual_fma_f32", gfx_arch="GFX11/RDNA3"
        )
        == []
    )
    err_dual_gfx9 = amd_rdna.validate_rdna_microarchitecture(
        "v_dual_fma_f32", gfx_arch="GFX9/CDNA"
    )
    assert any(
        "Dual-issue instruction 'v_dual_fma_f32' is strictly supported on GFX11+" in e
        for e in err_dual_gfx9
    )

    err_dual_wave64 = amd_rdna.validate_rdna_microarchitecture(
        "v_dual_fma_f32", gfx_arch="GFX11/RDNA3", wave_size=64
    )
    assert any("strictly requires Wave32 execution mode" in e for e in err_dual_wave64)

    # Matrix accumulator on CDNA vs RDNA
    assert (
        amd_rdna.validate_rdna_microarchitecture(
            "v_mfma_f32_16x16x16f16", gfx_arch="GFX9/CDNA"
        )
        == []
    )
    err_mfma = amd_rdna.validate_rdna_microarchitecture(
        "v_mfma_f32_16x16x16f16", gfx_arch="GFX11/RDNA3"
    )
    assert any("strictly supported on GFX9/CDNA" in e for e in err_mfma)

    # CDNA wave-size check: CDNA is Wave64
    err_cdna_w32 = amd_rdna.validate_rdna_microarchitecture(
        "v_add_f32", gfx_arch="GFX9/CDNA", wave_size=32
    )
    assert any(
        "executes in Wave64 mode and does not support Wave32" in e for e in err_cdna_w32
    )


def test_check_rdna_instruction_extended() -> None:
    """Test check_rdna_instruction with modifiers, wave_size, and constant bus validation."""
    from ml_framework_snapshots.mcp_server import check_rdna_instruction

    # 1. Valid instruction
    res_valid = check_rdna_instruction(
        "v_add_f32",
        operands=["v0", "v1", "v2"],
        encoding="VOP2",
        gfx_arch="GFX11/RDNA3",
        wave_size=32,
    )
    assert res_valid["is_valid"] is True

    # 2. Constant bus error on CDNA
    res_cbus = check_rdna_instruction(
        "v_add_f32",
        operands=["v0", "s0", "s1"],
        encoding="VOP2",
        gfx_arch="GFX9/CDNA",
    )
    assert res_cbus["is_valid"] is False
    assert any("Constant bus limitation violation" in e for e in res_cbus["errors"])

    # 3. Modifier error on VOP2
    res_mod = check_rdna_instruction(
        "v_add_f32",
        modifiers=["omod:2"],
        encoding="VOP2",
        gfx_arch="GFX11/RDNA3",
    )
    assert res_mod["is_valid"] is False
    assert any(
        "Output modifier 'omod:2' is only valid on VOP3" in e for e in res_mod["errors"]
    )


def test_cli_check_rdna(capsys: Any, tmp_path: Any) -> None:
    """Test CLI check-rdna subcommand with file, mnemonic, and errors."""
    from ml_framework_snapshots.cli import cmd_check_rdna
    import argparse
    import pytest

    # 1. Valid mnemonic
    args_valid = argparse.Namespace(
        mnemonic="v_add_f32",
        operands="v0,v1,v2",
        encoding="VOP2",
        gfx_arch="GFX11/RDNA3",
        modifiers=None,
        wave_size=32,
        file=None,
    )
    cmd_check_rdna(args_valid)
    out_valid = capsys.readouterr().out
    assert "RDNA Instruction 'v_add_f32' is valid." in out_valid

    # 2. Invalid mnemonic
    args_invalid = argparse.Namespace(
        mnemonic="invalid_op_rdna",
        operands=None,
        encoding=None,
        gfx_arch=None,
        modifiers=None,
        wave_size=None,
        file=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_rdna(args_invalid)
    assert exc_info.value.code == 1

    # 3. Missing arguments
    args_empty = argparse.Namespace(
        mnemonic=None,
        operands=None,
        encoding=None,
        gfx_arch=None,
        modifiers=None,
        wave_size=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_rdna(args_empty)

    # 4. Valid file
    valid_file = tmp_path / "test_valid.s"
    valid_file.write_text("v_add_f32 v0, v1, v2\n", encoding="utf-8")
    args_file_valid = argparse.Namespace(
        file=str(valid_file),
        mnemonic=None,
        operands=None,
        encoding=None,
        gfx_arch="GFX11/RDNA3",
        modifiers=None,
        wave_size=None,
    )
    cmd_check_rdna(args_file_valid)
    out_file = capsys.readouterr().out
    assert "RDNA snippet verified compliant" in out_file

    # 5. Non-existent file
    args_file_missing = argparse.Namespace(
        file=str(tmp_path / "nonexistent.s"),
        mnemonic=None,
        operands=None,
        encoding=None,
        gfx_arch=None,
        modifiers=None,
        wave_size=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_rdna(args_file_missing)

    # 6. Invalid file (contains error)
    invalid_file = tmp_path / "test_invalid.s"
    invalid_file.write_text("v_dual_fma_f32 v0, v1, v2, v3\n", encoding="utf-8")
    args_file_invalid = argparse.Namespace(
        file=str(invalid_file),
        mnemonic=None,
        operands=None,
        encoding=None,
        gfx_arch="GFX9/CDNA",
        modifiers=None,
        wave_size=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_rdna(args_file_invalid)


def test_validate_rdna_register_extended_alignment() -> None:
    """Test 256-bit and 512-bit register alignment rules for CDNA matrix accumulators."""
    # Modulo-8 (256-bit oct)
    assert amd_rdna.validate_rdna_register("a[0:7]", gfx_arch="GFX9/CDNA") == []
    assert amd_rdna.validate_rdna_register("a[8:15]", gfx_arch="GFX9/CDNA") == []
    err_mod8 = amd_rdna.validate_rdna_register("a[2:9]", gfx_arch="GFX9/CDNA")
    assert any("modulo-8" in e for e in err_mod8)

    # Modulo-16 (512-bit hex)
    assert amd_rdna.validate_rdna_register("a[0:15]", gfx_arch="GFX9/CDNA") == []
    assert amd_rdna.validate_rdna_register("a[16:31]", gfx_arch="GFX9/CDNA") == []
    err_mod16 = amd_rdna.validate_rdna_register("a[4:19]", gfx_arch="GFX9/CDNA")
    assert any("modulo-16" in e for e in err_mod16)


def test_validate_vopd_pairing() -> None:
    """Test VOPD dual-issue instruction pairing rules and restrictions."""
    opx_valid = {"mnemonic": "v_dual_fmac_f32", "dst": "v0"}
    opy_valid = {"mnemonic": "v_dual_fma_f32", "dst": "v1"}

    # Valid pairing on GFX11
    assert (
        amd_rdna.validate_vopd_pairing(opx_valid, opy_valid, gfx_arch="GFX11/RDNA3")
        == []
    )

    # Invalid architecture
    err_arch = amd_rdna.validate_vopd_pairing(
        opx_valid, opy_valid, gfx_arch="GFX9/CDNA"
    )
    assert any("only supported on GFX11/RDNA3" in e for e in err_arch)

    # Invalid opcodes
    err_op = amd_rdna.validate_vopd_pairing(
        {"mnemonic": "v_add_f32"}, {"mnemonic": "v_mul_f32"}, gfx_arch="GFX11/RDNA3"
    )
    assert any("not a valid VOPD OpX opcode" in e for e in err_op)
    assert any("not a valid VOPD OpY opcode" in e for e in err_op)

    # Same destination register conflict
    opx_conflict = {"mnemonic": "v_dual_fmac_f32", "dst": "v5"}
    opy_conflict = {"mnemonic": "v_dual_add_f32", "dst": "v5"}
    err_dst = amd_rdna.validate_vopd_pairing(
        opx_conflict, opy_conflict, gfx_arch="GFX11/RDNA3"
    )
    assert any("destination conflict" in e for e in err_dst)


def test_load_exhaustive_rdna_variants() -> None:
    """Test loading AMD RDNA instructions from envelope, dict, and list sources."""
    mock_envelope = {
        "schema_version": "1.0.0",
        "target": "amd_rdna",
        "categories": {
            "UTIL": [{"mnemonic": "v_test_env"}],
            "non_list": "invalid",
        },
    }
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_envelope))
    ):
        ops = amd_rdna._load_exhaustive_rdna()
        assert len(ops) == 1
        assert ops[0]["mnemonic"] == "v_test_env"

    mock_inst_dict = {
        "instructions": [{"mnemonic": "v_test_dict"}],
    }
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_inst_dict))
    ):
        ops = amd_rdna._load_exhaustive_rdna()
        assert len(ops) == 1
        assert ops[0]["mnemonic"] == "v_test_dict"

    mock_raw_list = [{"mnemonic": "v_test_list"}]
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_raw_list))
    ):
        ops = amd_rdna._load_exhaustive_rdna()
        assert len(ops) == 1
        assert ops[0]["mnemonic"] == "v_test_list"
