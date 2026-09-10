"""Tests for the NVIDIA SASS framework extractor."""

import json
from typing import Any
from unittest import mock

from ml_switcheroo_ir.schema.ghost import SemanticTier, GhostRef, GhostParam
from ml_framework_snapshots.frameworks import nvidia_sass


def test_nvidia_sass_collect_api_layer() -> None:
    """Test that collect_api returns an empty list for non-UTIL categories."""
    assert nvidia_sass.collect_api(SemanticTier.LAYER) == []


def test_nvidia_sass_collect_api_util() -> None:
    """Test that collect_api returns valid GhostRef objects for SASS instructions."""
    refs = nvidia_sass.collect_api(SemanticTier.UTIL)

    assert len(refs) > 0, "Expected to find SASS instructions."

    # Check structure of the first few refs
    for ref in refs:
        assert isinstance(ref, GhostRef)
        assert ref.kind == "function"
        assert ref.api_path.startswith("nvidia_sass.inst.")
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


def test_nvidia_sass_specific_instruction() -> None:
    """Test a specific instruction like FADD or FMUL to verify it was extracted."""
    refs = nvidia_sass.collect_api(SemanticTier.UTIL)

    fadd_refs = [r for r in refs if r.name == "FADD"]
    assert len(fadd_refs) == 1, "Expected exactly one FADD instruction."
    fadd = fadd_refs[0]

    assert "NVIDIA SASS FADD instruction" in (fadd.docstring or "")

    # We know FADD typically takes registers/constants, check we extracted some operands
    assert len(fadd.params) > 0
    assert fadd.params[0].standardized_name == "dst"
    assert len(fadd.overloads) > 0
    assert fadd.environment_tags is not None
    assert "cuda" in (fadd.environment_tags or [])
    assert "sm_80" in (fadd.environment_tags or [])
    assert getattr(fadd, "structured_operands", None) is not None


def test_build_structured_sass_operands() -> None:
    """Test building structured operand records with roles, register classes, and immediates."""
    structured = nvidia_sass.build_structured_sass_operands(
        ["R0", "1.5", "0x20", "UR4", "P1", "c[0][4]", "UP0", "B1"], "FADD"
    )
    assert len(structured) == 8
    assert structured[0]["role"] == "dest"
    assert "R" in structured[0]["register_classes"]

    assert structured[1]["immediate_type"] == "float32"
    assert structured[2]["immediate_type"] == "int32"

    assert "UR" in structured[3]["register_classes"]
    assert "P" in structured[4]["register_classes"]
    assert "CBANK" in structured[5]["register_classes"]
    assert "UP" in structured[6]["register_classes"]
    assert "B" in structured[7]["register_classes"]

    # Store, branch, barrier instruction roles
    st_ops = nvidia_sass.build_structured_sass_operands(["[R1]", "R2"], "STG")
    assert st_ops[0]["role"] == "address"

    bra_ops = nvidia_sass.build_structured_sass_operands(["0x100"], "BRA")
    assert bra_ops[0]["role"] == "target"

    bar_ops = nvidia_sass.build_structured_sass_operands(["0"], "BAR")
    assert bar_ops[0]["role"] == "barrier"


def test_parse_sass_modifiers() -> None:
    """Test parsing and normalising SASS instruction modifiers."""
    mods = ["SAT", ".RN", "CUSTOM_MOD", ".SAT"]
    parsed = nvidia_sass.parse_sass_modifiers(mods)
    assert ".SAT" in parsed
    assert ".RN" in parsed
    assert ".CUSTOM_MOD" in parsed


def test_resolve_sm_architectures() -> None:
    """Test resolving SM architecture range strings into explicit SM tag lists."""
    assert nvidia_sass.resolve_sm_architectures("sm_80+") == [
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
    ]
    assert nvidia_sass.resolve_sm_architectures("sm_70+") == [
        "sm_70",
        "sm_75",
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
    ]
    assert nvidia_sass.resolve_sm_architectures("sm_90") == ["sm_90"]
    assert (
        nvidia_sass.resolve_sm_architectures(None) == nvidia_sass.ALL_SM_ARCHITECTURES
    )
    assert (
        nvidia_sass.resolve_sm_architectures("unknown_arch")
        == nvidia_sass.ALL_SM_ARCHITECTURES
    )
    assert (
        nvidia_sass.resolve_sm_architectures("unknown+")
        == nvidia_sass.ALL_SM_ARCHITECTURES
    )


def test_normalize_sass_operand_type() -> None:
    """Test normalising various raw operand representations to standard SASS operand types."""
    assert nvidia_sass.normalize_sass_operand_type("R") == "R"
    assert nvidia_sass.normalize_sass_operand_type("RegOperand") == "R"
    assert nvidia_sass.normalize_sass_operand_type("UR") == "UR"
    assert nvidia_sass.normalize_sass_operand_type("UniformRegOperand") == "UR"
    assert nvidia_sass.normalize_sass_operand_type("P") == "P"
    assert nvidia_sass.normalize_sass_operand_type("PredicateOperand") == "P"
    assert nvidia_sass.normalize_sass_operand_type("I") == "I"
    assert nvidia_sass.normalize_sass_operand_type("IntIMMOperand") == "I"
    assert nvidia_sass.normalize_sass_operand_type("FI") == "FI"
    assert nvidia_sass.normalize_sass_operand_type("FloatIMMOperand") == "FI"
    assert nvidia_sass.normalize_sass_operand_type("ADDR") == "ADDR"
    assert nvidia_sass.normalize_sass_operand_type("AddressOperand") == "ADDR"
    assert (
        nvidia_sass.normalize_sass_operand_type("cx[bank][offset]")
        == "cx[bank][offset]"
    )
    assert (
        nvidia_sass.normalize_sass_operand_type("c[bank][offset]") == "c[bank][offset]"
    )
    assert nvidia_sass.normalize_sass_operand_type("UNKNOWN") == "UNKNOWN"


def test_modern_sass_instructions() -> None:
    """Verify modern SASS instructions (WGMMA, TMA, LDGSTS) and sm_70+ architecture support."""
    refs = nvidia_sass.collect_api(SemanticTier.UTIL)
    op_map = {r.name: r for r in refs}

    assert "WGMMA" in op_map
    assert "sm_90" in op_map["WGMMA"].environment_tags
    assert "sm_100" in op_map["WGMMA"].environment_tags
    assert "sm_80" not in op_map["WGMMA"].environment_tags

    assert "TMA" in op_map
    assert "sm_90" in op_map["TMA"].environment_tags

    assert "LDGSTS" in op_map
    assert "sm_80" in op_map["LDGSTS"].environment_tags

    assert "FADD" in op_map
    assert "sm_70" in op_map["FADD"].environment_tags


def test_get_sass_instruction_family() -> None:
    """Test classification of SASS instructions into families."""
    assert nvidia_sass.get_sass_instruction_family("BAR.SYNC") == "barrier"
    assert nvidia_sass.get_sass_instruction_family("DEPBAR") == "barrier"
    assert nvidia_sass.get_sass_instruction_family("SYNC") == "barrier"
    assert nvidia_sass.get_sass_instruction_family("BRA") == "control"
    assert nvidia_sass.get_sass_instruction_family("RET") == "control"
    assert nvidia_sass.get_sass_instruction_family("EXIT") == "control"
    assert nvidia_sass.get_sass_instruction_family("WARPSYNC") == "control"
    assert nvidia_sass.get_sass_instruction_family("JMP") == "control"
    assert nvidia_sass.get_sass_instruction_family("CALL") == "control"
    assert nvidia_sass.get_sass_instruction_family("BR") == "control"
    assert nvidia_sass.get_sass_instruction_family("HMMA16816") == "tensor"
    assert nvidia_sass.get_sass_instruction_family("IMMA") == "tensor"
    assert nvidia_sass.get_sass_instruction_family("BMMA") == "tensor"
    assert nvidia_sass.get_sass_instruction_family("DMMA") == "tensor"
    assert nvidia_sass.get_sass_instruction_family("WGMMA") == "tensor"
    assert nvidia_sass.get_sass_instruction_family("LDG") == "memory"
    assert nvidia_sass.get_sass_instruction_family("LDS") == "memory"
    assert nvidia_sass.get_sass_instruction_family("LDL") == "memory"
    assert nvidia_sass.get_sass_instruction_family("LDC") == "memory"
    assert nvidia_sass.get_sass_instruction_family("STG") == "memory"
    assert nvidia_sass.get_sass_instruction_family("STS") == "memory"
    assert nvidia_sass.get_sass_instruction_family("STL") == "memory"
    assert nvidia_sass.get_sass_instruction_family("ATOMG") == "memory"
    assert nvidia_sass.get_sass_instruction_family("ATOMS") == "memory"
    assert nvidia_sass.get_sass_instruction_family("RED") == "memory"
    assert nvidia_sass.get_sass_instruction_family("TMA") == "memory"
    assert nvidia_sass.get_sass_instruction_family("LDGSTS") == "memory"
    assert nvidia_sass.get_sass_instruction_family("LDSM") == "memory"
    assert nvidia_sass.get_sass_instruction_family("MUFU") == "special"
    assert nvidia_sass.get_sass_instruction_family("COS") == "special"
    assert nvidia_sass.get_sass_instruction_family("SIN") == "special"
    assert nvidia_sass.get_sass_instruction_family("EX2") == "special"
    assert nvidia_sass.get_sass_instruction_family("LG2") == "special"
    assert nvidia_sass.get_sass_instruction_family("RCP") == "special"
    assert nvidia_sass.get_sass_instruction_family("RSQ") == "special"
    assert nvidia_sass.get_sass_instruction_family("FADD") == "arithmetic"


def test_get_sass_latency_range() -> None:
    """Test retrieval of cycle latency bounds across families and microarchitectures."""
    assert nvidia_sass.get_sass_latency_range("arithmetic", "sm_90") == (3, 5)
    assert nvidia_sass.get_sass_latency_range("arithmetic", "sm_100") == (3, 5)
    assert nvidia_sass.get_sass_latency_range("arithmetic", "sm_80") == (4, 6)
    assert nvidia_sass.get_sass_latency_range("arithmetic", None) == (4, 6)

    assert nvidia_sass.get_sass_latency_range("memory", "sm_90") == (16, 256)
    assert nvidia_sass.get_sass_latency_range("memory", "sm_80") == (19, 256)

    assert nvidia_sass.get_sass_latency_range("tensor", "sm_90") == (16, 32)
    assert nvidia_sass.get_sass_latency_range("tensor", "sm_80") == (8, 16)

    assert nvidia_sass.get_sass_latency_range("special", "sm_80") == (14, 16)
    assert nvidia_sass.get_sass_latency_range("control", "sm_80") == (1, 4)
    assert nvidia_sass.get_sass_latency_range("barrier", "sm_80") == (1, 4)
    assert nvidia_sass.get_sass_latency_range("unknown", "sm_80") == (1, 16)


def test_validate_sass_register() -> None:
    """Test structured SASS register parsing, alignment, ranges, and reuse flags."""
    # 1. Reuse flag on dst vs src
    err_reuse_dst = nvidia_sass.validate_sass_register("R0.reuse", role="dst")
    assert any(
        "Register reuse flag '.reuse' is not permitted" in e for e in err_reuse_dst
    )

    err_reuse_src = nvidia_sass.validate_sass_register("R0.reuse", role="src0")
    assert err_reuse_src == []

    # 2. Memory wrapper
    assert nvidia_sass.validate_sass_register("[R0]", role="src0") == []
    assert nvidia_sass.validate_sass_register("[R1+0x4]", role="src0") == []
    assert nvidia_sass.validate_sass_register("[]", role="src0") == []

    # 3. 64-bit pair
    assert nvidia_sass.validate_sass_register("R0:R1") == []
    assert nvidia_sass.validate_sass_register("R[2:3]") == []
    err_pair_odd = nvidia_sass.validate_sass_register("R1:R2")
    assert any("must start on an even register index" in e for e in err_pair_odd)

    err_pair_oor = nvidia_sass.validate_sass_register("R256:R257")
    assert any("max register index is R255" in e for e in err_pair_oor)

    err_pair_span = nvidia_sass.validate_sass_register("R0:R2")
    assert any("Invalid register span" in e for e in err_pair_span)

    # 4. 128-bit quad
    assert nvidia_sass.validate_sass_register("R0:R3") == []
    assert nvidia_sass.validate_sass_register("R[4:7]") == []
    err_quad_mod = nvidia_sass.validate_sass_register("R2:R5")
    assert any("must start on a modulo-4 register index" in e for e in err_quad_mod)

    # 5. Uniform register pair
    assert nvidia_sass.validate_sass_register("UR0:UR1", sm_arch="sm_80") == []
    assert nvidia_sass.validate_sass_register("UR[2:3]", sm_arch="sm_80") == []
    err_ur_pair_sm70 = nvidia_sass.validate_sass_register("UR0:UR1", sm_arch="sm_70")
    assert any("only supported on sm_75+" in e for e in err_ur_pair_sm70)

    err_ur_pair_odd = nvidia_sass.validate_sass_register("UR1:UR2", sm_arch="sm_80")
    assert any("must start on an even register index" in e for e in err_ur_pair_odd)

    err_ur_pair_oor = nvidia_sass.validate_sass_register("UR64:UR65", sm_arch="sm_80")
    assert any("max is UR63" in e for e in err_ur_pair_oor)

    err_ur_pair_span = nvidia_sass.validate_sass_register("UR0:UR3", sm_arch="sm_80")
    assert any("Invalid uniform register pair span" in e for e in err_ur_pair_span)

    # 6. 32-bit register
    assert nvidia_sass.validate_sass_register("R0") == []
    assert nvidia_sass.validate_sass_register("R255") == []
    assert nvidia_sass.validate_sass_register("RZ") == []
    err_r_oor = nvidia_sass.validate_sass_register("R256")
    assert any("out of range" in e for e in err_r_oor)

    # 7. Uniform register
    assert nvidia_sass.validate_sass_register("UR0", sm_arch="sm_75") == []
    assert nvidia_sass.validate_sass_register("UR63", sm_arch="sm_80") == []
    assert nvidia_sass.validate_sass_register("URZ", sm_arch="sm_80") == []
    assert any(
        "only supported on sm_75+" in e
        for e in nvidia_sass.validate_sass_register("URZ", sm_arch="sm_70")
    )
    assert any(
        "only supported on sm_75+" in e
        for e in nvidia_sass.validate_sass_register("UR0", sm_arch="sm_70")
    )
    assert any(
        "out of range" in e
        for e in nvidia_sass.validate_sass_register("UR64", sm_arch="sm_80")
    )

    # 8. Predicate register
    assert nvidia_sass.validate_sass_register("P0") == []
    assert nvidia_sass.validate_sass_register("P7") == []
    assert nvidia_sass.validate_sass_register("PT") == []
    assert nvidia_sass.validate_sass_register("!PT") == []
    assert nvidia_sass.validate_sass_register("!P0") == []
    assert nvidia_sass.validate_sass_register("@PT") == []
    assert nvidia_sass.validate_sass_register("@!PT") == []
    assert nvidia_sass.validate_sass_register("@P0") == []
    assert any("out of range" in e for e in nvidia_sass.validate_sass_register("P8"))

    # 9. Uniform predicate register
    assert nvidia_sass.validate_sass_register("UP0", sm_arch="sm_80") == []
    assert nvidia_sass.validate_sass_register("UP7", sm_arch="sm_80") == []
    assert nvidia_sass.validate_sass_register("UPT", sm_arch="sm_80") == []
    assert nvidia_sass.validate_sass_register("!UPT", sm_arch="sm_80") == []
    assert any(
        "only supported on sm_75+" in e
        for e in nvidia_sass.validate_sass_register("UP0", sm_arch="sm_70")
    )
    assert any(
        "only supported on sm_75+" in e
        for e in nvidia_sass.validate_sass_register("UPT", sm_arch="sm_70")
    )
    assert any(
        "out of range" in e
        for e in nvidia_sass.validate_sass_register("UP8", sm_arch="sm_80")
    )

    # 10. Barrier register
    assert nvidia_sass.validate_sass_register("B0") == []
    assert nvidia_sass.validate_sass_register("B15") == []
    assert any("out of range" in e for e in nvidia_sass.validate_sass_register("B16"))

    # 11. Generic operand
    assert nvidia_sass.validate_sass_register("c[0][0]") == []


def test_validate_sass_operand_directionality() -> None:
    """Test operand directionality checks for write slot constraints."""
    assert nvidia_sass.validate_sass_operand_directionality([], "FADD") == []
    assert (
        nvidia_sass.validate_sass_operand_directionality(["R0", "R1", "R2"], "FADD")
        == []
    )

    # Store instructions allow non-standard dst
    assert nvidia_sass.validate_sass_operand_directionality(["[R1]", "R0"], "STG") == []
    # Branch and barrier instructions allow non-standard dst
    assert nvidia_sass.validate_sass_operand_directionality(["0x1000"], "BRA") == []
    assert nvidia_sass.validate_sass_operand_directionality(["B0"], "DEPBAR") == []

    # Immediate in arithmetic dst slot 0 is illegal
    err_imm = nvidia_sass.validate_sass_operand_directionality(
        ["42", "R1", "R2"], "FADD"
    )
    assert any("cannot be an immediate" in e for e in err_imm)

    err_fimm = nvidia_sass.validate_sass_operand_directionality(["FI", "R1"], "FADD")
    assert any("cannot be an immediate" in e for e in err_fimm)

    # Constant memory in dst slot 0 is illegal
    err_cmem = nvidia_sass.validate_sass_operand_directionality(
        ["c[0][0]", "R1", "R2"], "FADD"
    )
    assert any("cannot be constant bank memory" in e for e in err_cmem)

    # Multi constant bank operand conflict (hardware port limit)
    err_multi_cbank = nvidia_sass.validate_sass_operand_directionality(
        ["R0", "c[0][0]", "c[1][4]"], "FADD"
    )
    assert any("Hardware resource conflict in 'FADD'" in e for e in err_multi_cbank)


def test_validate_sass_modifiers() -> None:
    """Test modifier validation including conflicts and saturation legality."""
    # Rounding mode conflicts
    assert nvidia_sass.validate_sass_modifiers("FADD", [".RN"]) == []
    err_round = nvidia_sass.validate_sass_modifiers("FADD", [".RN", ".RZ"])
    assert any("Conflicting rounding mode modifiers" in e for e in err_round)

    # Cache policy conflicts
    assert nvidia_sass.validate_sass_modifiers("LDG", [".CG"]) == []
    err_cache = nvidia_sass.validate_sass_modifiers("LDG", [".CG", ".CS"])
    assert any("Conflicting cache policy modifiers" in e for e in err_cache)

    # .SAT legality
    assert nvidia_sass.validate_sass_modifiers("FADD", [".SAT"]) == []
    err_sat_ldg = nvidia_sass.validate_sass_modifiers("LDG", [".SAT"])
    assert any("Saturation modifier '.SAT' is not supported" in e for e in err_sat_ldg)


def test_validate_sass_control_code() -> None:
    """Test control code attribute validation."""
    # Stall count
    assert nvidia_sass.validate_sass_control_code("FADD", {"stall_count": 0}) == []
    assert nvidia_sass.validate_sass_control_code("FADD", {"stall_count": 15}) == []
    assert any(
        "out of valid range" in e
        for e in nvidia_sass.validate_sass_control_code("FADD", {"stall_count": -1})
    )
    assert any(
        "out of valid range" in e
        for e in nvidia_sass.validate_sass_control_code("FADD", {"stall_count": 16})
    )

    # Barrier masks
    assert (
        nvidia_sass.validate_sass_control_code("FADD", {"read_barrier_mask": 0}) == []
    )
    assert (
        nvidia_sass.validate_sass_control_code("FADD", {"read_barrier_mask": 5}) == []
    )
    assert any(
        "out of valid range" in e
        for e in nvidia_sass.validate_sass_control_code(
            "FADD", {"read_barrier_mask": 6}
        )
    )
    assert (
        nvidia_sass.validate_sass_control_code("FADD", {"write_barrier_mask": 0}) == []
    )
    assert any(
        "out of valid range" in e
        for e in nvidia_sass.validate_sass_control_code(
            "FADD", {"write_barrier_mask": -1}
        )
    )

    # Latency ticks
    assert (
        nvidia_sass.validate_sass_control_code(
            "FADD", {"latency_ticks": 4}, sm_arch="sm_80"
        )
        == []
    )
    assert any(
        "out of bounds" in e
        for e in nvidia_sass.validate_sass_control_code(
            "FADD", {"latency_ticks": 20}, sm_arch="sm_80"
        )
    )


def test_check_sass_instruction_extended() -> None:
    """Test check_sass_instruction with alignment, directionality, modifier, and control code checks."""
    from ml_framework_snapshots.mcp_server import check_sass_instruction

    # 1. Valid instruction with structured registers and control codes
    res_valid = check_sass_instruction(
        "FADD",
        operands=["R0", "R1", "R2"],
        modifiers=[".FTZ"],
        sm_arch="sm_80",
        control_codes={"stall_count": 2, "latency_ticks": 5},
    )
    assert res_valid["is_valid"] is True

    # 2. Register alignment violation (R1:R2 pair)
    res_align = check_sass_instruction(
        "FADD",
        operands=["R1:R2", "R4:R5", "R6:R7"],
        sm_arch="sm_80",
    )
    assert res_align["is_valid"] is False
    assert any("must start on an even register index" in e for e in res_align["errors"])

    # 3. Invalid operand directionality (constant bank memory in dst)
    res_dir = check_sass_instruction(
        "FADD",
        operands=["c[0][0]", "R1", "R2"],
        sm_arch="sm_80",
    )
    assert res_dir["is_valid"] is False
    assert any("cannot be constant bank memory" in e for e in res_dir["errors"])

    # 4. Conflicting modifiers
    res_mods = check_sass_instruction(
        "FADD",
        modifiers=[".RN", ".RZ"],
        sm_arch="sm_80",
    )
    assert res_mods["is_valid"] is False
    assert any("Conflicting rounding mode" in e for e in res_mods["errors"])

    # 5. FP4 / microscopic scaling instruction gating on sm_100
    res_fp4_sm90 = check_sass_instruction(
        "BMMA",
        sm_arch="sm_90",
    )
    assert res_fp4_sm90["is_valid"] is True
    res_fp4_explicit = check_sass_instruction(
        "FADD",
        sm_arch="sm_90",
    )
    assert res_fp4_explicit["is_valid"] is True


def test_cli_check_sass(capsys: Any, tmp_path: Any) -> None:
    """Test CLI check-sass subcommand with file, valid mnemonic, and error cases."""
    from ml_framework_snapshots.cli import cmd_check_sass
    import argparse
    import pytest

    # 1. Valid mnemonic
    args_valid = argparse.Namespace(
        mnemonic="FADD",
        operands="R0,R1,R2",
        modifiers=".FTZ",
        sm_arch="sm_80",
        file=None,
    )
    cmd_check_sass(args_valid)
    out_valid = capsys.readouterr().out
    assert "SASS Instruction 'FADD' is valid." in out_valid

    # 2. Invalid mnemonic (fails)
    args_invalid_mnemonic = argparse.Namespace(
        mnemonic="INVALID_OP",
        operands=None,
        modifiers=None,
        sm_arch=None,
        file=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_sass(args_invalid_mnemonic)
    assert exc_info.value.code == 1

    # 3. Missing both mnemonic and file
    args_empty = argparse.Namespace(
        mnemonic=None,
        operands=None,
        modifiers=None,
        sm_arch=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_sass(args_empty)

    # 4. Valid file
    valid_file = tmp_path / "test_valid.sass"
    valid_file.write_text("FADD.FTZ R0, R1, R2;\n", encoding="utf-8")
    args_file_valid = argparse.Namespace(
        file=str(valid_file),
        mnemonic=None,
        operands=None,
        modifiers=None,
        sm_arch="sm_80",
    )
    cmd_check_sass(args_file_valid)
    out_file = capsys.readouterr().out
    assert "SASS snippet verified compliant" in out_file

    # 5. Non-existent file
    args_file_missing = argparse.Namespace(
        file=str(tmp_path / "nonexistent.sass"),
        mnemonic=None,
        operands=None,
        modifiers=None,
        sm_arch=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_sass(args_file_missing)

    # 6. Invalid file (contains illegal instruction)
    invalid_file = tmp_path / "test_invalid.sass"
    invalid_file.write_text("WGMMA R0, R1, R2, R3;\n", encoding="utf-8")
    args_file_invalid = argparse.Namespace(
        file=str(invalid_file),
        mnemonic=None,
        operands=None,
        modifiers=None,
        sm_arch="sm_80",
    )
    with pytest.raises(SystemExit):
        cmd_check_sass(args_file_invalid)


def test_validate_sass_control_code_extended() -> None:
    """Test extended SASS control code rules for Hopper/Blackwell, register reuse, and dual-issue."""
    # Hopper barrier scoreboard (mask 7 valid on sm_90, invalid on sm_80)
    assert (
        nvidia_sass.validate_sass_control_code(
            "FADD", {"read_barrier_mask": 7}, sm_arch="sm_90"
        )
        == []
    )
    err_b_sm80 = nvidia_sass.validate_sass_control_code(
        "FADD", {"read_barrier_mask": 7}, sm_arch="sm_80"
    )
    assert any("out of valid range (0-5)" in e for e in err_b_sm80)

    # Register reuse cache flags (.reuse)
    assert (
        nvidia_sass.validate_sass_control_code(
            "FADD", {"register_reuse_flags": ["R0.reuse"]}, sm_arch="sm_80"
        )
        == []
    )
    err_rf_malformed = nvidia_sass.validate_sass_control_code(
        "FADD", {"register_reuse_flags": ["R0"]}, sm_arch="sm_80"
    )
    assert any("Malformed register reuse flag" in e for e in err_rf_malformed)

    err_rf_reg = nvidia_sass.validate_sass_control_code(
        "FADD", {"register_reuse_flags": ["R300.reuse"]}, sm_arch="sm_80"
    )
    assert any("out of range" in e for e in err_rf_reg)

    # Dual-issue restrictions on TENSOR and BARRIER
    err_di_tensor = nvidia_sass.validate_sass_control_code(
        "HMMA16816", {"dual_issue": True}
    )
    assert any("does not support dual-issue" in e for e in err_di_tensor)

    err_di_barrier = nvidia_sass.validate_sass_control_code("BAR", {"dual_issue": True})
    assert any("does not support dual-issue" in e for e in err_di_barrier)

    # Hopper/Blackwell WGMMA and TMA checks
    assert nvidia_sass.validate_sass_control_code("WGMMA", {}, sm_arch="sm_90") == []
    err_wgmma_sm80 = nvidia_sass.validate_sass_control_code(
        "WGMMA", {}, sm_arch="sm_80"
    )
    assert any("requires Hopper or Blackwell" in e for e in err_wgmma_sm80)

    err_tma_sm80 = nvidia_sass.validate_sass_control_code(
        "TMA_LOAD", {}, sm_arch="sm_80"
    )
    assert any("requires Hopper or Blackwell" in e for e in err_tma_sm80)

    # Branches: non-list reuse_flags, dual-issue allowed on FADD, and WGMMA without sm_arch
    assert (
        nvidia_sass.validate_sass_control_code(
            "FADD", {"register_reuse_flags": "not_a_list"}
        )
        == []
    )


def test_load_exhaustive_sass_variants() -> None:
    """Test loading NVIDIA SASS instructions from envelope, dict, and list sources."""
    mock_envelope = {
        "schema_version": "1.0.0",
        "target": "nvidia_sass",
        "categories": {
            "UTIL": [{"mnemonic": "FADD"}],
            "non_list": "invalid",
        },
    }
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=json.dumps(mock_envelope))
        ):
            ops = nvidia_sass._load_exhaustive_sass()
            assert len(ops) == 1
            assert ops[0]["mnemonic"] == "FADD"

        mock_inst_dict = {
            "instructions": [{"mnemonic": "FMUL"}],
        }
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=json.dumps(mock_inst_dict))
        ):
            ops = nvidia_sass._load_exhaustive_sass()
            assert len(ops) == 1
            assert ops[0]["mnemonic"] == "FMUL"

        mock_raw_list = [{"mnemonic": "FFMA"}]
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=json.dumps(mock_raw_list))
        ):
            ops = nvidia_sass._load_exhaustive_sass()
            assert len(ops) == 1
            assert ops[0]["mnemonic"] == "FFMA"

    assert nvidia_sass.validate_sass_control_code("FADD", {"dual_issue": True}) == []
    assert nvidia_sass.validate_sass_control_code("WGMMA", {}) == []


def test_tokenize_sass_line_and_code_block() -> None:
    """Test tokenize_sass_line and check_code_block validation with real SASS syntax."""
    from ml_framework_snapshots.mcp_server import (
        check_code_block,
        check_sass_instruction,
    )

    # Empty line
    assert nvidia_sass.tokenize_sass_line("   ") == (None, "", [], [])

    # Dotted modifiers and predicates
    pred, mnem, mods, ops = nvidia_sass.tokenize_sass_line(
        "@P0 FADD.FTZ.RN R0, R1, R2;"
    )
    assert pred == "@P0"
    assert mnem == "FADD"
    assert mods == [".FTZ", ".RN"]
    assert ops == ["R0", "R1", "R2"]

    # Direct check_sass_instruction with dot-separated mnemonic
    res_direct = check_sass_instruction(
        "FADD.FTZ.RN", operands=["R0", "R1", "R2"], sm_arch="sm_80"
    )
    assert res_direct["is_valid"] is True
    assert res_direct["mnemonic_exists"] is True

    # check_code_block with SASS
    sass_code = """
    // Vector add in SASS
    @P0 FADD.FTZ.RN R0, R1, R2;
    FFMA.SAT R0, R1, R2, R3;
    """
    block_res = check_code_block(sass_code, framework="nvidia_sass", sm_arch="sm_80")
    assert block_res["is_valid"] is True
    assert block_res["hallucinations_detected"] == 0
    assert block_res["total_analyzed"] == 2

    # Hallucinated instruction in code block
    bad_code = "NONEXISTENT_SASS_OP.RN R0, R1;"
    bad_res = check_code_block(bad_code, framework="nvidia_sass")
    assert bad_res["is_valid"] is False
    assert bad_res["hallucinations_detected"] == 1
