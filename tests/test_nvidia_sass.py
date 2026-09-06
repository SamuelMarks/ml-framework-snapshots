"""Tests for the NVIDIA SASS framework extractor."""

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
