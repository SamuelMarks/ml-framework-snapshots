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
