"""Tests for the AMD RDNA framework extractor."""

from unittest import mock
from ml_switcheroo_ir.schema.ghost import SemanticTier, GhostRef, GhostParam
from ml_framework_snapshots.frameworks import amd_rdna


def test_amd_rdna_collect_api_layer() -> None:
    """Test that collect_api returns an empty list for non-UTIL categories."""
    assert amd_rdna.collect_api(SemanticTier.LAYER) == []


@mock.patch("ml_framework_snapshots.frameworks.amd_rdna._load_exhaustive_rdna")
def test_amd_rdna_collect_api_util(mock_load: mock.MagicMock) -> None:
    """Test that collect_api returns valid GhostRef objects for RDNA instructions."""
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
    """Test a specific instruction to verify it was extracted."""
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
