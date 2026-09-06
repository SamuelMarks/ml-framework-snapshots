"""Tests for the AMD RDNA scraper script."""

import json
import os
import tempfile
import urllib.request
from unittest import mock
import typing

from ml_framework_snapshots.tools import scrape_amd_rdna


def test_fetch_td_file_success() -> None:
    """Test fetch_td_file on success."""
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = mock.MagicMock()
        mock_response.read.return_value = b'defm V_ADD_F32 : VOP1Inst <"v_add_f32"'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        content = scrape_amd_rdna.fetch_td_file("test.td")
        assert "defm V_ADD_F32" in content


def test_fetch_td_file_error() -> None:
    """Test fetch_td_file on error."""
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("Not Found")
        content = scrape_amd_rdna.fetch_td_file("test.td")
        assert content == ""


def test_scrape_amd_rdna_main() -> None:
    """Test the AMD RDNA scrape script with mocked fetches."""
    mock_td_content = """
defm V_ADD_F32 : VOP2Inst <"v_add_f32", VOP_F32_F32_F32>;
defm V_ADD_F32_DUP : VOP2Inst <"v_add_f32", VOP_F32_F32_F32>;
defm V_SUB_F32 : VOP2Inst <"v_sub_f32", VOP_F32_F32_F32>;
def S_MOV_B32 : SOP1_32 <"s_mov_b32">;
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "amd_rdna_exhaustive.json")

        with mock.patch(
            "ml_framework_snapshots.tools.scrape_amd_rdna.fetch_td_file"
        ) as mock_fetch:

            def fetch_side_effect(filename: str) -> str:
                """Mock fetch_td_file side effect.

                Args:
                    filename: The filename to fetch.

                Returns:
                    The mocked content.
                """
                if filename == "VOP1Instructions.td":
                    return mock_td_content
                return ""

            mock_fetch.side_effect = fetch_side_effect

            with mock.patch(
                "ml_framework_snapshots.tools.scrape_amd_rdna.open"
            ) as mock_open:
                original_open = open

                def open_side_effect(
                    path: str, *args: typing.Any, **kwargs: typing.Any
                ) -> typing.Any:
                    """Mock open side effect.

                    Args:
                        path: The file path.
                        *args: Additional args.
                        **kwargs: Additional kwargs.

                    Returns:
                        The mocked file object.
                    """
                    if "amd_rdna_exhaustive.json" in path:
                        return original_open(output_path, *args, **kwargs)
                    return original_open(path, *args, **kwargs)  # pragma: no cover

                mock_open.side_effect = open_side_effect

                # Run main
                scrape_amd_rdna.main()

        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            result_data = json.load(f)

        assert len(result_data) == 3

        v_add_entry = next(e for e in result_data if e["mnemonic"] == "v_add_f32")
        assert v_add_entry["architecture"] == "GFX10+"
        assert "_e32" in v_add_entry["modifiers"]
        assert ["VGPR", "VGPR", "VGPR"] in v_add_entry["operands"]

        s_mov_entry = next(e for e in result_data if e["mnemonic"] == "s_mov_b32")
        assert s_mov_entry["architecture"] == "GFX10+"
        assert ["SGPR", "SGPR"] in s_mov_entry["operands"]


def test_resolve_td_instruction_info_profiles() -> None:
    """Test resolving instruction definitions across all major TableGen classes and architectures."""
    # VOP1
    vop1 = scrape_amd_rdna.resolve_td_instruction_info(
        "V_MOV_B32", "VOP1Inst", "v_mov_b32"
    )
    assert vop1["encoding"] == "VOP1"
    assert ["VGPR", "VGPR"] in vop1["operands"]

    # VOP3
    vop3 = scrape_amd_rdna.resolve_td_instruction_info(
        "V_FMA_F32", "VOP3Inst_gfx10", "v_fma_f32"
    )
    assert vop3["encoding"] == "VOP3"
    assert "clamp" in vop3["modifiers"]
    assert vop3["architecture"] == "GFX10/RDNA1"

    # VOPC
    vopc = scrape_amd_rdna.resolve_td_instruction_info(
        "V_CMP_EQ_F32", "VOPCInst_gfx11", "v_cmp_eq_f32"
    )
    assert vopc["encoding"] == "VOPC"
    assert ["VCC", "VGPR", "VGPR"] in vopc["operands"]
    assert vopc["architecture"] == "GFX11/RDNA3"

    # VOP3P
    vop3p = scrape_amd_rdna.resolve_td_instruction_info(
        "V_PK_ADD_F16", "VOP3PInst_gfx12", "v_pk_add_f16"
    )
    assert vop3p["encoding"] == "VOP3P"
    assert vop3p["architecture"] == "GFX12/RDNA4"

    # SOP2
    sop2 = scrape_amd_rdna.resolve_td_instruction_info(
        "S_ADD_U32", "SOP2_32_gfx10_3", "s_add_u32"
    )
    assert sop2["encoding"] == "SOP2"
    assert sop2["architecture"] == "GFX10.3/RDNA2"

    # SOPK
    sopk = scrape_amd_rdna.resolve_td_instruction_info(
        "S_MOVK_I32", "SOPK_32", "s_movk_i32"
    )
    assert sopk["encoding"] == "SOPK"
    assert ["SGPR", "simm16"] in sopk["operands"]

    # SOPP
    sopp = scrape_amd_rdna.resolve_td_instruction_info(
        "S_BRANCH", "SOPP_Pseudo", "s_branch"
    )
    assert sopp["encoding"] == "SOPP"

    # Memory: SM, FLAT, BUF, DS
    sm = scrape_amd_rdna.resolve_td_instruction_info(
        "S_LOAD_DWORD", "SM_Real_gfx9", "s_load_dword"
    )
    assert sm["encoding"] == "SM"
    assert sm["architecture"] == "GFX9/CDNA"

    flat = scrape_amd_rdna.resolve_td_instruction_info(
        "FLAT_LOAD_DWORD", "FLAT_Real_gfx11_5", "flat_load_dword"
    )
    assert flat["encoding"] == "FLAT"
    assert flat["architecture"] == "GFX11.5"

    buf = scrape_amd_rdna.resolve_td_instruction_info(
        "BUFFER_LOAD_DWORD", "BUF_Real", "buffer_load_dword"
    )
    assert buf["encoding"] == "BUF"

    ds = scrape_amd_rdna.resolve_td_instruction_info(
        "DS_READ_B32", "DS_Real", "ds_read_b32"
    )
    assert ds["encoding"] == "DS"

    # Fallback / default
    fallback = scrape_amd_rdna.resolve_td_instruction_info(
        "CUSTOM_INST", "CUSTOM_CLASS", "custom_inst"
    )
    assert fallback["encoding"] == "VOP1"

    # VOPD (dual-issue)
    vopd = scrape_amd_rdna.resolve_td_instruction_info(
        "V_DUAL_ADD_F32", "VOPD_Real_gfx11", "v_dual_add_f32"
    )
    assert vopd["encoding"] == "VOPD"
    assert "dual" in vopd["modifiers"]
    assert vopd["architecture"] == "GFX11/RDNA3"


def test_parse_td_content_multiclass() -> None:
    """Test parse_td_content evaluating TableGen multiclass expansions."""
    content = """
    defm V_DUAL_FMAC_F32 : VOPD_Real_gfx11;
    defm V_DUAL_FMAC_F32 : VOPD_Real_gfx11;
    """
    ops = scrape_amd_rdna.parse_td_content(content)
    assert len(ops) == 1
    assert ops[0]["mnemonic"] == "v_dual_fmac_f32"
    assert ops[0]["encoding"] == "VOPD"
