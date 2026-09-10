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


def test_fetch_td_file_cache(tmp_path: typing.Any, monkeypatch: typing.Any) -> None:
    """Test fetch_td_file with disk cache hit and write.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = os.path.join(str(tmp_path), "ml_framework_snapshots", "amdgpu_td")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cached.td")
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write("cached_td_content")

    # Hit cache without urlopen
    cached_content = scrape_amd_rdna.fetch_td_file("cached.td", use_cache=True)
    assert cached_content == "cached_td_content"

    # Miss cache and write new file
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = mock.MagicMock()
        mock_response.read.return_value = b"downloaded_td_content"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        downloaded = scrape_amd_rdna.fetch_td_file("new.td", use_cache=True)
        assert downloaded == "downloaded_td_content"
        new_cache_file = os.path.join(cache_dir, "new.td")
        assert os.path.isfile(new_cache_file)


def test_fetch_td_file_cache_exceptions(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    """Test fetch_td_file handling of read/write cache exceptions and empty content.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = os.path.join(str(tmp_path), "ml_framework_snapshots", "amdgpu_td")
    os.makedirs(cache_dir, exist_ok=True)
    unreadable = os.path.join(cache_dir, "unreadable.td")
    with open(unreadable, "w", encoding="utf-8") as f:
        f.write("content")

    # 1. Read exception falls through to urlopen
    with mock.patch(
        "builtins.open", side_effect=[OSError("Permission denied"), mock.mock_open()()]
    ):
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = b"fallback_content"
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            res = scrape_amd_rdna.fetch_td_file("unreadable.td", use_cache=True)
            assert res == "fallback_content"

    # 2. Write exception after download does not crash
    with mock.patch("os.makedirs", side_effect=OSError("Disk full")):
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = b"no_write_content"
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            res2 = scrape_amd_rdna.fetch_td_file("no_write.td", use_cache=True)
            assert res2 == "no_write_content"

    # 3. Empty downloaded content (branch use_cache and content: content is "")
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res3 = scrape_amd_rdna.fetch_td_file("empty.td", use_cache=True)
        assert res3 == ""


def test_fetch_td_file_error() -> None:
    """Test fetch_td_file on error."""
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("Not Found")
        content = scrape_amd_rdna.fetch_td_file("error.td", use_cache=False)
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

            def fetch_side_effect(
                filename: str, local_dir: typing.Optional[str] = None
            ) -> str:
                """Mock fetch_td_file side effect.

                Args:
                    filename: The filename to fetch.
                    local_dir: Optional local directory parameter.

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


def test_parse_llvm_tblgen_json() -> None:
    """Test parse_llvm_tblgen_json extracting register classes, encodings, and modifiers."""
    mock_tblgen = {
        "!tablegen_json_version": 1,
        "!instanceof": {"VOP1Inst": ["V_MOV_B32_e32"]},
        "ignore_str": "not_a_dict",
        "V_MOV_B32_e32": {
            "!superclasses": ["VOP1Inst", "GFX10"],
            "Mnemonic": "v_mov_b32_e32",
            "OutOperandList": [{"def": "VGPR_32"}],
            "InOperandList": [{"def": "VGPR_32"}],
        },
        "V_FMA_F32_e64": {
            "!superclasses": ["VOP3Inst", "GFX11"],
            "Mnemonic": {"def": "v_fma_f32"},
            "OutOperandList": ["VGPR_32"],
            "InOperandList": ["VGPR_32", "VGPR_32", "VGPR_32"],
        },
        "V_PK_ADD_F16": {
            "!superclasses": ["VOP3PInst", "GFX12"],
            "Mnemonic": "v_pk_add_f16",
            "OutOperandList": ["VReg_64"],
            "InOperandList": ["VReg_64", "VReg_64"],
        },
        "V_DUAL_ADD_F32": {
            "!superclasses": ["VOPDInst", "GFX11_5"],
            "OutOperandList": ["VGPR_32"],
            "InOperandList": ["VGPR_32", "VGPR_32"],
        },
        "V_MFMA_F32": {
            "!superclasses": ["VOP3Inst", "GFX9"],
            "Mnemonic": "v_mfma_f32_16x16x16f16",
            "OutOperandList": ["AReg_32"],
            "InOperandList": ["VReg_64", "VReg_64", "AReg_32"],
        },
        "V_NOP": {
            "!superclasses": ["VOP1Inst", "GFX10_3"],
            "Mnemonic": "v_nop",
            "OutOperandList": [],
            "InOperandList": [],
        },
        "V_ADD_GENERIC": {
            "!superclasses": ["VOP2Inst"],
            "Mnemonic": "v_add_generic",
            "OutOperandList": ["VGPR_32"],
            "InOperandList": ["VGPR_32", "VGPR_32"],
        },
    }

    results = scrape_amd_rdna.parse_llvm_tblgen_json(mock_tblgen)
    res_map = {r["mnemonic"]: r for r in results}

    assert "v_add_generic" in res_map
    assert res_map["v_add_generic"]["architecture"] == "GFX10+"
    assert "_e32" in res_map["v_add_generic"]["modifiers"]

    assert "v_mov_b32" in res_map
    assert res_map["v_mov_b32"]["encoding"] == "VOP1"
    assert res_map["v_mov_b32"]["architecture"] == "GFX10/RDNA1"
    assert res_map["v_mov_b32"]["operands"] == [["VGPR_32", "VGPR_32"]]
    assert "_e32" in res_map["v_mov_b32"]["modifiers"]

    assert "v_fma_f32" in res_map
    assert res_map["v_fma_f32"]["encoding"] == "VOP3"
    assert res_map["v_fma_f32"]["architecture"] == "GFX11/RDNA3"
    assert "clamp" in res_map["v_fma_f32"]["modifiers"]

    assert "v_pk_add_f16" in res_map
    assert res_map["v_pk_add_f16"]["encoding"] == "VOP3P"
    assert res_map["v_pk_add_f16"]["architecture"] == "GFX12/RDNA4"
    assert "op_sel" in res_map["v_pk_add_f16"]["modifiers"]

    assert "v_dual_add_f32" in res_map
    assert res_map["v_dual_add_f32"]["encoding"] == "VOPD"
    assert res_map["v_dual_add_f32"]["architecture"] == "GFX11.5"
    assert "dual" in res_map["v_dual_add_f32"]["modifiers"]

    assert "v_mfma_f32_16x16x16f16" in res_map
    assert res_map["v_mfma_f32_16x16x16f16"]["architecture"] == "GFX9/CDNA"

    assert "v_nop" in res_map
    assert res_map["v_nop"]["architecture"] == "GFX10.3/RDNA2"


def test_scrape_amd_rdna_with_tblgen_json(tmp_path: typing.Any) -> None:
    """Test scrape_amd_rdna reading from a TableGen JSON export file."""
    mock_tblgen = {
        "V_ADD_F32_e32": {
            "!superclasses": ["VOP2Inst", "GFX10"],
            "Mnemonic": "v_add_f32",
            "OutOperandList": [{"def": "VGPR_32"}],
            "InOperandList": [{"def": "VGPR_32"}, {"def": "VGPR_32"}],
        }
    }
    json_path = tmp_path / "dump.json"
    json_path.write_text(json.dumps(mock_tblgen), encoding="utf-8")
    out_path = tmp_path / "out.json"

    result = scrape_amd_rdna.scrape_amd_rdna(
        output_path=str(out_path), tblgen_json_path=str(json_path)
    )
    assert len(result) == 1
    assert result[0]["mnemonic"] == "v_add_f32"
    assert result[0]["encoding"] == "VOP2"


def test_parse_llvm_tblgen_json_operand_and_encoding_fallbacks() -> None:
    """Test TableGen JSON parsing edge cases for operand lists, unknown encoding, and non-VOPD."""
    mock_tblgen = {
        # 1. Unknown encoding profile with empty operands -> covers 305->327, 332->339, 339->345, 349->350, line 350
        "CUSTOM_NOP": {
            "!superclasses": ["SomeCustomClass"],
            "Mnemonic": "custom_nop",
            "OutOperandList": None,
            "InOperandList": None,
        },
        # 2. Empty op_name in OutOperandList and InOperandList -> covers 335->333, 342->340
        "V_EMPTY_OPS": {
            "!superclasses": ["VOP1Inst"],
            "Mnemonic": "v_empty_ops",
            "OutOperandList": [{"def": ""}],
            "InOperandList": [{"def": ""}],
        },
        # 3. Non-VOPD encoding with non-empty operands -> covers 359->363
        "S_MOV_SOP1": {
            "!superclasses": ["SOP1_32"],
            "Mnemonic": "s_mov_sop1",
            "OutOperandList": ["SGPR_32"],
            "InOperandList": ["SGPR_32"],
        },
    }

    results = scrape_amd_rdna.parse_llvm_tblgen_json(mock_tblgen)
    res_map = {r["mnemonic"]: r for r in results}

    assert "custom_nop" in res_map
    assert "v_empty_ops" in res_map
    assert "s_mov_sop1" in res_map
    assert res_map["s_mov_sop1"]["encoding"] == "SOP1"
    assert res_map["s_mov_sop1"]["modifiers"] == []


def test_scrape_amd_rdna_duplicate_operand_signatures(tmp_path: typing.Any) -> None:
    """Test scrape_amd_rdna when multiple records produce duplicate operand signatures."""
    mock_tblgen = {
        "V_ADD_F32_e32_1": {
            "!superclasses": ["VOP2Inst", "GFX10"],
            "Mnemonic": "v_add_f32",
            "OutOperandList": ["VGPR_32"],
            "InOperandList": ["VGPR_32", "VGPR_32"],
        },
        "V_ADD_F32_e32_2": {
            "!superclasses": ["VOP2Inst", "GFX10"],
            "Mnemonic": "v_add_f32",
            "OutOperandList": ["VGPR_32"],
            "InOperandList": ["VGPR_32", "VGPR_32"],
        },
    }
    json_path = tmp_path / "dump_dup.json"
    json_path.write_text(json.dumps(mock_tblgen), encoding="utf-8")
    out_path = tmp_path / "out_dup.json"

    result = scrape_amd_rdna.scrape_amd_rdna(
        output_path=str(out_path), tblgen_json_path=str(json_path)
    )
    assert len(result) == 1
    assert result[0]["mnemonic"] == "v_add_f32"
    assert len(result[0]["operands"]) == 1


def test_fetch_td_file_local_dir(tmp_path: typing.Any) -> None:
    """Test fetching TableGen file from local checkout directory."""
    local_td = tmp_path / "VOPDInstructions.td"
    local_td.write_text('def V_DUAL_ADD : VOPD<"v_dual_add">;', encoding="utf-8")

    content = scrape_amd_rdna.fetch_td_file(
        "VOPDInstructions.td", local_dir=str(tmp_path)
    )
    assert 'def V_DUAL_ADD : VOPD<"v_dual_add">' in content

    # Test scrape_amd_rdna using local_dir
    out_path = tmp_path / "local_out.json"
    res = scrape_amd_rdna.scrape_amd_rdna(
        output_path=str(out_path), local_dir=str(tmp_path)
    )
    assert isinstance(res, list)
