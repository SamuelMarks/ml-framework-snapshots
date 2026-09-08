"""Unit test suite for the NVIDIA PTX ISA TableGen scraper tool."""

import json
import os
import urllib.error
from unittest import mock

from ml_framework_snapshots.tools import scrape_nvidia_ptx


def test_fetch_nvptx_td_file_local_dir(tmp_path: os.PathLike[str]) -> None:
    """Test fetching TableGen file from a local directory path.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    td_file = os.path.join(tmp_path, "NVPTXInstrInfo.td")
    with open(td_file, "w", encoding="utf-8") as f:
        f.write('def ADD_F32 : NVPTXInst<"add.f32">;')

    content = scrape_nvidia_ptx.fetch_nvptx_td_file(
        "NVPTXInstrInfo.td", local_dir=str(tmp_path)
    )
    assert 'def ADD_F32 : NVPTXInst<"add.f32">' in content


def test_fetch_nvptx_td_file_remote_success() -> None:
    """Test fetching TableGen file successfully over HTTP mock."""
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = mock.MagicMock()
        mock_response.read.return_value = b'def SUB_F32 : NVPTXInst<"sub.f32">;'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        content = scrape_nvidia_ptx.fetch_nvptx_td_file("NVPTXInstrInfo.td")
        assert 'def SUB_F32 : NVPTXInst<"sub.f32">' in content


def test_fetch_nvptx_td_file_remote_error() -> None:
    """Test handling urllib error gracefully when fetching TableGen files."""
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("Network unreachable")
        content = scrape_nvidia_ptx.fetch_nvptx_td_file("NVPTXInstrInfo.td")
        assert content == ""


def test_parse_nvptx_td_content() -> None:
    """Test extracting instruction records from TableGen text content."""
    mock_content = """
    def INSTR_ADD : NVPTXInst<"add">;
    def INSTR_SUB : NVPTXInst<"sub">;
    def SOME_OTHER_DEF : OtherClass;
    """
    records = scrape_nvidia_ptx.parse_nvptx_td_content(mock_content)
    mnemonics = [r["mnemonic"] for r in records]
    assert "add" in mnemonics
    assert "sub" in mnemonics
    assert len(records) == 2


def test_scrape_ptx_end_to_end(tmp_path: os.PathLike[str]) -> None:
    """Test end-to-end PTX scraper with output file creation.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    out_file = os.path.join(tmp_path, "nvidia_ptx_exhaustive.json")
    with mock.patch(
        "ml_framework_snapshots.tools.scrape_nvidia_ptx.fetch_nvptx_td_file",
        return_value='def DUMMY_OP : NVPTXInst<"custom_ptx_op">;',
    ):
        catalog = scrape_nvidia_ptx.scrape_ptx(output_path=out_file)
        assert os.path.isfile(out_file)
        assert any(op["mnemonic"] == "custom_ptx_op" for op in catalog)
        assert any(op["mnemonic"] == "cp.async" for op in catalog)
        assert any(op["mnemonic"] == "wgmma.mma_async" for op in catalog)

        with open(out_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
            assert len(saved) == len(catalog)


def test_fetch_nvptx_td_file_candidate_not_file(tmp_path: os.PathLike[str]) -> None:
    """Test falling through to urlopen when local_dir candidate is not a file."""
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'def FALLBACK : NVPTXInst<"fallback">;'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        content = scrape_nvidia_ptx.fetch_nvptx_td_file(
            "nonexistent.td", local_dir=str(tmp_path)
        )
        assert 'def FALLBACK : NVPTXInst<"fallback">' in content


def test_build_exhaustive_ptx_catalog_edge_cases() -> None:
    """Test catalog building with missing mnemonics and duplicate mnemonics."""
    base_insts = [
        {"category": "no_mnemonic"},
        {"mnemonic": "add", "description": "Duplicate add"},
    ]
    catalog = scrape_nvidia_ptx.build_exhaustive_ptx_catalog(base_insts)
    assert any(op["mnemonic"] == "add" for op in catalog)


def test_scrape_ptx_with_empty_fetch(tmp_path: os.PathLike[str]) -> None:
    """Test scrape_ptx when TableGen fetch returns empty string."""
    out_file = os.path.join(tmp_path, "empty_ptx.json")
    with mock.patch(
        "ml_framework_snapshots.tools.scrape_nvidia_ptx.fetch_nvptx_td_file",
        return_value="",
    ):
        catalog = scrape_nvidia_ptx.scrape_ptx(output_path=out_file)
        assert len(catalog) > 0


def test_main_invocation() -> None:
    """Test the main entrypoint function invokes scrape_ptx."""
    with mock.patch(
        "ml_framework_snapshots.tools.scrape_nvidia_ptx.scrape_ptx"
    ) as mock_scrape:
        scrape_nvidia_ptx.main()
        mock_scrape.assert_called_once()
