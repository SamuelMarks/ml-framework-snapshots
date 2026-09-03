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
def S_MOV_B32 : SOP1_Pseudo <"s_mov_b32">;
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "amd_rdna_exhaustive.json")

        with mock.patch(
            "ml_framework_snapshots.tools.scrape_amd_rdna.fetch_td_file"
        ) as mock_fetch:
            # Only return content for the first file to simulate
            # one file having content and the rest being empty or similar
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
        assert ["vGPR", "vGPR"] in v_add_entry["operands"]

        s_mov_entry = next(e for e in result_data if e["mnemonic"] == "s_mov_b32")
        assert s_mov_entry["architecture"] == "GFX10+"
