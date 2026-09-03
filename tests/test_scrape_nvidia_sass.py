"""Tests for the NVIDIA SASS scraper script."""

import json
import os
import tempfile
from unittest import mock

from ml_framework_snapshots.tools import scrape_nvidia_sass

import typing


def test_scrape_nvidia_sass() -> None:
    """Test the NVIDIA SASS scrape script with a mock JSON input."""
    # Create mock input data
    mock_input_data = {
        "1057.FADD_R_R_FI": {
            "parsed": {
                "base_name": "FADD",
                "modifiers": ["FTZ"],
                "operands": [
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "FloatIMMOperand"},
                ],
            },
            "opcode_modis": ["RN"],
        },
        "514.MOV_R_R_I": {
            "parsed": {
                "base_name": "MOV",
                "modifiers": [],
                "operands": [
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "IntIMMOperand"},
                ],
            }
        },
        "2820.F2F_R_cx[UR][I]": {
            "parsed": {
                "base_name": "F2F",
                "operands": [
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "ConstantMemOperand", "cx": True},
                    {"type": "ConstantMemOperand", "cx": False},
                    {"type": "AddressOperand"},
                    {"type": "UnknownType"},
                ],
            }
        },
        "invalid_entry": {"parsed": {}},  # No base_name
        "duplicate_entry": {
            "parsed": {
                "base_name": "FADD",
                "operands": [
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "FloatIMMOperand"},
                ],
            }
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "isa.json")
        output_path = os.path.join(temp_dir, "nvidia_sass_exhaustive.json")

        with open(input_path, "w") as f:
            json.dump(mock_input_data, f)

        # Patch the file paths in the script
        with mock.patch(
            "ml_framework_snapshots.tools.scrape_nvidia_sass.open"
        ) as mock_open:
            # We want to use the real open, but intercept the paths
            original_open = open

            def side_effect(
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
                if "/tmp/isa.json" in path:
                    return original_open(input_path, *args, **kwargs)
                if "nvidia_sass_exhaustive.json" in path:
                    return original_open(output_path, *args, **kwargs)
                return original_open(path, *args, **kwargs)  # pragma: no cover

            mock_open.side_effect = side_effect

            # Run the main function
            scrape_nvidia_sass.main()

        # Verify output
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            result_data = json.load(f)

        assert len(result_data) == 3

        fadd_entry = next(e for e in result_data if e["mnemonic"] == "FADD")
        assert fadd_entry["modifiers"] == ["FTZ", "RN"]
        assert fadd_entry["operands"] == [["R", "R", "FI"]]
        assert fadd_entry["architecture"] == "sm_80+"

        mov_entry = next(e for e in result_data if e["mnemonic"] == "MOV")
        assert mov_entry["modifiers"] == []
        assert mov_entry["operands"] == [["R", "R", "I"]]

        f2f_entry = next(e for e in result_data if e["mnemonic"] == "F2F")
        assert f2f_entry["operands"] == [
            ["R", "cx[x][y]", "c[x][y]", "ADDR", "UnknownType"]
        ]
