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
                "modifiers": ["FTZ", "???INVALID", ""],
                "operands": [
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "RegOperand", "reg_type": "R"},
                    {"type": "FloatIMMOperand"},
                ],
            },
            "opcode_modis": ["RN", "???BAD", ""],
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
                    {"type": "PredicateOperand"},
                    {"type": "AttributeOperand"},
                    {"type": "DescOperand"},
                    {"type": "SNOWFLAKE"},
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
        assert fadd_entry["modifiers"] == [".FTZ", ".RN"]
        assert fadd_entry["operands"] == [["R", "R", "FI"]]
        assert fadd_entry["architecture"] == [
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
        ]

        mov_entry = next(e for e in result_data if e["mnemonic"] == "MOV")
        assert mov_entry["modifiers"] == []
        assert mov_entry["operands"] == [["R", "R", "I"]]

        f2f_entry = next(e for e in result_data if e["mnemonic"] == "F2F")
        assert f2f_entry["operands"] == [
            [
                "R",
                "cx[bank][offset]",
                "c[bank][offset]",
                "ADDR",
                "P",
                "ATTR",
                "DESC",
                "R",
                "UnknownType",
            ]
        ]


def test_parse_nvdisasm_binary_info() -> None:
    """Test parsing text disassembly lines from nvdisasm output."""
    raw_disasm = """// Disassembly for sm_80
# Comments to ignore
/*0010*/  @!P0 FADD.FTZ.RN R1, R2, 0.5 ;
/*0015*/  @!P0 FADD.FTZ.RN R1, R2, 0.5 ;
/*0020*/  @P1 MOV R3, c[0x0][0x160] ;
/*0030*/  LDG.E.STRONG.GPU R4, [R5] ;
/*0040*/  ISETP.GE.AND P0, PT, R1, 0x10, PT ;
/*0050*/  NOP ;
/*0055*/  @P0 ;
/*0060*/  // Comment line
/*0070*/
"""
    records = scrape_nvidia_sass.parse_nvdisasm_binary_info(raw_disasm)
    assert len(records) >= 4

    fadd = next(r for r in records if r["mnemonic"] == "FADD")
    assert ".FTZ" in fadd["modifiers"]
    assert ".RN" in fadd["modifiers"]
    assert fadd["operands"] == [["R", "R", "FI"]]

    mov = next(r for r in records if r["mnemonic"] == "MOV")
    assert mov["operands"] == [["R", "c[bank][offset]"]]

    ldg = next(r for r in records if r["mnemonic"] == "LDG")
    assert ldg["operands"] == [["R", "ADDR"]]

    nop = next(r for r in records if r["mnemonic"] == "NOP")
    assert nop["operands"] == [[]]


def test_scrape_sass_nvdisasm_file() -> None:
    """Test scrape_sass when processing a raw nvdisasm text file."""
    raw_disasm = """/*0010*/  @P0 FFMA.RN R1, R2, R3, R4 ;
/*0020*/  FADD R0, UR1, 10 ;
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "nvdisasm.txt")
        output_path = os.path.join(temp_dir, "output.json")

        with open(input_path, "w") as f:
            f.write(raw_disasm)

        records = scrape_nvidia_sass.scrape_sass(
            input_path=input_path, output_path=output_path
        )
        assert os.path.exists(output_path)
        assert len(records) == 2
        ffma = next(r for r in records if r["mnemonic"] == "FFMA")
        assert ffma["operands"] == [["R", "R", "R", "R"]]
        fadd = next(r for r in records if r["mnemonic"] == "FADD")
        assert fadd["operands"] == [["R", "UR", "I"]]


def test_scrape_sass_json_list() -> None:
    """Test scrape_sass when input is valid JSON but formatted as a list."""
    json_content = json.dumps(["/*0010*/ FADD R1, R2, R3 ;"])
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "list.json")
        output_path = os.path.join(temp_dir, "output.json")

        with open(input_path, "w") as f:
            f.write(json_content)

        records = scrape_nvidia_sass.scrape_sass(
            input_path=input_path, output_path=output_path
        )
        assert os.path.exists(output_path)
        assert len(records) == 1
        assert records[0]["mnemonic"] == "FADD"


def test_scrape_sass_json_scalar() -> None:
    """Test scrape_sass when input is valid JSON but formatted as a scalar."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "scalar.json")
        output_path = os.path.join(temp_dir, "output.json")

        with open(input_path, "w") as f:
            f.write("123")

        records = scrape_nvidia_sass.scrape_sass(
            input_path=input_path, output_path=output_path
        )
        assert os.path.exists(output_path)
        assert len(records) == 1
        assert records[0]["mnemonic"] == "123"


def test_scrape_sass_env_var() -> None:
    """Test scrape_sass using the NVIDIA_SASS_INPUT_PATH environment variable."""
    raw_disasm = """/*0010*/  FADD R1, R2, R3 ;
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.txt")
        output_path = os.path.join(temp_dir, "output.json")

        with open(input_path, "w") as f:
            f.write(raw_disasm)

        with mock.patch.dict(os.environ, {"NVIDIA_SASS_INPUT_PATH": input_path}):
            records = scrape_nvidia_sass.scrape_sass(output_path=output_path)
            assert len(records) == 1


def test_resolve_instruction_architectures_and_expanded_catalog() -> None:
    """Test resolve_instruction_architectures branches and build_expanded_sass_catalog."""
    from ml_framework_snapshots.tools.scrape_nvidia_sass import (
        resolve_instruction_architectures,
        build_expanded_sass_catalog,
    )

    # 1. FP4/Blackwell branch
    assert resolve_instruction_architectures("MMA_SCALE_FP4") == ["sm_100"]

    # 2. WGMMA / TMA branch
    assert resolve_instruction_architectures("WGMMA_MMA") == ["sm_90", "sm_100"]
    assert resolve_instruction_architectures("TMA_LOAD") == ["sm_90", "sm_100"]

    # 3. HMMA / LDGSTS / LDSM branch
    assert resolve_instruction_architectures("HMMA16816") == [
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
    ]

    # 4. Uniform branch
    assert resolve_instruction_architectures("UIADD3") == [
        "sm_75",
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
    ]

    # 5. Default branch
    assert resolve_instruction_architectures("FADD") == [
        "sm_70",
        "sm_75",
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
    ]

    # 6. Test build_expanded_sass_catalog
    catalog = build_expanded_sass_catalog(
        [{"mnemonic": "CUSTOM_OP", "operands": [["R", "R"]]}]
    )
    assert len(catalog) > 800

    # 7. Test build_expanded_sass_catalog with duplicates and pre-existing entries
    base_with_all: typing.List[typing.Dict[str, typing.Any]] = [
        {"mnemonic": "DUPLICATE", "operands": [["R", "R"]]},
        {"mnemonic": "DUPLICATE", "operands": [["R", "R"]]},
        {"mnemonic": "YIELD", "operands": []},
        {"mnemonic": "LDG_E_32", "operands": [["R", "ADDR"]]},
        {"mnemonic": "ATOMG_ADD_32", "operands": [["R", "ADDR", "R"]]},
        {"mnemonic": "FADD_F32_RN", "operands": [["R", "R", "R"]]},
        {"mnemonic": "CVT_F32_TO_F64", "operands": [["R", "R"]]},
        {"mnemonic": "UIADD3_32", "operands": [["UR", "UR", "UR"]]},
    ]
    catalog_all = build_expanded_sass_catalog(base_with_all)
    assert len(catalog_all) > 800
