"""Test CLI check command."""

from pathlib import Path
import os

from typing import Any


import argparse
from unittest.mock import patch, mock_open
from ml_framework_snapshots.cli import cmd_check

MOCK_JSON = """
{
    "categories": {
        "functions": [
            {
                "name": "func4",
                "framework": "mod",
                "api_path": "mod.func4",
                "symbol": "func4",
                "kind": "function",
                "is_public": true,
                "params": [],
                "aliases": [],
                "docstring": "Mock func4",
                "signature": "()"
            },
            {
                "name": "func5",
                "framework": "other",
                "api_path": "other.func5",
                "symbol": "func5",
                "kind": "function",
                "is_public": true,
                "params": [],
                "aliases": [],
                "docstring": "Mock func5",
                "signature": "()"
            }
        ]
    }
}
"""


@patch("ml_framework_snapshots.compliance.score_compliance")
@patch("ml_framework_snapshots.compliance.extract_target_refs")
@patch("builtins.open", new_callable=mock_open, read_data=MOCK_JSON)
def test_cmd_check(
    mock_file: Any, mock_extract: Any, mock_score: Any, capsys: Any
) -> None:
    """Test basic check command reporting.

    Args:
        mock_file: mock
        mock_extract: mock
        mock_score: mock
        capsys: capsys fixture
    """
    args = argparse.Namespace(
        snapshot_json="dummy.json",
        target_path="dummy_target",
        target_prefix="t_pref",
        reference_prefix="r_pref",
    )

    mock_extract.return_value = ["mock_ref"]
    mock_score.return_value = {
        "score_percentage": 75.5,
        "matched": ["mod.func1", "mod.func2", "other.func3"],
        "missing": ["mod.func4"],
        "mismatched": [{"api_path": "other.func5", "expected": [], "actual": []}],
    }

    import json

    cmd_check(args)

    mock_extract.assert_called_once_with("dummy_target", "t_pref", "r_pref")
    mock_score.assert_called_once_with(json.loads(MOCK_JSON), ["mock_ref"])

    captured = capsys.readouterr()
    assert "Extracting target APIs from dummy_target" in captured.out
    assert "Overall Compliance: 75.5%" in captured.out
    assert "- mod: 66.7% (2/3)" in captured.out
    assert "- other: 100.0% (1/1)" in captured.out
    assert "Missing APIs (1):" in captured.out
    assert "mod.func4" in captured.out
    assert "Mismatched APIs (1):" in captured.out
    assert "other.func5" in captured.out


@patch("ml_framework_snapshots.compliance.score_compliance")
@patch("ml_framework_snapshots.compliance.extract_target_refs")
@patch("builtins.open", new_callable=mock_open, read_data='{"categories": {}}')
def test_cmd_check_pagination(
    mock_file: Any, mock_extract: Any, mock_score: Any, capsys: Any
) -> None:
    """Test pagination in check command reporting.

    Args:
        mock_file: mock
        mock_extract: mock
        mock_score: mock
        capsys: capsys fixture
    """
    args = argparse.Namespace(
        snapshot_json="dummy.json",
        target_path="dummy_target",
        target_prefix="t_pref",
        reference_prefix="r_pref",
    )

    mock_extract.return_value = []

    # Generate > 20 missing and mismatched
    missing = [f"mod.func{i}" for i in range(25)]
    mismatched = [
        {"api_path": f"mod.func{i}", "expected": [], "actual": []}
        for i in range(30, 55)
    ]

    mock_score.return_value = {
        "score_percentage": 0.0,
        "matched": [],
        "missing": missing,
        "mismatched": mismatched,
    }

    cmd_check(args)

    captured = capsys.readouterr()
    assert "... and 5 more" in captured.out


@patch("ml_framework_snapshots.compliance.score_compliance")
@patch("ml_framework_snapshots.compliance.extract_target_refs")
@patch("builtins.open", new_callable=mock_open, read_data='{"categories": {}}')
def test_cmd_check_no_missing_mismatched(
    mock_file: Any, mock_extract: Any, mock_score: Any, capsys: Any
) -> None:
    """Test check command with complete compliance.

    Args:
        mock_file: mock
        mock_extract: mock
        mock_score: mock
        capsys: capsys fixture
    """
    args = argparse.Namespace(
        snapshot_json="dummy.json",
        target_path="dummy_target",
        target_prefix="t_pref",
        reference_prefix="r_pref",
    )

    mock_extract.return_value = []
    mock_score.return_value = {
        "score_percentage": 100.0,
        "matched": ["mod.func1"],
        "missing": [],
        "mismatched": [],
    }

    cmd_check(args)
    captured = capsys.readouterr()
    assert "Missing APIs" not in captured.out
    assert "Mismatched APIs" not in captured.out


def test_cmd_check_output_formatting(mocker: Any, capsys: Any, tmp_path: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
        tmp_path: Parameter.
    """
    from ml_framework_snapshots.cli import cmd_check
    from ml_switcheroo_ir.schema.ghost import GhostRef, GhostParam
    import json

    ref = GhostRef(
        name="func",
        api_path="torch.nn.func",
        kind="FUNCTION",
        params=[
            GhostParam(
                name="p1", kind="POSITIONAL_OR_KEYWORD", annotation="int", default="0"
            ),
            GhostParam(name="p2", kind="POSITIONAL_OR_KEYWORD"),
        ],
        returns_type="int",
        docstring="Long docstring that is really long and should be truncated if it exceeds one hundred characters let us see if it is",
        aliases=["torch.func"],
    )

    ref2 = GhostRef(
        name="mod",
        api_path="torch.nn.mod",
        kind="MODULE",
        params=[],
        docstring="",
        aliases=[],
    )

    snapshot = {"categories": {"LAYER": [ref.model_dump(), ref2.model_dump()]}}

    snap_file = Path(os.path.join(tmp_path, "snap.json"))
    snap_file.write_text(json.dumps(snapshot))

    mocker.patch(
        "ml_framework_snapshots.cli.resolve_snapshot_path", return_value=str(snap_file)
    )
    mocker.patch(
        "ml_framework_snapshots.compliance.extract_target_refs", return_value=[]
    )
    mocker.patch(
        "ml_framework_snapshots.compliance.score_compliance",
        return_value={
            "score_percentage": 0.0,
            "missing": ["torch.nn.func", "torch.nn.mod", "torch.func"],
        },
    )

    import argparse

    args = argparse.Namespace(
        snapshot_json="torch",
        target_path="tests",
        target_prefix="torch",
        reference_prefix="torch",
    )
    cmd_check(args)
    out, _ = capsys.readouterr()
    assert "p1: int=0" in out
    assert "-> int" in out
    assert "..." in out
    assert "No docstring available." in out
    assert "module" in out


def test_cmd_check_sass_no_modifiers(mocker: Any, capsys: Any) -> None:
    """Test cmd_check_sass when valid instruction has no valid modifiers.

    Args:
        mocker: Pytest mocker fixture.
        capsys: Pytest capsys fixture.
    """
    from ml_framework_snapshots.cli import cmd_check_sass

    mocker.patch(
        "ml_framework_snapshots.mcp_server.check_sass_instruction",
        return_value={
            "is_valid": True,
            "supported_architectures": ["sm_80"],
            "valid_modifiers": [],
        },
    )
    args = argparse.Namespace(
        mnemonic="NOP",
        operands=None,
        modifiers=None,
        sm_arch="sm_80",
        file=None,
    )
    cmd_check_sass(args)
    out, _ = capsys.readouterr()
    assert "SASS Instruction 'NOP' is valid." in out
    assert "Recognized Modifiers" not in out


def test_cmd_check_rdna_no_encoding(mocker: Any, capsys: Any) -> None:
    """Test cmd_check_rdna when valid instruction has no encoding returned.

    Args:
        mocker: Pytest mocker fixture.
        capsys: Pytest capsys fixture.
    """
    from ml_framework_snapshots.cli import cmd_check_rdna

    mocker.patch(
        "ml_framework_snapshots.mcp_server.check_rdna_instruction",
        return_value={
            "is_valid": True,
            "supported_architectures": ["gfx1100"],
            "encoding": None,
        },
    )
    args = argparse.Namespace(
        mnemonic="s_nop",
        operands=None,
        encoding=None,
        gfx_arch="gfx1100",
        modifiers=None,
        wave_size=32,
        file=None,
    )
    cmd_check_rdna(args)
    out, _ = capsys.readouterr()
    assert "RDNA Instruction 's_nop' is valid." in out
    assert "Encoding:" not in out


def test_cmd_check_non_python_auto_detect(mocker: Any, capsys: Any) -> None:
    """Test cmd_check auto-detecting .sass, .s, and .mlir file extensions."""
    import pytest
    from ml_framework_snapshots.cli import cmd_check

    # 1. SASS file compliant
    mocker.patch(
        "ml_framework_snapshots.compliance.check_sass_assembly_compliance",
        return_value={
            "is_compliant": True,
            "total_instructions": 2,
            "verified_instructions": 2,
            "errors": [],
        },
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data='{"target": "nvidia_sass"}'),
    )
    args_sass = argparse.Namespace(
        snapshot_json="nvidia_sass.json",
        target_path="kernel.sass",
        target_prefix="",
        reference_prefix="",
    )
    cmd_check(args_sass)
    out, _ = capsys.readouterr()
    assert "NVIDIA SASS snippet compliance: 2/2 instructions valid." in out

    # 2. SASS file non-compliant (exits 1)
    mocker.patch(
        "ml_framework_snapshots.compliance.check_sass_assembly_compliance",
        return_value={
            "is_compliant": False,
            "total_instructions": 2,
            "verified_instructions": 1,
            "errors": ["Bad opcode"],
        },
    )
    with pytest.raises(SystemExit):
        cmd_check(args_sass)

    # 3. RDNA .s file compliant
    mocker.patch(
        "ml_framework_snapshots.compliance.check_rdna_assembly_compliance",
        return_value={
            "is_compliant": True,
            "total_instructions": 3,
            "verified_instructions": 3,
            "errors": [],
        },
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data='{"target": "amd_rdna"}'),
    )
    args_rdna = argparse.Namespace(
        snapshot_json="amd_rdna.json",
        target_path="kernel.s",
        target_prefix="",
        reference_prefix="",
    )
    cmd_check(args_rdna)
    out, _ = capsys.readouterr()
    assert "AMD RDNA snippet compliance: 3/3 instructions valid." in out

    # 4. RDNA .s file non-compliant (exits 1)
    mocker.patch(
        "ml_framework_snapshots.compliance.check_rdna_assembly_compliance",
        return_value={
            "is_compliant": False,
            "total_instructions": 3,
            "verified_instructions": 2,
            "errors": ["Bad reg"],
        },
    )
    with pytest.raises(SystemExit):
        cmd_check(args_rdna)

    # 5. MLIR file compliant
    mocker.patch(
        "ml_framework_snapshots.compliance.check_mlir_text_compliance",
        return_value={
            "is_compliant": True,
            "total_ops": 4,
            "verified_ops": 4,
            "errors": [],
        },
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data='{"target": "mlir"}'),
    )
    args_mlir = argparse.Namespace(
        snapshot_json="mlir.json",
        target_path="module.mlir",
        target_prefix="",
        reference_prefix="",
    )
    cmd_check(args_mlir)
    out, _ = capsys.readouterr()
    assert "MLIR snippet compliance: 4/4 operations valid." in out

    # 6. MLIR file non-compliant (exits 1)
    mocker.patch(
        "ml_framework_snapshots.compliance.check_mlir_text_compliance",
        return_value={
            "is_compliant": False,
            "total_ops": 4,
            "verified_ops": 2,
            "errors": ["Unknown dialect op"],
        },
    )
    with pytest.raises(SystemExit):
        cmd_check(args_mlir)
