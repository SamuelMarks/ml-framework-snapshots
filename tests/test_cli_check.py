"""Test CLI check command."""

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
    """Function docstring."""
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

    snap_file = tmp_path / "snap.json"
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
