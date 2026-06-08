"""Module docstring."""

from ml_framework_snapshots.cli import main
from unittest.mock import mock_open


def test_cli_capture(mocker, capsys):
    """Function docstring."""
    mocker.patch("sys.argv", ["ml-snapshots", "capture", "--out-dir", "test_out"])

    # Mock extract_snapshot and write_snapshot
    def mock_extract(fw, include_nonpublic=False):
        """Function docstring.

        Args:
            fw: description
            include_nonpublic: description
        """
        if fw == "torch":
            return {"version": "1.0"}
        return {}

    mocker.patch(
        "ml_framework_snapshots.cli.extract_snapshot", side_effect=mock_extract
    )
    mocker.patch(
        "ml_framework_snapshots.cli.write_snapshot",
        return_value="test_out/torch_v1.0.json",
    )

    # Run CLI
    main()

    captured = capsys.readouterr()
    assert "Saved snapshot to test_out/torch_v1.0.json" in captured.out
    assert "Saved snapshot to test_out/torch_v1.0.json" in captured.out
    assert "Skipping jax, not installed" in captured.out


def test_cli_diff(mocker, capsys):
    """Function docstring."""
    mocker.patch("sys.argv", ["ml-snapshots", "diff", "1.json", "2.json"])

    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch("json.load", side_effect=[{"categories": {}}, {"categories": {}}])

    from ml_framework_snapshots.diff import DiffResult

    mocker.patch(
        "ml_framework_snapshots.cli.diff_snapshots",
        return_value=DiffResult(added=["a"], removed=["b"], signature_changed=["c"]),
    )

    main()
    captured = capsys.readouterr()
    assert "ADDED: 1" in captured.out
    assert "+ a" in captured.out
    assert "- b" in captured.out
    assert "~ c" in captured.out


def test_cli_diff_changelog(mocker, capsys):
    """Function docstring."""
    mocker.patch(
        "sys.argv", ["ml-snapshots", "diff", "1.json", "2.json", "--changelog"]
    )

    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch("json.load", side_effect=[{"categories": {}}, {"categories": {}}])

    from ml_framework_snapshots.diff import DiffResult

    mocker.patch(
        "ml_framework_snapshots.cli.diff_snapshots",
        return_value=DiffResult(added=["a"], removed=["b"], signature_changed=["c"]),
    )

    mocker.patch(
        "ml_framework_snapshots.cli.generate_changelog",
        return_value="## Changelog Mock",
    )

    main()
    captured = capsys.readouterr()
    assert "## Changelog Mock" in captured.out


def test_cli_stubs(mocker, capsys):
    """Function docstring."""
    mocker.patch(
        "sys.argv",
        ["ml-snapshots", "generate-stubs", "--input", "in.json", "--out-dir", "out"],
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch("json.load", return_value={"categories": {}})

    mock_gen = mocker.patch("ml_framework_snapshots.cli.generate_stubs")

    main()
    captured = capsys.readouterr()
    mock_gen.assert_called_once()
    assert "Stubs generated in out" in captured.out


def test_cli_export_openapi(mocker, capsys):
    """Function docstring."""
    mocker.patch(
        "sys.argv",
        [
            "ml-snapshots",
            "export",
            "--input",
            "in.json",
            "--out-dir",
            "out",
            "--format",
            "openapi",
        ],
    )

    mock_snap = {
        "categories": {
            "test": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "params": [],
                }
            ]
        }
    }

    mocker.patch("builtins.open", mock_open(read_data="{}"))
    mocker.patch("json.load", return_value=mock_snap)
    mocker.patch("os.makedirs")

    mock_to_openapi = mocker.patch(
        "ml_framework_snapshots.export.to_openapi", return_value={"openapi": "3.0.0"}
    )

    main()
    captured = capsys.readouterr()
    mock_to_openapi.assert_called_once()
    assert "Exported OpenAPI spec to out/openapi.json" in captured.out


def test_cli_export_json_schema(mocker, capsys):
    """Function docstring."""
    mocker.patch(
        "sys.argv",
        [
            "ml-snapshots",
            "export",
            "--input",
            "in.json",
            "--out-dir",
            "out",
            "--format",
            "json_schema",
        ],
    )

    mock_snap = {
        "categories": {
            "test": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "params": [],
                }
            ]
        }
    }

    mocker.patch("builtins.open", mock_open(read_data="{}"))
    mocker.patch("json.load", return_value=mock_snap)
    mocker.patch("os.makedirs")

    mock_to_json_schema = mocker.patch(
        "ml_framework_snapshots.export.to_json_schema", return_value={"$id": "test"}
    )

    main()
    captured = capsys.readouterr()
    mock_to_json_schema.assert_called_once()
    assert "Exported 1 JSON schemas to out" in captured.out


def test_cli_export_pydantic(mocker, capsys):
    """Function docstring."""
    mocker.patch(
        "sys.argv",
        [
            "ml-snapshots",
            "export",
            "--input",
            "in.json",
            "--out-dir",
            "out",
            "--format",
            "pydantic",
        ],
    )

    mock_snap = {
        "categories": {
            "test": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "params": [],
                }
            ]
        }
    }

    mocker.patch("builtins.open", mock_open(read_data="{}"))
    mocker.patch("json.load", return_value=mock_snap)
    mocker.patch("os.makedirs")

    mock_to_pydantic = mocker.patch(
        "ml_framework_snapshots.export.to_pydantic", return_value="class Linear:"
    )

    main()
    captured = capsys.readouterr()
    mock_to_pydantic.assert_called_once()
    assert "Exported 1 Pydantic models to out" in captured.out


def test_cli_export_protobuf(mocker, capsys):
    """Function docstring."""
    mocker.patch(
        "sys.argv",
        [
            "ml-snapshots",
            "export",
            "--input",
            "in.json",
            "--out-dir",
            "out",
            "--format",
            "protobuf",
        ],
    )

    mock_snap = {
        "categories": {
            "test": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "params": [],
                }
            ]
        }
    }

    mocker.patch("builtins.open", mock_open(read_data="{}"))
    mocker.patch("json.load", return_value=mock_snap)
    mocker.patch("os.makedirs")

    mock_to_protobuf = mocker.patch(
        "ml_framework_snapshots.export.to_protobuf", return_value="message Linear {}"
    )

    main()
    captured = capsys.readouterr()
    mock_to_protobuf.assert_called_once()
    assert "Exported 1 Protobuf definitions to out" in captured.out


def test_cli_export_unknown_format(mocker, capsys):
    """Function docstring."""
    from ml_framework_snapshots.cli import cmd_export
    import pytest

    mock_args = mocker.Mock()
    mock_args.command = "export"
    mock_args.input = "in.json"
    mock_args.out_dir = "out"
    mock_args.format = "unknown"
    mock_args.func = cmd_export
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=mock_args)

    mock_snap = {
        "categories": {
            "test": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "params": [],
                }
            ]
        }
    }

    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch("json.load", return_value=mock_snap)
    mocker.patch("os.makedirs")

    with pytest.raises(ValueError, match="Unknown format"):
        main()


def test_cmd_capture_wildcard(capsys) -> None:
    """Test function."""
    from ml_framework_snapshots.cli import cmd_capture
    import argparse
    from unittest.mock import patch

    args = argparse.Namespace(frameworks=["*"], include_nonpublic=False, out_dir="out")

    with patch(
        "ml_framework_snapshots.cli.extract_snapshot", return_value={"data": "fake"}
    ):
        with patch(
            "ml_framework_snapshots.cli.write_snapshot", return_value="out.json"
        ):
            cmd_capture(args)

    captured = capsys.readouterr()
    assert "Saved snapshot to out.json" in captured.out


def test_cmd_capture_unsupported(capsys) -> None:
    """Test function."""
    from ml_framework_snapshots.cli import cmd_capture
    import argparse
    from unittest.mock import patch

    args = argparse.Namespace(
        frameworks=["unsupported_fw", "torch"], include_nonpublic=False, out_dir="out"
    )

    with patch("ml_framework_snapshots.cli.extract_snapshot", return_value=None):
        cmd_capture(args)

    captured = capsys.readouterr()
    assert (
        "Warning: The following frameworks are unsupported and will be skipped: unsupported_fw"
        in captured.out
    )
    assert "Skipping torch" in captured.out


def test_cmd_capture_all(capsys) -> None:
    """Test function."""
    from ml_framework_snapshots.cli import cmd_capture
    import argparse
    from unittest.mock import patch

    args = argparse.Namespace(frameworks=[], include_nonpublic=False, out_dir="out")

    with patch(
        "ml_framework_snapshots.api.get_available_frameworks",
        return_value={"mock_fw": None},
    ):
        with patch(
            "ml_framework_snapshots.cli.extract_snapshot", return_value={"data": "fake"}
        ):
            with patch(
                "ml_framework_snapshots.cli.write_snapshot", return_value="out.json"
            ):
                cmd_capture(args)

    captured = capsys.readouterr()
    assert "Saved snapshot to out.json" in captured.out


def test_cmd_capture_missing(capsys) -> None:
    """Test function."""
    from ml_framework_snapshots.cli import cmd_capture
    import argparse
    from unittest.mock import patch

    args = argparse.Namespace(
        frameworks=["mock_fw"], include_nonpublic=False, out_dir="out"
    )

    with patch(
        "ml_framework_snapshots.api.get_available_frameworks",
        return_value={"mock_fw": None},
    ):
        with patch("ml_framework_snapshots.cli.extract_snapshot", return_value={}):
            cmd_capture(args)

    captured = capsys.readouterr()
    assert "Skipping mock_fw" in captured.out
