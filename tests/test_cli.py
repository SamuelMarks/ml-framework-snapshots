"""Module docstring."""

import os
from typing import Any


from ml_framework_snapshots.cli import main
from unittest.mock import mock_open


def test_cli_capture(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
    mocker.patch("sys.argv", ["ml-snapshots", "capture", "--out-dir", "test_out"])

    # Mock extract_snapshot and write_snapshot
    def mock_extract(fw: Any, include_nonpublic=False) -> Any:  # type: ignore
        """Function docstring.

        Args:
            fw: description
            include_nonpublic: description


        Returns:
            Return value.
        """
        if fw == "torch":
            return {"version": "1.0"}
        return {}

    mocker.patch(
        "ml_framework_snapshots.cli.extract_snapshot", side_effect=mock_extract
    )
    mocker.patch(
        "ml_framework_snapshots.cli.write_snapshot",
        return_value=os.path.join("test_out", "torch_v1.0.json"),
    )

    # Run CLI
    main()

    captured = capsys.readouterr()
    assert "Saved snapshot to test_out/torch_v1.0.json" in captured.out
    assert "Saved snapshot to test_out/torch_v1.0.json" in captured.out
    assert "Skipping jax, not installed" in captured.out


def test_cli_diff(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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


def test_cli_diff_changelog(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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


def test_cli_stubs(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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


def test_cli_export_openapi(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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


def test_cli_export_json_schema(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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


def test_cli_export_pydantic(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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


def test_cli_export_protobuf(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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


def test_cli_export_llm_prompt(mocker: Any, capsys: Any) -> None:
    """Test the export command with llm_prompt format.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
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
            "llm_prompt",
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

    mock_export = mocker.patch(
        "ml_framework_snapshots.export.export_llm_prompt_context",
        return_value="### `torch.nn.Linear`",
    )

    main()
    captured = capsys.readouterr()
    mock_export.assert_called_once()
    assert "Exported LLM prompt context" in captured.out


def test_cli_mcp(mocker: Any) -> None:
    """Test the mcp command invoker.

    Args:
        mocker: Parameter.
    """
    mocker.patch("sys.argv", ["ml-snapshots", "mcp"])
    mock_run = mocker.patch("ml_framework_snapshots.mcp_server.run_mcp_server")
    main()
    mock_run.assert_called_once()


def test_cli_offline_flag(mocker: Any, monkeypatch: Any) -> None:
    """Test passing --offline flag sets ML_SNAPSHOTS_OFFLINE environment variable.

    Args:
        mocker: Pytest mocker fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.delenv("ML_SNAPSHOTS_OFFLINE", raising=False)
    mocker.patch("sys.argv", ["ml-snapshots", "--offline", "mcp"])
    mocker.patch("ml_framework_snapshots.mcp_server.run_mcp_server")
    try:
        main()
        assert os.environ.get("ML_SNAPSHOTS_OFFLINE") == "1"
    finally:
        os.environ.pop("ML_SNAPSHOTS_OFFLINE", None)


def test_cli_export_unknown_format(mocker: Any, capsys: Any) -> None:
    """Function docstring.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
    from ml_framework_snapshots.cli import cmd_export
    import pytest

    mock_args = mocker.Mock()
    mock_args.command = "export"
    mock_args.input = "in.json"
    mock_args.out_dir = "out"
    mock_args.format = "unknown"
    mock_args.offline = False
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


def test_cli_export_non_container(mocker: Any, capsys: Any) -> None:
    """Test cmd_export when snapshot is neither a list nor a dict.

    Args:
        capsys: Pytest capsys fixture.
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.cli import cmd_export

    mock_args = mocker.Mock()
    mock_args.command = "export"
    mock_args.input = "in.json"
    mock_args.out_dir = "out"
    mock_args.format = "openapi"

    mocker.patch("builtins.open", mocker.mock_open(read_data="123"))
    mocker.patch("json.load", return_value=123)
    mocker.patch("os.makedirs")
    mock_to_openapi = mocker.patch(
        "ml_framework_snapshots.export.to_openapi", return_value={}
    )

    cmd_export(mock_args)
    mock_to_openapi.assert_called_once_with([])


def test_cli_export_list_snapshot(mocker: Any) -> None:
    """Test cmd_export when snapshot is a raw list of symbols.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.cli import cmd_export

    mock_args = mocker.Mock()
    mock_args.command = "export"
    mock_args.input = "in.json"
    mock_args.out_dir = "out"
    mock_args.format = "openapi"

    raw_list_snap = [
        {
            "name": "test_op",
            "api_path": "torch.test_op",
            "kind": "function",
            "params": [],
        }
    ]
    mocker.patch("builtins.open", mocker.mock_open(read_data="[]"))
    mocker.patch("json.load", return_value=raw_list_snap)
    mocker.patch("os.makedirs")
    mock_to_openapi = mocker.patch(
        "ml_framework_snapshots.export.to_openapi", return_value={}
    )

    cmd_export(mock_args)
    mock_to_openapi.assert_called_once()
    assert len(mock_to_openapi.call_args[0][0]) == 1
    assert mock_to_openapi.call_args[0][0][0].name == "test_op"


def test_cmd_capture_wildcard(capsys: Any) -> None:
    """Test function.

    Args:
        capsys: Parameter.
    """
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


def test_cmd_capture_unsupported(capsys: Any) -> None:
    """Test function.

    Args:
        capsys: Parameter.
    """
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


def test_cmd_capture_all(capsys: Any) -> None:
    """Test function.

    Args:
        capsys: Parameter.
    """
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


def test_cmd_capture_missing(capsys: Any) -> None:
    """Test function.

    Args:
        capsys: Parameter.
    """
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


def test_cmd_list_snapshots(capsys: Any) -> None:
    """Test cmd_list_snapshots prints available snapshots.

    Args:
        capsys: Parameter.
    """
    from ml_framework_snapshots.cli import cmd_list_snapshots
    import argparse

    args = argparse.Namespace()
    cmd_list_snapshots(args)
    captured = capsys.readouterr()
    assert "Available Snapshots:" in captured.out


def test_cmd_list_snapshots_empty(capsys: Any, mocker: Any) -> None:
    """Test cmd_list_snapshots when no snapshots exist.

    Args:
        capsys: Parameter.
        mocker: Parameter.
    """
    from ml_framework_snapshots.cli import cmd_list_snapshots
    import argparse

    mocker.patch("os.path.isdir", return_value=False)
    args = argparse.Namespace()
    cmd_list_snapshots(args)
    captured = capsys.readouterr()
    assert "No snapshot files found." in captured.out


def test_cmd_list_snapshots_duplicates(capsys: Any, mocker: Any, tmp_path: Any) -> None:
    """Test cmd_list_snapshots suppresses duplicate entries from identical paths.

    Args:
        capsys: Pytest capsys fixture.
        mocker: Pytest mocker fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_framework_snapshots.cli import cmd_list_snapshots
    import argparse

    fake_dir = tmp_path / "dup"
    fake_snap_dir = fake_dir / "snapshots"
    fake_snap_dir.mkdir(parents=True)
    test_json = fake_snap_dir / "test.json"
    test_json.write_text("{}", encoding="utf-8")

    mocker.patch(
        "ml_framework_snapshots.index.get_cache_dir",
        return_value=str(fake_dir),
    )
    # Mock snapshots_dirs to include duplicate entries of the same dir
    mocker.patch(
        "os.listdir",
        return_value=["test.json"],
    )
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch("os.path.getsize", return_value=42)

    args = argparse.Namespace()
    cmd_list_snapshots(args)
    captured = capsys.readouterr()
    assert "test.json" in captured.out


def test_cmd_pull_version(capsys: Any, mocker: Any, tmp_path: Any) -> None:
    """Test cmd_pull with explicit version and out-dir.

    Args:
        capsys: Pytest capsys fixture.
        mocker: Pytest mocker fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_framework_snapshots.cli import cmd_pull
    import argparse

    mock_retrieve = mocker.patch("urllib.request.urlretrieve")
    out_dir = str(tmp_path / "pulled")
    args = argparse.Namespace(target="torch@2.4.0", out_dir=out_dir)

    cmd_pull(args)
    captured = capsys.readouterr()
    assert "Pulling snapshot for torch (2.4.0)" in captured.out
    assert "Saved snapshot to" in captured.out
    mock_retrieve.assert_called_once()


def test_cmd_pull_latest_and_failure(capsys: Any, mocker: Any, tmp_path: Any) -> None:
    """Test cmd_pull with latest default version and failure handling.

    Args:
        capsys: Pytest capsys fixture.
        mocker: Pytest mocker fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_framework_snapshots.cli import cmd_pull
    import argparse

    mocker.patch("urllib.request.urlretrieve", side_effect=Exception("Network error"))
    mocker.patch(
        "ml_framework_snapshots.index.get_cache_dir",
        return_value=str(tmp_path / "cache"),
    )
    args = argparse.Namespace(target="jax", out_dir=None)

    cmd_pull(args)
    captured = capsys.readouterr()
    assert "Pulling snapshot for jax (latest)" in captured.out
    assert "Failed to download snapshot: Network error" in captured.out


def test_cmd_index_clear(capsys: Any, mocker: Any) -> None:
    """Test cmd_index with --clear flag.

    Args:
        capsys: Pytest capsys fixture.
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.cli import cmd_index
    import argparse

    mocker.patch("ml_framework_snapshots.index.clear_index", return_value=True)
    mocker.patch(
        "ml_framework_snapshots.index.get_index_db_path", return_value="/tmp/test.db"
    )
    args = argparse.Namespace(clear=True, rebuild=False, status=False)
    cmd_index(args)
    captured = capsys.readouterr()
    assert "Cleared index database at /tmp/test.db" in captured.out

    mocker.patch("ml_framework_snapshots.index.clear_index", return_value=False)
    cmd_index(args)
    captured2 = capsys.readouterr()
    assert "Failed to clear index database at /tmp/test.db" in captured2.out


def test_cmd_index_rebuild_and_status(capsys: Any, mocker: Any, tmp_path: Any) -> None:
    """Test cmd_index with --rebuild and --status.

    Args:
        capsys: Pytest capsys fixture.
        mocker: Pytest mocker fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_framework_snapshots.cli import cmd_index, main
    from ml_framework_snapshots.index import init_db
    import argparse

    db_file = str(tmp_path / "index.db")
    mocker.patch("ml_framework_snapshots.index.get_index_db_path", return_value=db_file)

    # 1. Status when absent
    args_status = argparse.Namespace(clear=False, rebuild=False, status=True)
    cmd_index(args_status)
    captured1 = capsys.readouterr()
    assert "Index database does not exist yet" in captured1.out

    # 2. Rebuild
    def mock_ensure_index(*args: Any, **kwargs: Any) -> Any:
        """Mock ensure_index initializing the test database.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Initialized SQLite database connection.
        """
        return init_db(db_file)

    mocker.patch(
        "ml_framework_snapshots.index.ensure_index", side_effect=mock_ensure_index
    )
    args_rebuild = argparse.Namespace(clear=False, rebuild=True, status=False)
    cmd_index(args_rebuild)
    captured2 = capsys.readouterr()
    assert "Rebuilt index with 0 symbols" in captured2.out

    # 3. Status when present
    cmd_index(args_status)
    captured3 = capsys.readouterr()
    assert "Index database:" in captured3.out
    assert "Total symbols:     0" in captured3.out

    # 4. Invoke via main() CLI dispatch
    mocker.patch(
        "sys.argv",
        ["ml_framework_snapshots", "pull", "torch@2.2.0", "--out-dir", str(tmp_path)],
    )
    mocker.patch("urllib.request.urlretrieve")
    main()
    captured_main = capsys.readouterr()
    assert "Pulling snapshot for torch (2.2.0)" in captured_main.out


def test_cmd_index_cache_clean(capsys: Any, mocker: Any) -> None:
    """Test index-cache --clean CLI subcommand.

    Args:
        capsys: Pytest capsys fixture.
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.cli import main

    mocker.patch(
        "ml_framework_snapshots.index.get_index_db_path",
        return_value="/tmp/test_cache.db",
    )
    mocker.patch("ml_framework_snapshots.index.clear_index", return_value=True)

    mocker.patch("sys.argv", ["ml_framework_snapshots", "index-cache", "--clean"])
    main()
    captured = capsys.readouterr()
    assert "Cleared index database at /tmp/test_cache.db" in captured.out
