"""Module docstring."""

from typing import Any
from unittest.mock import patch
from ml_framework_snapshots.cli import resolve_snapshot_path


def test_resolve_snapshot_path_existing_file(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    file_path = tmp_path / "my_snapshot.json"
    file_path.touch()

    resolved = resolve_snapshot_path(str(file_path))
    assert resolved == str(file_path)


def test_resolve_snapshot_path_append_json(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    file_path = tmp_path / "my_snapshot.json"
    file_path.touch()

    # Try resolving without .json extension
    base_path = str(tmp_path / "my_snapshot")
    resolved = resolve_snapshot_path(base_path)
    assert resolved == str(file_path)


def test_resolve_snapshot_path_fallback_to_repo(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    # Mock __file__ so repo root points to our temp directory
    pkg_dir = tmp_path / "src" / "ml_framework_snapshots"
    pkg_dir.mkdir(parents=True)

    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    target_file = snapshots_dir / "jax_v0.4.30.json"
    target_file.touch()

    with patch("ml_framework_snapshots.cli.__file__", str(pkg_dir / "cli.py")):
        # Providing just the basename
        resolved = resolve_snapshot_path("jax_v0.4.30.json")
        assert resolved == str(target_file)

        # Providing a path that doesn't exist locally, but basename matches
        resolved = resolve_snapshot_path("./some_other_dir/jax_v0.4.30.json")
        assert resolved == str(target_file)

        # Providing basename without .json
        resolved = resolve_snapshot_path("jax_v0.4.30")
        assert resolved == str(target_file)

        # Providing just prefix
        resolved = resolve_snapshot_path("jax")
        assert resolved == str(target_file)


def test_resolve_snapshot_path_not_found(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    # Should just return the input path if nothing works
    with patch("ml_framework_snapshots.cli.__file__", "/tmp/does_not_exist/cli.py"):
        resolved = resolve_snapshot_path("missing_framework")
        assert resolved == "missing_framework"


def test_resolve_multiple_matches(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    pkg_dir = tmp_path / "src" / "ml_framework_snapshots"
    pkg_dir.mkdir(parents=True)

    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    (snapshots_dir / "torch-1.12.json").touch()
    (snapshots_dir / "torch-2.0.json").touch()

    with patch("ml_framework_snapshots.cli.__file__", str(pkg_dir / "cli.py")):
        res = resolve_snapshot_path("torch")
        assert res == str(snapshots_dir / "torch-2.0.json")


def test_resolve_no_matches_in_existing_dir(tmp_path: Any) -> None:
    """Test when snapshots_dir exists but no glob matches are found.

    Args:
        tmp_path: Parameter.
    """
    pkg_dir = tmp_path / "src" / "ml_framework_snapshots"
    pkg_dir.mkdir(parents=True)

    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    with patch("ml_framework_snapshots.cli.__file__", str(pkg_dir / "cli.py")):
        res = resolve_snapshot_path("nonexistent_framework")
        assert res == "nonexistent_framework"
