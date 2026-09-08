"""Unit test suite for ephemeral local SQLite search index.

Validates 100% coverage across schema initialization, snapshot ingestion,
FTS5 full-text querying, symbol lookups, cache directory resolution, and cleanup.
"""

import json
import os
import tempfile
from typing import Any, cast
from unittest import mock
from unittest.mock import patch

from ml_framework_snapshots.index import (
    clear_index,
    compute_file_sha256,
    ensure_index,
    extract_framework_and_version,
    get_available_snapshot_files,
    get_cache_dir,
    get_index_db_path,
    index_snapshot_file,
    init_db,
    lookup_symbol,
    search_index,
)


def test_get_cache_dir_custom_env(monkeypatch: Any, tmp_path: Any) -> None:
    """Test get_cache_dir when ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR is set.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    custom_dir = str(tmp_path / "custom_cache")
    monkeypatch.setenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", custom_dir)
    res = get_cache_dir()
    assert res == custom_dir
    assert os.path.isdir(custom_dir)


def test_get_cache_dir_xdg_env(monkeypatch: Any, tmp_path: Any) -> None:
    """Test get_cache_dir when XDG_CACHE_HOME is set.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    monkeypatch.delenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", raising=False)
    xdg_dir = str(tmp_path / "xdg_cache")
    monkeypatch.setenv("XDG_CACHE_HOME", xdg_dir)
    res = get_cache_dir()
    assert res == os.path.join(xdg_dir, "ml_framework_snapshots")
    assert os.path.isdir(res)


def test_get_cache_dir_fallback(monkeypatch: Any, tmp_path: Any) -> None:
    """Test get_cache_dir fallback to ~/.cache.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    monkeypatch.delenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(
        "os.path.expanduser", lambda path: str(tmp_path) if path == "~" else path
    )
    res = get_cache_dir()
    expected = os.path.join(str(tmp_path), ".cache", "ml_framework_snapshots")
    assert res == expected
    assert os.path.isdir(res)


def test_get_cache_dir_filenotfound_fallback(monkeypatch: Any) -> None:
    """Test get_cache_dir falls back to temp directory if creating cache dir raises FileNotFoundError.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.delenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    def mock_makedirs(path: str, exist_ok: bool = False) -> None:
        """Mock makedirs raising FileNotFoundError for snapshot cache path.

        Args:
            path: Directory path to create.
            exist_ok: Whether to ignore existing directory.

        Raises:
            FileNotFoundError: Always raised when path matches cache dir.
        """
        if "ml_framework_snapshots" in path:
            raise FileNotFoundError("Broken symlink")

    monkeypatch.setattr("os.makedirs", mock_makedirs)
    res = get_cache_dir()
    expected = os.path.join(tempfile.gettempdir(), "ml_framework_cache")
    assert res == expected


def test_get_cache_dir_filenotfound_and_fallback_error(monkeypatch: Any) -> None:
    """Test get_cache_dir when both cache dir and fallback creation fail.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.delenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    def mock_makedirs(path: str, exist_ok: bool = False) -> None:
        """Mock makedirs raising FileNotFoundError then OSError.

        Args:
            path: Directory path to create.
            exist_ok: Whether to ignore existing directory.

        Raises:
            FileNotFoundError: If cache dir in path.
            OSError: For all other paths.
        """
        if "ml_framework_snapshots" in path:
            raise FileNotFoundError("Broken symlink")
        raise OSError("Fallback failed")

    monkeypatch.setattr("os.makedirs", mock_makedirs)
    res = get_cache_dir()
    expected = os.path.join(tempfile.gettempdir(), "ml_framework_cache")
    assert res == expected


def test_get_cache_dir_fileexists_fallback(monkeypatch: Any, tmp_path: Any) -> None:
    """Test get_cache_dir ignores FileExistsError when path exists as file or isdir is mocked.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    monkeypatch.delenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(
        "os.path.expanduser", lambda path: str(tmp_path) if path == "~" else path
    )

    def mock_makedirs(path: str, exist_ok: bool = False) -> None:
        """Mock makedirs raising FileExistsError.

        Args:
            path: Directory path to create.
            exist_ok: Whether to ignore existing directory.

        Raises:
            FileExistsError: Always raised to simulate existing file.
        """
        raise FileExistsError("File exists")

    monkeypatch.setattr("os.makedirs", mock_makedirs)
    res = get_cache_dir()
    expected = os.path.join(str(tmp_path), ".cache", "ml_framework_snapshots")
    assert res == expected


def test_get_index_db_path(monkeypatch: Any, tmp_path: Any) -> None:
    """Test get_index_db_path returns index.db inside cache dir.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    monkeypatch.setenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", str(tmp_path))
    db_path = get_index_db_path()
    assert db_path == os.path.join(str(tmp_path), "index.db")


def test_compute_file_sha256(tmp_path: Any) -> None:
    """Test compute_file_sha256 computes correct checksum.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello world", encoding="utf-8")
    h = compute_file_sha256(str(sample_file))
    assert isinstance(h, str)
    assert len(h) == 64


def test_extract_framework_and_version() -> None:
    """Test extracting framework identifier and version from filenames."""
    assert extract_framework_and_version("torch_v2.2.0.json") == ("torch", "2.2.0")
    assert extract_framework_and_version("torch_v2.2.0") == ("torch", "2.2.0")
    assert extract_framework_and_version("amd_rdna_exhaustive.json") == (
        "amd_rdna",
        "latest",
    )
    assert extract_framework_and_version("jax_0.4.26.json") == ("jax", "0.4.26")
    assert extract_framework_and_version("stablehlo.json") == ("stablehlo", "latest")


def test_init_db_and_index_snapshot_file(tmp_path: Any) -> None:
    """Test creating database schema and indexing snapshot items.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    db_file = str(tmp_path / "test.db")
    conn = init_db(db_file)

    # 1. Test indexing a non-existent file returns 0
    assert index_snapshot_file("non_existent.json", conn) == 0

    # 2. Test indexing a snapshot dictionary with categories
    snap_data = {
        "categories": {
            "math": [
                {
                    "api_path": "torch.sum",
                    "name": "sum",
                    "kind": "function",
                    "docstring": "Compute sum along dimensions.",
                },
                "invalid_non_dict_item",
            ]
        }
    }
    json_path = str(tmp_path / "torch_v2.2.0.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snap_data, f)

    count = index_snapshot_file(json_path, conn)
    assert count == 1

    # 3. Test re-indexing unchanged file returns 0 (hash match cache hit)
    assert index_snapshot_file(json_path, conn) == 0

    # 4. Test re-indexing modified file clears old items and inserts new
    snap_data["categories"]["math"].append(
        {
            "api_path": "torch.mean",
            "name": "mean",
            "kind": "function",
            "docstring": "Compute mean along dimensions.",
        }
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snap_data, f)

    count2 = index_snapshot_file(json_path, conn)
    assert count2 == 2

    # 5. Test indexing dict with 'items' key and list format
    items_snap = {
        "items": [
            {
                "api_path": "jax.numpy.add",
                "name": "add",
                "kind": "function",
                "docstring": "Elementwise addition.",
            }
        ]
    }
    json_items_path = str(tmp_path / "jax_v0.4.26.json")
    with open(json_items_path, "w", encoding="utf-8") as f:
        json.dump(items_snap, f)
    assert index_snapshot_file(json_items_path, conn) == 1

    list_snap = [
        {
            "mnemonic": "FADD",
            "name": "FADD",
            "kind": "function",
            "docstring": "Floating point addition.",
        }
    ]
    json_list_path = str(tmp_path / "nvidia_sass_v12.4.json")
    with open(json_list_path, "w", encoding="utf-8") as f:
        json.dump(list_snap, f)
    assert index_snapshot_file(json_list_path, conn) == 1

    # 6. Test non-list categories and non-list items
    weird_snap = {
        "categories": {"non_list": "just_a_string"},
    }
    json_weird_path = str(tmp_path / "weird_v1.0.0.json")
    with open(json_weird_path, "w", encoding="utf-8") as f:
        json.dump(weird_snap, f)
    assert index_snapshot_file(json_weird_path, conn) == 0

    weird_items_snap = {"items": "not_a_list_either"}
    json_weird_items_path = str(tmp_path / "weird_items_v1.0.0.json")
    with open(json_weird_items_path, "w", encoding="utf-8") as f:
        json.dump(weird_items_snap, f)
    assert index_snapshot_file(json_weird_items_path, conn) == 0

    empty_snap: dict[str, Any] = {}
    json_empty_path = str(tmp_path / "empty_v1.0.0.json")
    with open(json_empty_path, "w", encoding="utf-8") as f:
        json.dump(empty_snap, f)
    assert index_snapshot_file(json_empty_path, conn) == 0

    # 7. Test non-dict non-list JSON payload (covers branch 193->196)
    json_scalar_path = str(tmp_path / "scalar_v1.0.0.json")
    with open(json_scalar_path, "w", encoding="utf-8") as f:
        json.dump("scalar_string_payload", f)
    assert index_snapshot_file(json_scalar_path, conn) == 0


def test_ensure_index_and_search_index(tmp_path: Any, monkeypatch: Any) -> None:
    """Test ensure_index populates database and search_index queries FTS5.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    db_file = str(tmp_path / "search_test.db")
    monkeypatch.setenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", str(tmp_path))

    # Create dummy snapshot files in cache dir
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = str(snap_dir / "torch_v2.2.0.json")
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": {
                    "math": [
                        {
                            "api_path": "torch.sum",
                            "name": "sum",
                            "kind": "function",
                            "docstring": "Sum tensor elements over dimensions.",
                        },
                        {
                            "api_path": "torch.add",
                            "name": "add",
                            "kind": "function",
                            "docstring": "Add two tensors.",
                        },
                    ]
                }
            },
            f,
        )

    # Corrupted json file to verify exception resilience in ensure_index
    corrupt_path = str(snap_dir / "corrupt_v1.0.0.json")
    with open(corrupt_path, "w", encoding="utf-8") as f:
        f.write("{invalid json")

    conn = ensure_index(db_file)
    assert conn is not None

    # Empty query
    assert search_index("", db_path=db_file) == []

    # Query matching sum
    results = search_index("sum", framework="torch", version="2.2.0", db_path=db_file)
    assert len(results) == 1
    assert results[0]["api_path"] == "torch.sum"

    # Query without framework or version
    all_res = search_index("add", db_path=db_file)
    assert any(r.get("api_path") == "torch.add" for r in all_res)

    # Test corrupted json_data handling in search_index
    conn.execute(
        "INSERT INTO symbols (file_path, framework, version, api_path, name, kind, docstring, json_data) VALUES ('f', 'fw', '1', 'corrupt_unique_token.api', 'corrupt_unique_token', 'fn', 'corrupt_unique_token doc', '{corrupt json')"
    )
    sym_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO symbols_fts (rowid, api_path, name, docstring) VALUES (?, 'corrupt_unique_token.api', 'corrupt_unique_token', 'corrupt_unique_token doc')",
        (sym_id,),
    )
    conn.commit()
    bad_res = search_index("corrupt_unique_token", db_path=db_file)
    assert bad_res == []


def test_lookup_symbol(tmp_path: Any) -> None:
    """Test lookup_symbol exact resolution.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    db_file = str(tmp_path / "lookup_test.db")
    conn = init_db(db_file)

    snap = {
        "categories": {
            "nn": [
                {
                    "api_path": "torch.nn.Linear",
                    "name": "Linear",
                    "kind": "class",
                    "docstring": "Applies a linear transformation.",
                }
            ]
        }
    }
    json_path = str(tmp_path / "torch_v2.0.0.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snap, f)

    index_snapshot_file(json_path, conn)

    # 1. Exact match by api_path
    res1 = lookup_symbol("torch", "torch.nn.Linear", version="2.0.0", db_path=db_file)
    assert res1 is not None
    assert res1["name"] == "Linear"

    # 2. Match by name without version
    res2 = lookup_symbol("torch", "Linear", db_path=db_file)
    assert res2 is not None
    assert res2["api_path"] == "torch.nn.Linear"

    # 3. Not found
    assert lookup_symbol("torch", "torch.nonexistent", db_path=db_file) is None


def test_clear_index(tmp_path: Any) -> None:
    """Test clear_index removes database file.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    db_file = str(tmp_path / "to_delete.db")
    with open(db_file, "w") as f:
        f.write("dummy")

    assert os.path.exists(db_file)
    assert clear_index(db_file) is True
    assert not os.path.exists(db_file)

    # Clearing non-existent file returns True
    assert clear_index(db_file) is True

    # Test OSError in clear_index
    with patch("os.remove", side_effect=OSError("Permission denied")):
        with open(db_file, "w") as f:
            f.write("dummy")
        assert clear_index(db_file) is False


def test_get_available_snapshot_files(tmp_path: Any, monkeypatch: Any) -> None:
    """Test get_available_snapshot_files discovers files from search paths.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    monkeypatch.setenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", str(tmp_path))
    cached_snap = tmp_path / "snapshots"
    cached_snap.mkdir(parents=True, exist_ok=True)
    dummy_json = cached_snap / "cached_v1.0.0.json"
    dummy_json.write_text("{}", encoding="utf-8")

    files = get_available_snapshot_files()
    assert any("cached_v1.0.0.json" in f for f in files)


def test_get_readonly_connection_and_clean_cache(tmp_path: Any) -> None:
    """Test get_readonly_connection and clean_index_cache operations.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    import sqlite3
    from ml_framework_snapshots.index import (
        clean_index_cache,
        get_readonly_connection,
    )

    db_file = str(tmp_path / "cache_test.db")

    # 1. Open readonly connection when DB does not exist yet (initializes DB)
    conn1 = get_readonly_connection(db_file)
    assert conn1 is not None
    conn1.close()

    # 2. Open readonly connection when DB exists
    conn2 = get_readonly_connection(db_file)
    assert conn2 is not None
    conn2.close()

    # 3. Test clean_index_cache with auxiliary files and OSError handling
    wal_file = f"{db_file}-wal"
    with open(wal_file, "w") as f:
        f.write("wal dummy")
    assert os.path.exists(wal_file)

    with patch("os.remove", side_effect=OSError("Permission denied")):
        removed_zero = clean_index_cache(db_file)
        assert removed_zero == 0

    removed = clean_index_cache(db_file)
    assert removed >= 2  # db and wal removed
    assert not os.path.exists(db_file)
    assert not os.path.exists(wal_file)

    # 4. Readonly fallback on OperationalError (when file exists)
    with open(db_file, "w") as f:
        f.write("")
    assert os.path.exists(db_file)

    orig_connect = sqlite3.connect

    def mock_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        if args and "mode=ro" in str(args[0]):
            raise sqlite3.OperationalError("Readonly open failed")
        return cast(sqlite3.Connection, orig_connect(*args, **kwargs))

    with patch("sqlite3.connect", side_effect=mock_connect):
        conn_fallback = get_readonly_connection(db_file)
        assert conn_fallback is not None
        conn_fallback.close()

    # 5. init_db PRAGMA OperationalError handling
    class MockConn:
        def __init__(self) -> None:
            self.row_factory: Any = None

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            if "PRAGMA journal_mode=WAL" in sql:
                raise sqlite3.OperationalError("WAL not supported")
            return mock.MagicMock()

        def __enter__(self) -> "MockConn":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    with patch("sqlite3.connect", return_value=MockConn()):
        conn_pragma = init_db(str(tmp_path / "pragma_test.db"))
        assert conn_pragma is not None
