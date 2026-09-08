"""Offline mode verification test suite.

Ensures that all framework snapshot lookups, MCP tools, ISA validators,
and CLI commands operate with complete network isolation (socket disabled)
and that zero database files are stored within the repository or package tree.
"""

import os
import socket
from typing import Any
import pytest

from ml_framework_snapshots.mcp_server import (
    check_mlir_op,
    check_rdna_instruction,
    check_sass_instruction,
    get_api_signature,
    get_framework_snapshot,
    search_apis,
)
from ml_framework_snapshots.index import get_index_db_path


@pytest.fixture(autouse=True)
def block_network(monkeypatch: Any) -> None:
    """Enforce strict network isolation across all offline tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    def guarded_socket(*args: Any, **kwargs: Any) -> Any:
        """Prevent socket creation during offline testing.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            OSError: Always raised to disable network access.
        """
        raise OSError("Network access disabled (offline-mode guarantee).")

    monkeypatch.setattr(socket, "socket", guarded_socket)


def test_offline_bundled_isa_and_dialects() -> None:
    """Verify that bundled SASS, RDNA, MLIR, and StableHLO snapshots load offline."""
    for fw in ["nvidia_sass", "amd_rdna", "mlir", "stablehlo"]:
        snap = get_framework_snapshot(fw)
        assert "categories" in snap
        assert len(snap["categories"]) > 0


def test_offline_mcp_validation_tools() -> None:
    """Verify that MCP validation tools run offline with zero network calls."""
    # SASS check
    sass_res = check_sass_instruction("FADD", operands=["R", "R", "R"], sm_arch="sm_80")
    assert sass_res["mnemonic_exists"] is True

    # RDNA check
    rdna_res = check_rdna_instruction("v_add_f32", operands=["VGPR", "VGPR", "VGPR"])
    assert rdna_res["mnemonic_exists"] is True

    # MLIR check
    mlir_res = check_mlir_op("arith.addf", operands_count=2)
    assert mlir_res["op_exists"] is True

    # StableHLO check
    hlo_res = check_mlir_op("stablehlo.dot_general", operands_count=2)
    assert hlo_res["op_exists"] is True


def test_offline_symbol_lookup_and_search() -> None:
    """Verify offline signature retrieval and keyword search."""
    sig = get_api_signature("nvidia_sass", "FADD")
    assert sig is not None
    assert (sig.get("name") or sig.get("mnemonic")) == "FADD"

    results = search_apis("amd_rdna", "v_fma")
    assert len(results) > 0


def test_zero_db_files_in_repo() -> None:
    """Verify no .db or .sqlite files exist in repository tree."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    forbidden_exts = (".db", ".db-wal", ".db-journal", ".sqlite", ".sqlite3")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip git directory and temporary test outputs
        if ".git" in dirpath or ".pytest_cache" in dirpath or "test_out" in dirpath:
            continue
        for fname in filenames:
            for ext in forbidden_exts:
                assert not fname.endswith(
                    ext
                ), f"Forbidden database file detected in repo: {os.path.join(dirpath, fname)}"


def test_index_db_stored_in_cache_dir_only(tmp_path: Any, monkeypatch: Any) -> None:
    """Verify that index database is strictly placed in local user cache directory.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    monkeypatch.setenv("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR", str(tmp_path))
    db_path = get_index_db_path()
    assert str(tmp_path) in db_path
    assert "src/ml_framework_snapshots" not in db_path


def test_wheel_package_contents_and_no_db(tmp_path: Any) -> None:
    """Verify built wheel packages all snapshot JSONs and excludes database files.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    import zipfile

    wheel_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist"
    )
    if not os.path.isdir(wheel_dir):
        return

    wheel_files = [f for f in os.listdir(wheel_dir) if f.endswith(".whl")]
    if not wheel_files:
        return

    wheel_path = os.path.join(wheel_dir, sorted(wheel_files)[-1])
    with zipfile.ZipFile(wheel_path, "r") as zf:
        namelist = zf.namelist()
        # Assert no database files
        for name in namelist:
            assert not name.endswith(
                (".db", ".sqlite", ".sqlite3")
            ), f"Database file inside wheel: {name}"

        # Assert JSON snapshots are present
        json_files = [n for n in namelist if n.endswith(".json")]
        assert len(json_files) >= 4

        # Unpack and verify out-of-the-box operation
        unpack_dir = tmp_path / "wheel_unpacked"
        zf.extractall(unpack_dir)

        # Verify JSONs exist in unpacked site-packages
        unpacked_jsons = list(unpack_dir.glob("**/*.json"))
        assert len(unpacked_jsons) >= 4

        # Verify zero .db or .sqlite files
        unpacked_dbs = list(unpack_dir.glob("**/*.db*")) + list(
            unpack_dir.glob("**/*.sqlite*")
        )
        assert len(unpacked_dbs) == 0


def test_offline_cli_execution(tmp_path: Any, monkeypatch: Any) -> None:
    """Verify CLI commands execute 100% offline with zero network calls.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    import sys
    from ml_framework_snapshots import cli

    # 1. check-sass
    monkeypatch.setattr(
        sys,
        "argv",
        ["ml_framework_snapshots", "check-sass", "FADD", "--operands", "R,R,R"],
    )
    cli.main()

    # 2. check-rdna
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ml_framework_snapshots",
            "check-rdna",
            "v_add_f32",
            "--operands",
            "VGPR,VGPR,VGPR",
        ],
    )
    cli.main()

    # 3. check-mlir
    monkeypatch.setattr(
        sys, "argv", ["ml_framework_snapshots", "check-mlir", "arith.muli"]
    )
    cli.main()

    # 4. check-stablehlo
    monkeypatch.setattr(
        sys,
        "argv",
        ["ml_framework_snapshots", "check-stablehlo", "stablehlo.dot_general"],
    )
    cli.main()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(cli.__file__)))
    json_path = os.path.join(base_dir, "frameworks", "nvidia_sass_exhaustive.json")

    # 5. export
    export_out = tmp_path / "export_dir"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ml_framework_snapshots",
            "export",
            "--input",
            json_path,
            "--out-dir",
            str(export_out),
            "--format",
            "pydantic",
        ],
    )
    cli.main()
    assert export_out.exists()

    # 6. generate-stubs
    stubs_out = tmp_path / "stubs_dir"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ml_framework_snapshots",
            "generate-stubs",
            "--input",
            json_path,
            "--out-dir",
            str(stubs_out),
        ],
    )
    cli.main()
    assert stubs_out.exists()

    # 7. diff
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ml_framework_snapshots",
            "diff",
            json_path,
            json_path,
        ],
    )
    cli.main()
