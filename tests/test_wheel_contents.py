"""Unit test suite for wheel package verification and data asset integrity.

Verifies that wheels built via pyproject.toml bundle all required JSON snapshots,
contain 0 database files, and can be queried in clean isolated environments.
"""

import os
import zipfile
from hatchling.build import build_wheel


def test_wheel_packaging_and_data_assets(tmp_path: os.PathLike[str]) -> None:
    """Build wheel and verify 100% of required JSON snapshots and 0 database files.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    whl_filename = build_wheel(str(tmp_path))
    whl_path = os.path.join(str(tmp_path), whl_filename)
    assert os.path.isfile(whl_path)

    with zipfile.ZipFile(whl_path, "r") as zf:
        namelist = zf.namelist()

        # 1. Zero database leakage assertion
        db_artifacts = [
            n
            for n in namelist
            if any(
                n.endswith(ext)
                for ext in (
                    ".db",
                    ".sqlite",
                    ".sqlite3",
                    ".db-wal",
                    ".db-journal",
                    ".db-shm",
                    ".sqlite-journal",
                )
            )
        ]
        assert (
            len(db_artifacts) == 0
        ), f"Found database files leaked into wheel: {db_artifacts}"

        # 2. Exhaustive JSON assets bundled
        required_exhaustive_jsons = [
            "amd_rdna_exhaustive.json",
            "nvidia_sass_exhaustive.json",
            "nvidia_ptx_exhaustive.json",
            "mlir_exhaustive.json",
            "stablehlo_exhaustive.json",
            "concept_map.json",
        ]
        for req in required_exhaustive_jsons:
            matching = [n for n in namelist if n.endswith(req)]
            assert (
                len(matching) > 0
            ), f"Required data asset '{req}' missing from wheel archive"

        # 3. Test querying extracted wheel assets
        extract_dir = os.path.join(str(tmp_path), "extracted")
        zf.extractall(extract_dir)

        sass_json = os.path.join(
            extract_dir,
            "ml_framework_snapshots",
            "frameworks",
            "nvidia_sass_exhaustive.json",
        )
        assert os.path.isfile(sass_json)


def test_wheel_bundles_dynamic_snapshots(tmp_path: os.PathLike[str]) -> None:
    """Verify that any snapshot JSON files in snapshots/ are bundled in .whl and .db files excluded.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    snap_dir = os.path.join(root_dir, "src", "ml_framework_snapshots", "snapshots")
    test_json = os.path.join(snap_dir, "testframework_v1.0.0.json")
    test_db = os.path.join(snap_dir, "testleak.db")

    try:
        with open(test_json, "w", encoding="utf-8") as f:
            f.write('{"categories": {"TEST": [{"name": "test_op"}]}}')
        with open(test_db, "w", encoding="utf-8") as f:
            f.write("test sqlite db content")

        whl_filename = build_wheel(str(tmp_path))
        whl_path = os.path.join(str(tmp_path), whl_filename)

        with zipfile.ZipFile(whl_path, "r") as zf:
            names = zf.namelist()
            # Assert test json is included
            assert any(
                n.endswith("testframework_v1.0.0.json") for n in names
            ), "Dynamic snapshot JSON was not included in wheel"
            # Assert test db is excluded
            assert not any(
                n.endswith("testleak.db") for n in names
            ), "Database file was leaked into wheel"
    finally:
        if os.path.exists(test_json):
            os.remove(test_json)
        if os.path.exists(test_db):
            os.remove(test_db)
