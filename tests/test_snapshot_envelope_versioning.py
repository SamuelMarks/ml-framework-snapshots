"""Unit tests for snapshot envelope versioning and dual format support across MLIR, StableHLO, NVIDIA, and AMD frameworks."""

import json
import os
import tempfile
from unittest import mock

from ml_framework_snapshots.api import (
    extract_snapshot,
    get_pkg_version,
    validate_snapshot_envelope,
)
from ml_framework_snapshots.frameworks.amd_rdna import _load_exhaustive_rdna
from ml_framework_snapshots.frameworks.mlir import _load_mlir_exhaustive
from ml_framework_snapshots.frameworks.nvidia_ptx import _load_exhaustive_ptx
from ml_framework_snapshots.frameworks.nvidia_sass import _load_exhaustive_sass
from ml_framework_snapshots.frameworks.stablehlo import _load_stablehlo_exhaustive
from ml_framework_snapshots.index import (
    extract_framework_and_version,
    index_snapshot_file,
    init_db,
)
from ml_framework_snapshots.models import SnapshotEnvelope
from ml_framework_snapshots.tools.build_stablehlo_snapshot import main as stablehlo_main
from ml_framework_snapshots.tools.scrape_amd_rdna import main as amd_rdna_main
from ml_framework_snapshots.tools.scrape_mlir import main as mlir_main
from ml_framework_snapshots.tools.scrape_nvidia_ptx import main as ptx_main
from ml_framework_snapshots.tools.scrape_nvidia_sass import main as sass_main


def test_snapshot_envelope_new_fields() -> None:
    """Test that SnapshotEnvelope accepts operations and instructions fields."""
    env = SnapshotEnvelope(
        target="stablehlo",
        version="1.9.0",
        upstream_version="1.9.0",
        upstream_commit="v1.9.0",
        source_type="tablegen",
        operations=[{"name": "stablehlo.add"}],
        instructions=[{"mnemonic": "add"}],
    )
    assert env.target == "stablehlo"
    assert env.version == "1.9.0"
    assert env.operations == [{"name": "stablehlo.add"}]
    assert env.instructions == [{"mnemonic": "add"}]


def test_framework_loaders_dual_format_support() -> None:
    """Test that all framework loaders support both enveloped dicts and bare lists."""
    dummy_op = {
        "class_name": "TestOp",
        "name": "test_op",
        "api_path": "test.op",
        "operands": [],
        "attributes": [],
        "results": [],
    }
    dummy_instr = {
        "mnemonic": "test_instr",
        "category": "arithmetic",
        "operands": [],
        "min_sm": "sm_50",
    }

    # StableHLO
    with mock.patch("os.path.exists", return_value=True):
        # Enveloped with operations
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps({"operations": [dummy_op]})),
        ):
            res = _load_stablehlo_exhaustive()
            assert len(res) == 1
            assert res[0].name == "TestOp"

        # Legacy bare list
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps([dummy_op])),
        ):
            res = _load_stablehlo_exhaustive()
            assert len(res) == 1
            assert res[0].name == "TestOp"

    # MLIR
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps({"operations": [dummy_op]})),
        ):
            res = _load_mlir_exhaustive()
            assert len(res) == 1
            assert res[0].name == "TestOp"

        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps([dummy_op])),
        ):
            res = _load_mlir_exhaustive()
            assert len(res) == 1
            assert res[0].name == "TestOp"

    # NVIDIA PTX
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps({"instructions": [dummy_instr]})),
        ):
            res_ptx = _load_exhaustive_ptx()
            assert len(res_ptx) == 1
            assert res_ptx[0]["mnemonic"] == "test_instr"

        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps([dummy_instr])),
        ):
            res_ptx = _load_exhaustive_ptx()
            assert len(res_ptx) == 1

    # NVIDIA SASS
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps({"instructions": [dummy_instr]})),
        ):
            res_sass = _load_exhaustive_sass()
            assert len(res_sass) == 1
            assert res_sass[0]["mnemonic"] == "test_instr"

    # AMD RDNA
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps({"instructions": [dummy_instr]})),
        ):
            res_rdna = _load_exhaustive_rdna()
            assert len(res_rdna) == 1
            assert res_rdna[0]["mnemonic"] == "test_instr"


def test_get_pkg_version_canonical_defaults() -> None:
    """Test get_pkg_version returns canonical semver/spec versions for hardware and IR targets."""
    with mock.patch("os.path.exists", return_value=False):
        assert get_pkg_version("stablehlo") == "1.9.0"
        assert get_pkg_version("mlir") == "19.1.0"
        assert get_pkg_version("nvidia_ptx") == "8.5"
        assert get_pkg_version("nvidia_sass") == "12.6.0"
        assert get_pkg_version("amd_rdna") == "19.1.0"
        assert get_pkg_version("html_dsl") == "0.0.2"
        assert get_pkg_version("latex_dsl") == "0.0.2"
        assert get_pkg_version("tikz") == "0.0.2"


def test_extract_snapshot_provenance_metadata() -> None:
    """Test extract_snapshot populates all required provenance metadata fields."""
    for target in (
        "nvidia_ptx",
        "nvidia_sass",
        "amd_rdna",
        "stablehlo",
        "mlir",
        "html_dsl",
        "latex_dsl",
        "tikz",
    ):
        snap = extract_snapshot(target)
        assert snap["target"] == target
        assert "version" in snap
        assert snap["version"] is not None
        assert "source_type" in snap
        assert "upstream_commit" in snap
        assert "upstream_version" in snap

        # Hardware targets must have supported_microarchitectures
        if target in ("nvidia_ptx", "nvidia_sass", "amd_rdna"):
            assert "supported_microarchitectures" in snap
            assert len(snap["supported_microarchitectures"]) > 0

        # Validate against schema
        envelope = validate_snapshot_envelope(snap)
        assert envelope.target == target


def test_scraper_cli_arguments_and_envelope_emission() -> None:
    """Test scraper CLI tools with custom version and tag parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # StableHLO
        out_snap = os.path.join(tmpdir, "stablehlo_v2.0.0.json")
        out_exh = os.path.join(tmpdir, "stablehlo_exh.json")
        with mock.patch(
            "ml_framework_snapshots.tools.build_stablehlo_snapshot.extract_ops",
            return_value=[{"name": "custom.op"}],
        ):
            stablehlo_main(
                [
                    "--version",
                    "2.0.0",
                    "--tag",
                    "v2.0.0",
                    "--out-dir",
                    tmpdir,
                    "--exhaustive-path",
                    out_exh,
                ]
            )
            assert os.path.exists(out_snap)
            assert os.path.exists(out_exh)
            with open(out_snap, "r") as f:
                snap_data = json.load(f)
                assert snap_data["version"] == "2.0.0"
                assert snap_data["upstream_commit"] == "v2.0.0"

        # MLIR
        out_mlir = os.path.join(tmpdir, "mlir_custom.json")
        with mock.patch(
            "ml_framework_snapshots.tools.scrape_mlir.fetch_html", return_value=""
        ):
            with mock.patch(
                "ml_framework_snapshots.tools.scrape_mlir.inspect_mlir_python_module",
                return_value=[],
            ):
                with mock.patch(
                    "ml_framework_snapshots.tools.scrape_mlir.scrape_stablehlo",
                    return_value=[],
                ):
                    mlir_main(
                        [
                            "--version",
                            "20.1.0",
                            "--tag",
                            "llvmorg-20.1.0",
                            "--output-path",
                            out_mlir,
                        ]
                    )
                    assert os.path.exists(out_mlir)
                    with open(out_mlir, "r") as f:
                        mlir_data = json.load(f)
                        assert mlir_data["version"] == "20.1.0"
                        assert mlir_data["upstream_commit"] == "llvmorg-20.1.0"

        # NVIDIA PTX
        out_ptx = os.path.join(tmpdir, "ptx_custom.json")
        with mock.patch(
            "ml_framework_snapshots.tools.scrape_nvidia_ptx.fetch_nvptx_td_file",
            return_value="",
        ):
            ptx_main(
                [
                    "--isa-version",
                    "8.6",
                    "--cuda-version",
                    "12.8.0",
                    "--output-path",
                    out_ptx,
                ]
            )
            assert os.path.exists(out_ptx)
            with open(out_ptx, "r") as f:
                ptx_data = json.load(f)
                assert ptx_data["version"] == "8.6"
                assert ptx_data["upstream_version"] == "cuda-12.8.0"

        # NVIDIA SASS
        out_sass = os.path.join(tmpdir, "sass_custom.json")
        input_sass = os.path.join(tmpdir, "sass_in.json")
        with open(input_sass, "w") as f:
            json.dump({"1.MOV": {"parsed": {"base_name": "MOV", "operands": []}}}, f)
        sass_main(
            [
                "--cuda-version",
                "12.8.0",
                "--input-path",
                input_sass,
                "--output-path",
                out_sass,
            ]
        )
        assert os.path.exists(out_sass)
        with open(out_sass, "r") as f:
            sass_data = json.load(f)
            assert sass_data["version"] == "12.8.0"
            assert sass_data["upstream_commit"] == "cuda-12.8.0-toolkit"

        # AMD RDNA
        out_rdna = os.path.join(tmpdir, "rdna_custom.json")
        with mock.patch(
            "ml_framework_snapshots.tools.scrape_amd_rdna.fetch_td_file",
            return_value="",
        ):
            amd_rdna_main(
                [
                    "--llvm-version",
                    "20.1.0",
                    "--rocm-version",
                    "6.3.0",
                    "--output-path",
                    out_rdna,
                ]
            )
            assert os.path.exists(out_rdna)
            with open(out_rdna, "r") as f:
                rdna_data = json.load(f)
                assert rdna_data["version"] == "20.1.0"
                assert rdna_data["upstream_version"] == "rocm-6.3.0"


def test_index_extract_framework_and_version() -> None:
    """Test index filename parsing for diverse version and multi-word target patterns."""
    assert extract_framework_and_version("stablehlo_v1.9.0.json") == (
        "stablehlo",
        "1.9.0",
    )
    assert extract_framework_and_version("nvidia_ptx_v8.5.json") == (
        "nvidia_ptx",
        "8.5",
    )
    assert extract_framework_and_version("nvidia_sass_v12.6.0.json") == (
        "nvidia_sass",
        "12.6.0",
    )
    assert extract_framework_and_version("amd_rdna_v19.1.0.json") == (
        "amd_rdna",
        "19.1.0",
    )
    assert extract_framework_and_version("html_dsl_v0.0.2.json") == (
        "html_dsl",
        "0.0.2",
    )
    assert extract_framework_and_version("latex_dsl_v0.0.2.json") == (
        "latex_dsl",
        "0.0.2",
    )
    assert extract_framework_and_version("tikz_v0.0.2.json") == ("tikz", "0.0.2")

    # Multi-word without _v
    assert extract_framework_and_version("nvidia_ptx_8.5.json") == ("nvidia_ptx", "8.5")
    assert extract_framework_and_version("amd_rdna_19.1.0.json") == (
        "amd_rdna",
        "19.1.0",
    )
    assert extract_framework_and_version("latex_dsl_0.0.2.json") == (
        "latex_dsl",
        "0.0.2",
    )


def test_index_enveloped_snapshot_file() -> None:
    """Test that index_snapshot_file successfully indexes enveloped JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = init_db(db_path)

        json_path = os.path.join(tmpdir, "stablehlo_v1.9.0.json")
        enveloped_data = {
            "schema_version": "2.0.0",
            "target": "stablehlo",
            "version": "1.9.0",
            "operations": [
                {
                    "name": "abs",
                    "api_path": "stablehlo.abs",
                    "docstring": "Computes elementwise absolute value.",
                }
            ],
        }
        with open(json_path, "w") as f:
            json.dump(enveloped_data, f)

        count = index_snapshot_file(json_path, conn)
        assert count == 1

        cur = conn.cursor()
        cur.execute("SELECT framework, version, name, api_path FROM symbols")
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["framework"] == "stablehlo"
        assert rows[0]["version"] == "1.9.0"
        assert rows[0]["name"] == "abs"
        assert rows[0]["api_path"] == "stablehlo.abs"
        conn.close()
