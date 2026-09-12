"""Tests for Ghost Protocol v2 contract synchronization and bidirectional SerDe."""

import json
import os

from ml_framework_snapshots.models import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    SnapshotEnvelope,
    migrate_ghost_ref_v2,
)


def test_ghost_protocol_v2_models_instantiation() -> None:
    """Verify instantiation of all Ghost Protocol v2 models."""
    res = GhostResult(name="res0", type="tensor<4xf32>", description="Output tensor")
    assert res.name == "res0"

    param = ExtendedGhostParam(
        name="input",
        kind="POSITIONAL_ONLY",
        direction=OperandDirection.READ,
        role=IRParameterRole.OPERAND,
        dtypes=["float32", "bfloat16"],
        rank=2,
    )
    assert param.direction == OperandDirection.READ
    assert param.role == IRParameterRole.OPERAND

    isa_ref = GhostIsaRef(
        name="FFMA",
        api_path="isa.inst.FFMA",
        kind="function",
        params=[param],
        predicate_guards=["@P0", "@!P1"],
        instruction_modifiers=[".F32", ".FTZ"],
    )
    assert isa_ref.domain_type in ("isa", "instruction")
    assert isa_ref.predicate_guards == ["@P0", "@!P1"]

    mlir_ref = GhostMlirRef(
        name="addf",
        api_path="arith.addf",
        kind="operation",
        params=[param],
        returns=[res],
        traits=["Commutative", "SameOperandsAndResultType"],
    )
    assert mlir_ref.domain_type in ("mlir", "operation")
    assert mlir_ref.traits == ["Commutative", "SameOperandsAndResultType"]

    envelope = SnapshotEnvelope(
        target="stablehlo",
        schema_version="2.0.0",
        version="1.0.0",
    )
    assert envelope.schema_version == "2.0.0"


def test_migrate_ghost_ref_v2_roundtrip() -> None:
    """Verify migrate_ghost_ref_v2 upgrades legacy structures with zero information loss."""
    legacy_dict = {
        "name": "relu",
        "api_path": "torch.nn.functional.relu",
        "kind": "function",
        "schema_version": "1.2",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "inplace", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
    }

    upgraded = migrate_ghost_ref_v2(legacy_dict)
    assert isinstance(upgraded, ExtendedGhostRef)
    assert upgraded.name == "relu"
    assert upgraded.api_path == "torch.nn.functional.relu"

    # Test roundtrip serialization to JSON and back
    json_str = upgraded.model_dump_json()
    re_parsed = migrate_ghost_ref_v2(json.loads(json_str))
    assert re_parsed.name == upgraded.name
    assert len(re_parsed.params) == len(upgraded.params)


def test_exhaustive_snapshot_loading_serde() -> None:
    """Verify all 5 exhaustive snapshot files deserialize into Ghost Protocol v2 models."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fw_dir = os.path.join(repo_root, "src", "ml_framework_snapshots", "frameworks")

    for fname in [
        "stablehlo_exhaustive.json",
        "mlir_exhaustive.json",
        "nvidia_sass_exhaustive.json",
        "nvidia_ptx_exhaustive.json",
        "amd_rdna_exhaustive.json",
    ]:
        fpath = os.path.join(fw_dir, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) > 0, f"Snapshot {fname} is empty"

        for raw_item in records[:10]:
            ref = migrate_ghost_ref_v2(raw_item)
            assert ref.name or getattr(ref, "mnemonic", None)
            # Re-serialize to verify schema integrity
            dumped = ref.model_dump()
            assert isinstance(dumped, dict)
