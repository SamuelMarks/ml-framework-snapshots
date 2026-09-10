"""Module docstring."""

from unittest.mock import patch
from ml_framework_snapshots.api import _consolidate_aliases, get_pkg_version
from ml_framework_snapshots.frameworks.optax_shim import collect_api
from ml_switcheroo_ir.schema.ghost import GhostParam
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_consolidate_aliases() -> None:
    """Function docstring."""
    ref1 = GhostRef(
        name="relu",
        api_path="torch.nn.functional.relu",
        kind="function",
        params=[GhostParam(name="input", kind="POSITIONAL_OR_KEYWORD")],
        docstring="ReLU",
        aliases=[],
    )
    ref2 = GhostRef(
        name="relu",
        api_path="torch.relu",
        kind="function",
        params=[GhostParam(name="input", kind="POSITIONAL_OR_KEYWORD")],
        docstring="ReLU",
        aliases=[],
    )

    # ref2 has shorter path
    consolidated = _consolidate_aliases([ref1, ref2])
    assert len(consolidated) == 1
    assert consolidated[0].api_path == "torch.relu"
    assert consolidated[0].aliases == ["torch.nn.functional.relu"]

    # ref1 has longer path, reverse order
    consolidated = _consolidate_aliases([ref2, ref1])
    assert len(consolidated) == 1
    assert consolidated[0].api_path == "torch.relu"
    assert consolidated[0].aliases == ["torch.nn.functional.relu"]


def test_get_pkg_version_aliases() -> None:
    """Test get_pkg_version aliases."""
    with patch("importlib.metadata.version", return_value="1.0.0") as mock_ver:
        assert get_pkg_version("optax_shim") == "1.0.0"
        mock_ver.assert_called_with("optax")

        assert get_pkg_version("huggingface") == "1.0.0"
        mock_ver.assert_called_with("transformers")

        assert get_pkg_version("orbax_checkpoint") == "1.0.0"
        mock_ver.assert_called_with("orbax-checkpoint")

        assert get_pkg_version("orbax") == "1.0.0"
        mock_ver.assert_called_with("orbax-checkpoint")


def test_optax_shim_collect_api() -> None:
    """Test optax_shim.collect_api."""
    with patch(
        "ml_framework_snapshots.frameworks.optax_shim.OptaxScanner.scan_optimizers",
        return_value=[],
    ):
        assert collect_api(SemanticTier.OPTIMIZER, False) == []
    with patch(
        "ml_framework_snapshots.frameworks.optax_shim.OptaxScanner.scan_losses",
        return_value=[],
    ):
        assert collect_api(SemanticTier.LOSS, False) == []
    with patch(
        "ml_framework_snapshots.frameworks.optax_shim.OptaxScanner.scan_schedulers",
        return_value=[],
    ):
        assert collect_api(SemanticTier.SCHEDULER, False) == []

    # default returns []
    assert collect_api(SemanticTier.UTIL, False) == []


def test_consolidate_aliases_extended_params_and_metadata() -> None:
    """Test that _consolidate_aliases preserves distinct instructions with different directionality or metadata."""
    from ml_framework_snapshots.models import (
        ExtendedGhostParam,
        ExtendedGhostRef,
        GhostResult,
        OperandDirection,
        IRParameterRole,
    )

    # Two instructions with identical name/kind/params except operand direction (WRITE vs READ)
    inst_write = ExtendedGhostRef(
        name="MOV",
        api_path="isa.inst.MOV_w",
        kind="instruction",
        params=[
            ExtendedGhostParam(
                name="op0",
                kind="POSITIONAL_ONLY",
                direction=OperandDirection.WRITE,
                role=IRParameterRole.OPERAND,
                dtypes=["float32"],
                rank=1,
            )
        ],
        docstring="Move instruction write",
    )
    inst_read = ExtendedGhostRef(
        name="MOV",
        api_path="isa.inst.MOV_r",
        kind="instruction",
        params=[
            ExtendedGhostParam(
                name="op0",
                kind="POSITIONAL_ONLY",
                direction=OperandDirection.READ,
                role=IRParameterRole.OPERAND,
                dtypes=["float32"],
                rank=1,
            )
        ],
        docstring="Move instruction write",
    )
    res = _consolidate_aliases([inst_write, inst_read])
    assert len(res) == 2

    # Two instructions differing only in domain_metadata (e.g. sm_80 vs sm_90)
    inst_sm80 = ExtendedGhostRef(
        name="MMA",
        api_path="isa.inst.MMA_sm80",
        kind="instruction",
        params=[GhostParam(name="op0", kind="POSITIONAL_ONLY")],
        domain_metadata={
            "arch": "sm_80",
            "features": ["tensor_core"],
            "tags": {"fast"},
        },
        docstring="MMA",
    )
    inst_sm90 = ExtendedGhostRef(
        name="MMA",
        api_path="isa.inst.MMA_sm90",
        kind="instruction",
        params=[GhostParam(name="op0", kind="POSITIONAL_ONLY")],
        domain_metadata={"arch": "sm_90", "features": ["wgmma"], "tags": {"fast"}},
        docstring="MMA",
    )
    res_arch = _consolidate_aliases([inst_sm80, inst_sm90])
    assert len(res_arch) == 2

    # Operations differing only in returns
    op_res1 = ExtendedGhostRef(
        name="add",
        api_path="mlir.add1",
        kind="operation",
        params=[GhostParam(name="lhs", kind="POSITIONAL_ONLY")],
        returns=[GhostResult(name="res1", type="tensor<f32>")],
        docstring="add op",
    )
    op_res2 = ExtendedGhostRef(
        name="add",
        api_path="mlir.add2",
        kind="operation",
        params=[GhostParam(name="lhs", kind="POSITIONAL_ONLY")],
        returns=[GhostResult(name="res2", type="tensor<f64>")],
        docstring="add op",
    )
    res_ops = _consolidate_aliases([op_res1, op_res2])
    assert len(res_ops) == 2
