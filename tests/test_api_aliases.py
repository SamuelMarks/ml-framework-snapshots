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
