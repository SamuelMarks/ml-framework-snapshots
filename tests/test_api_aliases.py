"""Module docstring."""

from ml_framework_snapshots.api import _consolidate_aliases
from ml_switcheroo_ir.schema.ghost import GhostParam
from ml_switcheroo_ir.schema.ghost import GhostRef


def test_consolidate_aliases():
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
