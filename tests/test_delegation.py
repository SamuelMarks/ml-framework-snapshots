"""Module docstring."""

from ml_framework_snapshots.models import GhostInspector


class BaseLayer:
    """Class docstring."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        """Function docstring.

        Args:
            hidden_dim: description
            dropout: description
        """
        pass


class SubLayer(BaseLayer):
    """Class docstring."""

    def __init__(self, name: str, **kwargs):
        """Function docstring.

        Args:
            name: description
            kwargs: description
        """
        super().__init__(**kwargs)


def test_delegation_tracing() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(SubLayer, "SubLayer")

    assert ref.name == "SubLayer"

    param_names = [p.name for p in ref.params]
    assert "name" in param_names
    assert "hidden_dim" in param_names
    assert "dropout" in param_names

    # kwargs might still be there or might be removed, but hidden_dim should be explicitly extracted

    hidden_param = next(p for p in ref.params if p.name == "hidden_dim")
    assert hidden_param.annotation == "int"
    assert hidden_param.default == "256"
