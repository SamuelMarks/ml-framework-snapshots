"""Module docstring."""

import sys
from ml_framework_snapshots.models import GhostInspector


class TensorRef:
    """Class docstring."""

    pass


def forward(x: "TensorRef") -> "TensorRef":
    """Function docstring.

    Args:
        x: description
    """
    return x


sys.modules[__name__].TensorRef = TensorRef  # type: ignore


def test_resolve_forward_ref():
    """Function docstring."""
    ref = GhostInspector.inspect(forward, "forward")
    assert ref.has_arg("x")
    # Due to stringification and sanitization, <class 'test_models_forwardref.TensorRef'>
    # should become 'test_models_forwardref.TensorRef' or 'TensorRef' depending on how it's resolved.
    assert "TensorRef" in ref.params[0].annotation
    assert "TensorRef" in ref.returns_type
