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


    Returns:
        Return value.
    """
    return x


setattr(sys.modules[__name__], "TensorRef", TensorRef)


def test_resolve_forward_ref() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(forward, "forward")
    assert ref.has_arg("x")
    # Due to stringification and sanitization, <class 'test_models_forwardref.TensorRef'>
    # should become 'test_models_forwardref.TensorRef' or 'TensorRef' depending on how it's resolved.
    assert ref.params[0].annotation is not None
    assert "TensorRef" in ref.params[0].annotation
    assert ref.returns_type is not None
    assert "TensorRef" in ref.returns_type
