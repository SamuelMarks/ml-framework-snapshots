"""Module docstring."""

from typing import Any


import inspect
from unittest.mock import patch
from ml_framework_snapshots.models import GhostInspector


def dummy_c_ext() -> None:
    """dummy_c_ext(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor"""  # noqa: D402, D415
    pass


original_sig = inspect.signature


def patched_sig(target: Any, *args: Any, **kwargs: Any) -> None:
    """Function docstring.

    Args:
        target: description
        args: description
        kwargs: description


    Raises:
        ValueError: Exception.

    Returns:
        Return value.
    """
    if target is dummy_c_ext:
        raise ValueError("no signature found")
    return original_sig(target, *args, **kwargs)  # type: ignore


def test_c_extension_signature_fallback() -> None:
    """Function docstring."""
    with patch("inspect.signature", side_effect=patched_sig):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_c_ext, "dummy_c_ext")

        assert len(ref.params) == 2
        assert ref.params[0].name == "input"
        assert ref.params[0].annotation == "Tensor"
        assert ref.params[1].name == "out"
        assert ref.params[1].kind == "KEYWORD_ONLY"
        assert ref.params[1].annotation == "Tensor | None"
