"""Module docstring."""

# ruff: noqa: D205, D415, D207, D212

from typing import Any


import inspect
from unittest.mock import patch
from ml_framework_snapshots.models import GhostInspector


def dummy_c_ext() -> None:
    """dummy_c_ext(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor"""  # noqa: D402, D415
    pass  # pragma: no cover


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
    if target is dummy_c_ext:  # pragma: no branch
        raise ValueError("no signature found")
    return original_sig(target, *args, **kwargs)  # type: ignore  # pragma: no cover


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


def dummy_posonly(*args: Any, **kwargs: Any) -> None:
    """dummy_posonly(a: int, /, b: str = 'x') -> None"""  # noqa: D402, D415
    pass  # pragma: no cover


def test_c_extension_posonlyargs() -> None:
    """Test extracting positional-only parameters from C-extension docstring."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_posonly, "dummy_posonly")
        assert len(ref.params) == 2
        assert ref.params[0].name == "a"
        assert str(ref.params[0].kind).endswith("POSITIONAL_ONLY")
        assert ref.params[1].name == "b"
        assert str(ref.params[1].kind).endswith("POSITIONAL_OR_KEYWORD")


def dummy_no_doc() -> None:
    """Dummy no doc function."""
    pass


def test_c_extension_inexact_signature_fallback() -> None:
    """Test that unresolvable functions receive inexact_signature tag."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_no_doc, "dummy_no_doc")
        assert "inexact_signature" in (ref.environment_tags or [])

        # When environment_tags already contains inexact_signature
        ref2 = inspector.inspect(
            dummy_no_doc,
            "dummy_no_doc",
            environment_tags=["custom_tag", "inexact_signature"],
        )
        assert "inexact_signature" in (ref2.environment_tags or [])


def dummy_torch_add() -> None:
    """add(input, other, *, out=None) -> Tensor
    add(input, other, *, alpha=1, out=None) -> Tensor

    Adds other scaled by alpha to input.
    """
    pass


def dummy_torch_clamp() -> None:
    """clamp(input, min=None, max=None, *, out=None) -> Tensor
    clamp(input, *, min=None, max=None, out=None) -> Tensor
    clamp(input, min: Tensor, max: Tensor, *, out=None) -> Tensor

    Clamps all elements in input into the range [min, max].
    """
    pass


def dummy_pybind11_overload() -> None:
    """process(*args, **kwargs)
    Overloaded function.

    1. process(input: Tensor, factor: float) -> Tensor

    2. process(input: Tensor, matrix: Tensor) -> Tensor
    """
    pass


def test_c_extension_multi_signature_torch_add() -> None:
    """Test multi-signature overload extraction matching torch.add patterns."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_torch_add, "add")

        assert ref.returns_type == "Tensor"
        assert len(ref.params) == 3
        assert [p.name for p in ref.params] == ["input", "other", "out"]
        assert ref.params[2].kind == "KEYWORD_ONLY"

        # Check secondary overload in ref.overloads
        assert len(ref.overloads) == 1
        ov = ref.overloads[0]
        assert ov.name == ref.name
        assert ov.returns_type == "Tensor"
        assert [p.name for p in ov.params] == ["input", "other", "alpha", "out"]
        assert ov.params[2].name == "alpha"
        assert ov.params[2].kind == "KEYWORD_ONLY"
        assert ov.params[2].default == "1"


def test_c_extension_multi_signature_torch_clamp() -> None:
    """Test multi-signature overload extraction matching torch.clamp with 3 overloads."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_torch_clamp, "clamp")

        assert ref.returns_type == "Tensor"
        assert len(ref.params) == 4
        assert [p.name for p in ref.params] == ["input", "min", "max", "out"]
        assert ref.params[1].kind == "POSITIONAL_OR_KEYWORD"

        # Check secondary and tertiary overloads
        assert len(ref.overloads) == 2
        ov1 = ref.overloads[0]
        assert [p.name for p in ov1.params] == ["input", "min", "max", "out"]
        assert ov1.params[1].kind == "KEYWORD_ONLY"
        assert ov1.params[2].kind == "KEYWORD_ONLY"
        assert ov1.params[3].kind == "KEYWORD_ONLY"

        ov2 = ref.overloads[1]
        assert [p.name for p in ov2.params] == ["input", "min", "max", "out"]
        assert ov2.params[1].annotation == "Tensor"
        assert ov2.params[2].annotation == "Tensor"


def test_c_extension_multi_signature_pybind11() -> None:
    """Test PyBind11 numbered overloads promote the first numbered overload to primary."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_pybind11_overload, "process")

        assert ref.returns_type == "Tensor"
        assert len(ref.params) == 2
        assert [p.name for p in ref.params] == ["input", "factor"]
        assert ref.params[1].annotation == "float"

        assert len(ref.overloads) == 1
        ov = ref.overloads[0]
        assert [p.name for p in ov.params] == ["input", "matrix"]
        assert ov.params[1].annotation == "Tensor"


def dummy_overloaded_header() -> None:
    """foo(a: int) -> int
    Overloaded function.
    foo(b: str) -> str
    """
    pass


def test_c_extension_overloaded_header_branch() -> None:
    """Test C-extension docstring containing 'overloaded function' header between signatures."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_overloaded_header, "foo")
        assert len(ref.params) == 1
        assert ref.params[0].name == "a"
        assert len(ref.overloads) == 1
        assert ref.overloads[0].params[0].name == "b"


def dummy_leading_overload_header() -> None:
    """Overloaded function.

    1. bar(x: int) -> int

    2. bar(y: str) -> str
    """
    pass


def dummy_overload_vararg() -> None:
    """baz(x: int) -> int
    baz(*args) -> int
    """
    pass


def dummy_no_ret_type() -> None:
    """qux(x: int)"""
    pass


def test_c_extension_leading_header_and_vararg_overloads() -> None:
    """Test leading 'overloaded function' header and varargs in secondary overload."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_leading_overload_header, "bar")
        assert len(ref.params) == 1
        assert ref.params[0].name == "x"
        assert len(ref.overloads) == 1
        assert ref.overloads[0].params[0].name == "y"

        ref_var = inspector.inspect(dummy_overload_vararg, "baz")
        assert len(ref_var.overloads) == 1
        assert ref_var.overloads[0].has_varargs is True

        ref_no_ret = inspector.inspect(dummy_no_ret_type, "qux")
        assert ref_no_ret.returns_type in (None, "NoneType")
        assert len(ref_no_ret.params) == 1


def dummy_leading_empty_and_overload_header() -> None:
    """

    Overloaded function.

    1. func(x: int) -> int

    2. func(y: str) -> str
    """
    pass


def test_c_extension_leading_empty_lines() -> None:
    """Test C-extension docstring with leading empty lines and 'Overloaded function.' header."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_leading_empty_and_overload_header, "func")
        assert len(ref.params) == 1
        assert ref.params[0].name == "x"
        assert len(ref.overloads) == 1
        assert ref.overloads[0].params[0].name == "y"
