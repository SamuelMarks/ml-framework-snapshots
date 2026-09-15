"""Tests for parameter kind parsing, overload merging, and inheritance resolution in GhostInspector."""

from typing import Any
from unittest.mock import patch

from ml_framework_snapshots.models import (
    GhostInspector,
    to_parameter_kind,
)
from ml_framework_snapshots.utils import CExtensionSignature
from ml_switcheroo_ir.schema.ghost import ParameterKind


def test_to_parameter_kind_branches() -> None:
    """Test to_parameter_kind returns existing enum or falls back to POSITIONAL_OR_KEYWORD."""
    assert (
        to_parameter_kind(ParameterKind.POSITIONAL_ONLY)
        == ParameterKind.POSITIONAL_ONLY
    )
    assert to_parameter_kind("NON_EXISTENT_KIND") == ParameterKind.POSITIONAL_OR_KEYWORD


def test_ghost_inspector_aten_overload_merge_and_returns() -> None:
    """Test merging aten_sig overloads when c_ext_params lacks overloads and inheriting returns_type."""

    def dummy_func_doc() -> None:
        """func(x: int) -> None."""
        pass

    c_sig = CExtensionSignature(
        [("x", "POSITIONAL_OR_KEYWORD", None, "int")],
        returns_type="Tensor",
    )
    aten_sig = CExtensionSignature(
        [("x", "POSITIONAL_OR_KEYWORD", None, "int")],
        overloads=[
            CExtensionSignature([("x", "POSITIONAL_OR_KEYWORD", None, "float")])
        ],
    )

    with (
        patch("inspect.signature", side_effect=ValueError),
        patch(
            "ml_framework_snapshots.models.extract_c_extension_signature",
            return_value=c_sig,
        ),
        patch(
            "ml_framework_snapshots.frameworks.torch.extract_aten_c_extension_signature",
            return_value=aten_sig,
        ),
    ):
        ref = GhostInspector.inspect(dummy_func_doc, "torch.func_doc")
        assert ref.overloads is not None
        assert len(ref.overloads) == 1
        assert ref.returns_type == "Tensor"


def test_ghost_inspector_opaque_c_extension_none_env_tags() -> None:
    """Test opaque C-extension handling when environment_tags is None."""

    def opaque_fn(*args: Any, **kwargs: Any) -> None:
        """Opaque function without signature or docstring."""
        pass

    with (
        patch("inspect.signature", side_effect=ValueError),
        patch(
            "ml_framework_snapshots.models.extract_c_extension_signature",
            return_value=None,
        ),
        patch(
            "ml_framework_snapshots.frameworks.torch.extract_aten_c_extension_signature",
            return_value=None,
        ),
    ):
        ref = GhostInspector.inspect(
            opaque_fn, "torch.opaque", kind="function", environment_tags=None
        )
        assert ref.environment_tags is not None
        assert "inexact_signature" in ref.environment_tags
        assert "opaque_c_extension" in ref.environment_tags


class SampleParentForSuper:
    """Parent class."""

    def __init__(self, a: int, b: str = "default") -> None:
        """Initialize Parent."""
        pass


class SampleChildForSuper(SampleParentForSuper):
    """Child class."""

    def __init__(self, a: int, **kwargs: Any) -> None:
        """Initialize Child."""
        super().__init__(a=a, **kwargs)


def test_ghost_inspector_super_kwargs_already_present_param() -> None:
    """Test super kwargs inspection when parent parameter already exists in extracted_params."""
    ref = GhostInspector.inspect(SampleChildForSuper, "tests.SampleChildForSuper")
    param_names = [p.name for p in ref.params]
    assert param_names == ["a", "kwargs", "b"]
