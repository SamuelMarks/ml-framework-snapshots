"""Module docstring."""

from typing import Any


from unittest.mock import patch, MagicMock
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.frameworks.triton import collect_api as triton_collect
from ml_framework_snapshots.frameworks.onnxruntime import collect_api as onnx_collect
from ml_framework_snapshots.frameworks.deepspeed import collect_api as ds_collect


def test_triton_collect() -> None:
    """Function docstring."""
    mock_tl = MagicMock()
    mock_tl.__dir__ = lambda self: ["constexpr", "tensor"]  # type: ignore

    def constexpr() -> Any:
        """Function docstring."""
        pass

    constexpr.__module__ = "triton.language"
    constexpr.__name__ = "constexpr"

    def tensor() -> Any:
        """Function docstring."""
        pass

    tensor.__module__ = "triton.language"
    tensor.__name__ = "tensor"

    mock_tl.constexpr = constexpr
    mock_tl.tensor = tensor

    with patch("ml_framework_snapshots.models.GhostInspector.inspect") as mock_inspect:
        mock_inspect.return_value = MagicMock()  # Return a dummy GhostRef
        with patch("importlib.import_module") as mock_import:

            def side_effect(name: Any) -> Any:
                """Function docstring.

                Args:
                    name: description
                """
                if name == "triton":
                    return MagicMock()
                elif name == "triton.language":
                    return mock_tl
                raise ImportError(name)

            mock_import.side_effect = side_effect

            res = triton_collect(SemanticTier.UTIL)
            assert len(res) == 2


def test_onnx_collect() -> None:
    """Function docstring."""
    mock_onnx = MagicMock()
    mock_onnx.__dir__ = lambda self: ["InferenceSession", "utils"]  # type: ignore

    class InferenceSession:
        """Class docstring."""

        pass

    mock_onnx.InferenceSession = InferenceSession

    with (
        patch("importlib.import_module", return_value=mock_onnx),
        patch(
            "ml_framework_snapshots.frameworks.onnxruntime.GhostInspector"
        ) as MockInspector,
    ):
        from ml_switcheroo_ir.schema.ghost import GhostRef

        MockInspector.return_value.inspect.side_effect = lambda obj, path: GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

        res = onnx_collect(SemanticTier.MODEL)
        assert len(res) == 1
        assert res[0].name == "InferenceSession"
        assert any(p.name == "providers" for p in res[0].params)


def test_ds_collect() -> None:
    """Function docstring."""
    mock_ds = MagicMock()
    mock_ds.__dir__ = lambda self: ["initialize", "utils"]  # type: ignore

    def initialize() -> Any:
        """Function docstring."""
        pass

    mock_ds.initialize = initialize

    with (
        patch("importlib.import_module", return_value=mock_ds),
        patch(
            "ml_framework_snapshots.frameworks.deepspeed.GhostInspector"
        ) as MockInspector,
    ):
        from ml_switcheroo_ir.schema.ghost import GhostRef

        MockInspector.return_value.inspect.side_effect = lambda obj, path: GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

        res = ds_collect(SemanticTier.MODEL)
        assert len(res) == 1
        assert res[0].name == "initialize"
        assert any(p.name == "config_params" for p in res[0].params)
