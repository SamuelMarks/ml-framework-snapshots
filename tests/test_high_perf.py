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
    mock_tl.__dir__ = lambda self: [  # type: ignore
        "constexpr",
        "tensor",
        "_priv",
        "not_callable",
    ]

    def constexpr() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    constexpr.__module__ = "triton.language"
    constexpr.__name__ = "constexpr"

    def tensor() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    tensor.__module__ = "triton.language"
    tensor.__name__ = "tensor"

    mock_tl.constexpr = constexpr
    mock_tl.tensor = tensor
    mock_tl.not_callable = 123
    mock_tl._priv = lambda: None

    import types

    mock_math = types.ModuleType("triton.language.math")
    setattr(mock_math, "a_bad", 123)
    setattr(mock_math, "sin", lambda: None)
    setattr(mock_math, "_priv", lambda: None)
    setattr(mock_math, "bad", 123)
    mock_tl.math = mock_math

    with patch("ml_framework_snapshots.models.GhostInspector.inspect") as mock_inspect:
        mock_inspect.return_value = MagicMock()  # Return a dummy GhostRef
        with patch("importlib.import_module") as mock_import:

            def side_effect(name: Any) -> Any:
                """Function docstring.

                Args:
                    name: description


                Raises:
                    ImportError: Exception.

                Returns:
                    Return value.
                """
                if name == "triton":
                    return MagicMock()
                elif name == "triton.language":  # pragma: no branch
                    return mock_tl
                raise ImportError(name)  # pragma: no cover

            mock_import.side_effect = side_effect

            res = triton_collect(SemanticTier.UTIL)
            assert len(res) == 3

            res_priv = triton_collect(SemanticTier.UTIL, include_nonpublic=True)
            assert len(res_priv) == 5

            # Now set mock_inspect to return None to cover if ref is False
            mock_inspect.return_value = None
            res_none = triton_collect(SemanticTier.UTIL)
            assert res_none == []


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
        pass  # pragma: no cover

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
