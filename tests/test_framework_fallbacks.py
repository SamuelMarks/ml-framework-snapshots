"""Tests for missing framework dependencies."""

from typing import Any
from unittest.mock import patch, MagicMock
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_deepspeed_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.deepspeed import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class DeepspeedMock:
        """Class docstring."""

        def __dir__(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return ["_hidden", "initialize", "nothing"]

        def initialize(self) -> Any:
            """Function docstring."""
            pass  # pragma: no cover

        def nothing(self) -> Any:
            """Function docstring."""
            pass  # pragma: no cover

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=lambda obj, name: (
            None
            if "nothing" in name
            else MagicMock(params=[MagicMock(**{"name": "config_params"})])
        ),
    ):
        with patch("importlib.import_module", return_value=DeepspeedMock()):
            collect_api(SemanticTier.MODEL)
            collect_api(SemanticTier.MODEL, include_nonpublic=True)

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        return_value=MagicMock(params=[]),
    ):
        with patch("importlib.import_module", return_value=DeepspeedMock()):
            collect_api(SemanticTier.MODEL)


def test_onnxruntime_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.onnxruntime import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class OnnxMock:
        """Class docstring."""

        def __dir__(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return ["_hidden", "InferenceSession", "nothing"]

        def InferenceSession(self) -> Any:
            """Function docstring."""
            pass  # pragma: no cover

        def nothing(self) -> Any:
            """Function docstring."""
            pass  # pragma: no cover

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=lambda obj, name: (
            None
            if "nothing" in name
            else MagicMock(params=[MagicMock(**{"name": "path_or_bytes"})])
        ),
    ):
        with patch("importlib.import_module", return_value=OnnxMock()):
            collect_api(SemanticTier.MODEL)
            collect_api(SemanticTier.MODEL, include_nonpublic=True)

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        return_value=MagicMock(params=[]),
    ):
        with patch("importlib.import_module", return_value=OnnxMock()):
            collect_api(SemanticTier.MODEL)


def test_triton_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.triton import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.UTIL) == []
    assert collect_api(SemanticTier.LOSS) == []

    mock_mod = MagicMock()
    with patch.dict("sys.modules", {"triton": mock_mod}):
        collect_api(SemanticTier.UTIL)

    mock_mod.__dir__ = lambda self: ["_hidden", "fn", "not_ref"]  # type: ignore

    def fake_fn(a: Any, b: Any) -> Any:
        """Function docstring.

        Args:
            a: description
            b: description
        """
        pass  # pragma: no cover

    fake_fn.__annotations__ = {"a": "int constexpr", "b": "float"}

    class Wrap:
        """Class docstring."""

        fn = fake_fn

    mock_mod.fn = Wrap
    mock_mod.not_ref = (
        lambda: None
    )  # mock inspector will return None for this because it's not a valid ref or we can mock inspector

    param_a = MagicMock()
    param_a.name = "a"
    param_b = MagicMock()
    param_b.name = "b"

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=lambda obj, name: (
            None if "not_ref" in name else MagicMock(params=[param_a, param_b])
        ),
    ):
        with patch(
            "importlib.import_module",
            side_effect=lambda x: (
                mock_mod
                if x == "triton.language" or x == "triton"
                else ImportError("No")
            ),
        ):
            collect_api(SemanticTier.UTIL)


def test_sklearn_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.sklearn import collect_api

    with patch(
        "ml_framework_snapshots.frameworks.sklearn.get_all_members",
        side_effect=Exception,
    ):
        res = collect_api(SemanticTier.LAYER)
        assert res == []


def test_huggingface_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.huggingface import (
        _extract_generation_kwargs,
        _parse_pretrained_config,
        collect_transformers,
    )
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    class M:
        """Class docstring."""

        def generate(self, input: Any, *args: Any, **kwargs: Any) -> Any:
            """Function docstring.

            Args:
                input: description
                args: description
                kwargs: description
            """
            pass  # pragma: no cover

    r = GhostRef(name="A", api_path="A", kind="class", params=[])
    _extract_generation_kwargs(M(), r)

    class M2:
        """Class docstring."""

        __annotations__ = {"a": int, "b": "str"}

    r2 = GhostRef(name="A", api_path="A", kind="class", params=[])
    _parse_pretrained_config(M2(), r2)

    with patch("importlib.import_module", side_effect=ImportError):
        collect_transformers(SemanticTier.MODEL)

    collect_transformers(SemanticTier.LOSS)

    class BadProp:
        """Class docstring."""

        @property
        def bad(self) -> Any:
            """Function docstring.

            Raises:
                ValueError: Exception.
            """
            raise ValueError()  # pragma: no cover

    mock_mod = MagicMock()
    mock_mod.__dir__ = lambda self: [  # type: ignore
        "_hidden",
        "DummyConfig",
        "AutoModel",
        "Other",
        "bad",
    ]
    mock_mod.DummyConfig = type("DummyConfig", (), {})()  # No __annotations__
    mock_mod.AutoModel = MagicMock()

    # Mock getattr correctly so it raises for 'bad'
    # we need to simulate the module behavior
    class FakeMod:
        """Class docstring."""

        def __dir__(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return [
                "_hidden",
                "DummyConfig",
                "AutoModelA",
                "AutoModelB",
                "Other",
                "bad",
                "ThrowConfig",
                "ReturnNone",
                "GenModel",
                "RefNone",
                "EmptyConfig",
                "EmptyConfigB",
            ]

        @property
        def bad(self) -> Any:
            """Function docstring.

            Raises:
                ValueError: Exception.
            """
            raise ValueError()

        @property
        def DummyConfig(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return type("DummyConfig", (), {"__annotations__": {}})()

        @property
        def EmptyConfig(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return type("EmptyConfig", (), {"__annotations__": {"a": int}})()

        @property
        def EmptyConfigB(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return type("EmptyConfigB", (), {"__annotations__": {"b": int}})()

        @property
        def AutoModelA(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def AutoModelB(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def Other(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def ThrowConfig(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def ReturnNone(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return None

        @property
        def GenModel(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """

            class M:
                """Class docstring."""

                def generate(self, input: Any, *args: Any, **kwargs: Any) -> Any:
                    """Function docstring.

                    Args:
                        input: description
                        args: description
                        kwargs: description
                    """
                    pass  # pragma: no cover

            return M()

        @property
        def RefNone(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

    def mock_inspect(obj: Any, name: Any) -> Any:
        """Function docstring.

        Args:
            obj: description
            name: description


        Raises:
            Exception: Exception.

        Returns:
            Return value.
        """
        if "ThrowConfig" in name:
            raise Exception("test")
        if "RefNone" in name:
            return None
        if "AutoModelA" in name:
            return MagicMock(params=[])
        if "AutoModelB" in name:
            return MagicMock(params=[MagicMock(**{"name": "config"})])
        if "GenModel" in name:
            return GhostRef(
                name="GenModel",
                api_path="GenModel",
                kind="class",
                params=[
                    GhostParam(
                        name="input",
                        kind="POSITIONAL_OR_KEYWORD",
                        default_value=None,
                        annotation="Any",
                    )
                ],
            )
        if "EmptyConfig" in name:
            return MagicMock(params=[MagicMock(**{"name": "a"})])
        return MagicMock(params=[MagicMock(**{"name": "config"})])

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=mock_inspect,
    ):
        with patch("importlib.import_module", return_value=FakeMod()):
            collect_transformers(SemanticTier.MODEL)
            collect_transformers(SemanticTier.UTIL)
