"""Module docstring."""

from typing import Any


from unittest.mock import patch, MagicMock
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.frameworks.huggingface import (
    collect_transformers,
    collect_diffusers,
    collect_tokenizers,
)


def test_collect_transformers() -> None:
    """Function docstring."""
    mock_mod = MagicMock()

    class DummyConfig:
        """Class docstring."""

        __annotations__ = {"vocab_size": int, "hidden_size": int}

    class AutoModelForCausalLM:
        """Class docstring."""

        pass

    class DummyModel:
        """Class docstring."""

        def generate(self, input_ids: str, max_length: int = 20) -> str:
            """Function docstring.

            Args:
                input_ids: description
                max_length: description


            Returns:
                Return value.
            """
            return ""

    mock_mod.DummyConfig = DummyConfig
    mock_mod.AutoModelForCausalLM = AutoModelForCausalLM
    mock_mod.DummyModel = DummyModel
    mock_mod.__dir__ = lambda self: [  # type: ignore
        "DummyConfig",
        "AutoModelForCausalLM",
        "DummyModel",
    ]

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch(
            "ml_framework_snapshots.frameworks.huggingface.GhostInspector"
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

        res = collect_transformers(SemanticTier.MODEL)
        models = {r.name: r for r in res}

        assert "AutoModelForCausalLM" in models
        config_params = {p.name: p for p in models["DummyConfig"].params}
        assert "vocab_size" in config_params
        assert config_params["vocab_size"].annotation == "int"

        assert "AutoModelForCausalLM" in models
        auto_params = {p.name: p for p in models["AutoModelForCausalLM"].params}
        assert "config" in auto_params
        assert auto_params["config"].annotation == "PreTrainedConfig"

        assert "DummyModel" in models
        dummy_params = {p.name: p for p in models["DummyModel"].params}
        assert "max_length" in dummy_params
        assert dummy_params["max_length"].annotation == "int"


def test_collect_diffusers() -> None:
    """Function docstring."""
    mock_mod = MagicMock()

    class DummyScheduler:
        """Class docstring."""

        pass

    mock_mod.DummyScheduler = DummyScheduler
    mock_mod.__dir__ = lambda self: ["DummyScheduler"]  # type: ignore

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch(
            "ml_framework_snapshots.frameworks.huggingface.GhostInspector"
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

        res = collect_diffusers(SemanticTier.OPTIMIZER)
        assert len(res) == 1
        assert res[0].name == "DummyScheduler"


def test_collect_tokenizers() -> None:
    """Function docstring."""
    mock_mod = MagicMock()

    class DummyTokenizer:
        """Class docstring."""

        pass

    mock_mod.DummyTokenizer = DummyTokenizer
    mock_mod.__dir__ = lambda self: ["DummyTokenizer"]  # type: ignore

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch(
            "ml_framework_snapshots.frameworks.huggingface.GhostInspector"
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

        res = collect_tokenizers(SemanticTier.MODEL)
        assert len(res) == 1
        assert res[0].name == "DummyTokenizer"


from ml_framework_snapshots.frameworks.huggingface import _parse_pretrained_config  # noqa: E402
from ml_switcheroo_ir.schema.ghost import GhostRef  # noqa: E402


def test_parse_pretrained_config_with_empty_annotations() -> None:
    """Function docstring."""
    ref = GhostRef(name="MyConfig", api_path="pkg.MyConfig", kind="class")

    class DummyConfig:
        """Class docstring."""

        __annotations__ = {}

        def __init__(self, **kwargs: Any) -> Any:  # type: ignore
            """Init docstring.

            Args:
                **kwargs: kwargs
            """
            pass

    _parse_pretrained_config(DummyConfig, ref)
    assert len(ref.params) == 0
