"""Module docstring."""

from typing import Any


import types
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostRef


def create_module(name: str, attrs: dict[str, Any]) -> types.ModuleType:
    """Function docstring.

    Args:
        name: description
        attrs: description


    Returns:
        Return value.
    """
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_torch_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import torch as torch_fw

    class Module:
        """Class docstring."""

        pass

    class MSELoss(Module):
        """Class docstring."""

        pass

    class _Loss(Module):
        """Class docstring."""

        pass

    class _PrivateLoss(Module):
        """Class docstring."""

        pass

    class Optimizer:
        """Class docstring."""

        pass

    class Adam(Optimizer):
        """Class docstring."""

        pass

    class _PrivateOptim(Optimizer):
        """Class docstring."""

        pass

    class OtherClass:
        """Class docstring."""

        pass

    class ReLU(Module):
        """Class docstring."""

        pass

    class Linear(Module):
        """Class docstring."""

        pass

    class _PrivateLayer(Module):
        """Class docstring."""

        pass

    class NotModuleLoss:
        """Class docstring."""

        pass

    class LRScheduler:
        """Class docstring."""

        pass

    class StepLR(LRScheduler):
        """Class docstring."""

        pass

    class _PrivateScheduler(LRScheduler):
        """Class docstring."""

        pass

    def xavier_uniform_() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def _private_init() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    class DataLoader:
        """Class docstring."""

        pass

    class _PrivateLoader:
        """Class docstring."""

        pass

    fake_nn_init = create_module(
        "torch.nn.init",
        {"xavier_uniform_": xavier_uniform_, "_private_init": _private_init},
    )

    fake_nn = create_module(
        "torch.nn",
        {
            "Module": Module,
            "MSELoss": MSELoss,
            "_Loss": _Loss,
            "_PrivateLoss": _PrivateLoss,
            "ReLU": ReLU,
            "Linear": Linear,
            "_PrivateLayer": _PrivateLayer,
            "NotModuleLoss": NotModuleLoss,
            "init": fake_nn_init,
        },
    )

    fake_lr_scheduler = create_module(
        "torch.optim.lr_scheduler",
        {
            "LRScheduler": LRScheduler,
            "StepLR": StepLR,
            "_PrivateScheduler": _PrivateScheduler,
        },
    )

    fake_optim = create_module(
        "torch.optim",
        {
            "Optimizer": Optimizer,
            "Adam": Adam,
            "_PrivateOptim": _PrivateOptim,
            "OtherClass": OtherClass,
            "lr_scheduler": fake_lr_scheduler,
        },
    )
    fake_data = create_module(
        "torch.utils.data", {"DataLoader": DataLoader, "_PrivateLoader": _PrivateLoader}
    )

    mocker.patch.object(torch_fw, "nn", fake_nn)
    mocker.patch.object(torch_fw, "optim", fake_optim)
    mocker.patch.object(torch_fw, "data", fake_data)

    mocker.patch.dict(
        "sys.modules",
        {
            "torch": create_module(
                "torch",
                {
                    "abs": lambda: None,
                    "fft": create_module(
                        "torch.fft",
                        {
                            "fft": lambda: None,
                            "_hidden": lambda: None,
                            "not_callable": 123,
                        },
                    ),
                    "nn": create_module(
                        "torch.nn",
                        {
                            "functional": create_module(
                                "torch.nn.functional",
                                {"interpolate": lambda: None, "non_existent": None},
                            )
                        },
                    ),
                },
            ),
        },
    )

    losses = torch_fw.collect_api(SemanticTier.LOSS)
    lnames = [x.name for x in losses]
    assert "MSELoss" in lnames
    assert "_Loss" not in lnames
    assert "_PrivateLoss" not in lnames
    assert "NotModuleLoss" not in lnames

    optims = torch_fw.collect_api(SemanticTier.OPTIMIZER)
    onames = [x.name for x in optims]
    assert "Adam" in onames
    assert "Optimizer" not in onames
    assert "_PrivateOptim" not in onames

    acts = torch_fw.collect_api(SemanticTier.ACTIVATION)
    anames = [x.name for x in acts]
    assert "ReLU" in anames

    layers = torch_fw.collect_api(SemanticTier.LAYER)
    lanames = [x.name for x in layers]
    assert "Linear" in lanames
    assert "ReLU" not in lanames
    assert "_PrivateLayer" not in lanames

    scheds = torch_fw.collect_api(SemanticTier.SCHEDULER)
    snames = [x.name for x in scheds]
    assert "StepLR" in snames
    assert "LRScheduler" not in snames
    assert "_PrivateScheduler" not in snames

    inits = torch_fw.collect_api(SemanticTier.INITIALIZER)
    inames = [x.name for x in inits]
    assert "xavier_uniform_" in inames
    assert "_private_init" not in inames

    mets = torch_fw.collect_api(SemanticTier.METRIC)
    assert mets == []

    loaders = torch_fw.collect_api(SemanticTier.DATALOADER)
    dnames = [x.name for x in loaders]
    assert "DataLoader" in dnames
    assert "_PrivateLoader" not in dnames

    # test include_nonpublic
    priv_inits = torch_fw.collect_api(SemanticTier.INITIALIZER, include_nonpublic=True)
    pinames = [x.name for x in priv_inits]
    assert "_private_init" in pinames

    priv_losses = torch_fw.collect_api(SemanticTier.LOSS, include_nonpublic=True)
    plnames = [x.name for x in priv_losses]
    assert "_PrivateLoss" in plnames

    priv_layers = torch_fw.collect_api(SemanticTier.LAYER, include_nonpublic=True)
    playnames = [x.name for x in priv_layers]
    assert "_PrivateLayer" in playnames

    priv_optims = torch_fw.collect_api(SemanticTier.OPTIMIZER, include_nonpublic=True)
    popnames = [x.name for x in priv_optims]
    assert "_PrivateOptim" in popnames

    priv_scheds = torch_fw.collect_api(SemanticTier.SCHEDULER, include_nonpublic=True)
    psnames = [x.name for x in priv_scheds]
    assert "_PrivateScheduler" in psnames

    priv_loaders = torch_fw.collect_api(SemanticTier.DATALOADER, include_nonpublic=True)
    pdnames = [x.name for x in priv_loaders]
    assert "_PrivateLoader" in pdnames

    mets = torch_fw.collect_api(SemanticTier.METRIC)
    assert mets == []

    loaders = torch_fw.collect_api(SemanticTier.DATALOADER)
    dnames = [x.name for x in loaders]
    assert "DataLoader" in dnames
    assert "_PrivateLoader" not in dnames

    arrays = torch_fw.collect_api(SemanticTier.ARRAY_API)
    assert any("abs" in x.api_path for x in arrays)

    assert torch_fw.collect_api("unknown") == []

    # Branch coverage for missing fft and nn.functional
    mocker.patch.dict(
        "sys.modules",
        {
            "torch": create_module("torch", {"abs": lambda: None}),
        },
    )
    torch_fw.collect_api(SemanticTier.ARRAY_API)


def test_torch_import_error(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import torch as torch_fw

    mocker.patch.object(torch_fw, "nn", None)
    mocker.patch.object(torch_fw, "optim", None)
    mocker.patch.object(torch_fw, "data", None)

    assert torch_fw.collect_api(SemanticTier.LOSS) == []
    assert torch_fw.collect_api(SemanticTier.OPTIMIZER) == []
    assert torch_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert torch_fw.collect_api(SemanticTier.LAYER) == []
    assert torch_fw.collect_api(SemanticTier.SCHEDULER) == []
    assert torch_fw.collect_api(SemanticTier.INITIALIZER) == []
    assert torch_fw.collect_api(SemanticTier.DATALOADER) == []
    import unittest.mock as mock

    with mock.patch.dict("sys.modules", {"torch": None}):
        res = torch_fw.collect_api(SemanticTier.ARRAY_API)
        assert len(res) == 1
        assert res[0].name == "float"


def test_torch_typeerror(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import torch as torch_fw

    class BadLoss:
        """Class docstring."""

        pass

    BadLoss.__name__ = "BadLoss"

    fake_nn = create_module(
        "torch.nn",
        {
            "Module": 1,  # Causes TypeError in issubclass
            "BadLoss": BadLoss,
        },
    )
    fake_optim = create_module("torch.optim", {"Optimizer": 1, "BadOptim": BadLoss})

    mocker.patch.object(torch_fw, "nn", fake_nn)
    mocker.patch.object(torch_fw, "optim", fake_optim)

    assert torch_fw.collect_api(SemanticTier.LOSS) == []
    assert torch_fw.collect_api(SemanticTier.OPTIMIZER) == []
    assert torch_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert torch_fw.collect_api(SemanticTier.LAYER) == []
    assert torch_fw.collect_api("unknown") == []


def test_tensorflow_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import tensorflow as tf_fw

    def relu() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    class DenseLayer:
        """Class docstring."""

        pass

    class _PrivateLayer:
        """Class docstring."""

        pass

    class MSELoss:
        """Class docstring."""

        pass

    class Adam:
        """Class docstring."""

        pass

    class CosineDecay:
        """Class docstring."""

        pass

    class GlorotUniform:
        """Class docstring."""

        pass

    class Accuracy:
        """Class docstring."""

        pass

    class Dataset:
        """Class docstring."""

        pass

    class _PrivateDataset:
        """Class docstring."""

        pass

    fake_nn = create_module("tf.nn", {"relu": relu})

    fake_layers = create_module(
        "tf.keras.layers", {"DenseLayer": DenseLayer, "_PrivateLayer": _PrivateLayer}
    )
    fake_losses = create_module("tf.keras.losses", {"MSELoss": MSELoss})

    fake_schedules = create_module("schedules", {"CosineDecay": CosineDecay})

    fake_optims = create_module(
        "tf.keras.optimizers", {"Adam": Adam, "schedules": fake_schedules}
    )
    fake_inits = create_module(
        "tf.keras.initializers", {"GlorotUniform": GlorotUniform}
    )
    fake_metrics = create_module("tf.keras.metrics", {"Accuracy": Accuracy})

    fake_keras = create_module(
        "tf.keras",
        {
            "layers": fake_layers,
            "losses": fake_losses,
            "optimizers": fake_optims,
            "initializers": fake_inits,
            "metrics": fake_metrics,
        },
    )
    fake_data = create_module(
        "tf.data", {"Dataset": Dataset, "_PrivateDataset": _PrivateDataset}
    )

    fake_tf = create_module(
        "tf", {"nn": fake_nn, "keras": fake_keras, "data": fake_data}
    )

    mocker.patch.object(tf_fw, "tf", fake_tf)

    assert any(x.name == "MSELoss" for x in tf_fw.collect_api(SemanticTier.LOSS))
    assert any(x.name == "Adam" for x in tf_fw.collect_api(SemanticTier.OPTIMIZER))
    assert any(x.name == "relu" for x in tf_fw.collect_api(SemanticTier.ACTIVATION))
    assert any(x.name == "DenseLayer" for x in tf_fw.collect_api(SemanticTier.LAYER))
    assert any(
        x.name == "CosineDecay" for x in tf_fw.collect_api(SemanticTier.SCHEDULER)
    )
    assert any(
        x.name == "GlorotUniform" for x in tf_fw.collect_api(SemanticTier.INITIALIZER)
    )
    assert any(x.name == "Accuracy" for x in tf_fw.collect_api(SemanticTier.METRIC))
    assert any(x.name == "Dataset" for x in tf_fw.collect_api(SemanticTier.DATALOADER))
    assert not any(
        x.name == "_PrivateLayer" for x in tf_fw.collect_api(SemanticTier.LAYER)
    )

    # test include_nonpublic
    assert any(
        x.name == "_PrivateLayer"
        for x in tf_fw.collect_api(SemanticTier.LAYER, include_nonpublic=True)
    )
    assert any(
        x.name == "_PrivateDataset"
        for x in tf_fw.collect_api(SemanticTier.DATALOADER, include_nonpublic=True)
    )
    assert tf_fw.collect_api("unknown") == []

    # Exception branch coverage
    mocker.patch(
        "ml_framework_snapshots.frameworks.tensorflow.get_all_members",
        side_effect=Exception,
    )
    assert tf_fw.collect_api(SemanticTier.LAYER) == []

    # Test when tf module lacks submodules (nn, keras, data)
    empty_tf = create_module("tf", {})

    mocker.patch.object(tf_fw, "tf", empty_tf)
    assert tf_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert tf_fw.collect_api(SemanticTier.LAYER) == []
    assert tf_fw.collect_api(SemanticTier.LOSS) == []
    assert tf_fw.collect_api(SemanticTier.OPTIMIZER) == []
    assert tf_fw.collect_api(SemanticTier.SCHEDULER) == []
    assert tf_fw.collect_api(SemanticTier.INITIALIZER) == []
    assert tf_fw.collect_api(SemanticTier.METRIC) == []
    assert tf_fw.collect_api(SemanticTier.DATALOADER) == []

    mocker.patch.object(tf_fw, "tf", None)
    assert tf_fw.collect_api(SemanticTier.LOSS) == []


def test_keras_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import keras as keras_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    class MockMember:
        """Mock class for griffe member."""

        def __init__(self, is_class: bool = False, is_function: bool = False) -> None:
            """Initialize MockMember.

            Args:
                is_class: Is a class.
                is_function: Is a function.
            """
            self.is_class = is_class
            self.is_function = is_function

    class MockModule:
        """Mock class for griffe module."""

        def __init__(self, members: dict[str, MockMember]) -> None:
            """Initialize MockModule.

            Args:
                members: Dictionary of members.
            """
            self.members = members

    def mock_load(path: str) -> MockModule:
        """Mock griffe.load function.

        Args:
            path: Module path.

        Returns:
            MockModule: Mocked module.

        Raises:
            Exception: If path is unknown.
        """
        if path == "keras.losses":
            return MockModule({"MSELoss": MockMember(is_class=True)})
        if path == "keras.optimizers":
            return MockModule({"Adam": MockMember(is_class=True)})
        if path == "keras.optimizers.schedules":
            return MockModule({"CosineDecay": MockMember(is_class=True)})
        if path == "keras.activations":
            return MockModule({"relu": MockMember(is_function=True)})
        if path == "keras.layers":
            return MockModule({"DenseLayer": MockMember(is_class=True)})
        if path == "keras.initializers":
            return MockModule({"GlorotUniform": MockMember(is_class=True)})
        if path == "keras.metrics":
            return MockModule({"Accuracy": MockMember(is_class=True)})
        raise Exception("Unknown path")

    mocker.patch.object(keras_fw.griffe, "load", mock_load)  # type: ignore[attr-defined]

    losses = keras_fw.collect_api(SemanticTier.LOSS)
    assert any(x.name == "MSELoss" for x in losses)

    optims = keras_fw.collect_api(SemanticTier.OPTIMIZER)
    assert any(x.name == "Adam" for x in optims)

    acts = keras_fw.collect_api(SemanticTier.ACTIVATION)
    assert any(x.name == "relu" for x in acts)

    layers = keras_fw.collect_api(SemanticTier.LAYER)
    assert any(x.name == "DenseLayer" for x in layers)

    scheds = keras_fw.collect_api(SemanticTier.SCHEDULER)
    assert any(x.name == "CosineDecay" for x in scheds)

    inits = keras_fw.collect_api(SemanticTier.INITIALIZER)
    assert any(x.name == "GlorotUniform" for x in inits)

    mets = keras_fw.collect_api(SemanticTier.METRIC)
    assert any(x.name == "Accuracy" for x in mets)

    assert keras_fw.collect_api("unknown") == []

    # Test blocklist logic: module doesn't exist, block_list skipping, etc.
    assert keras_fw._scan_griffe_module("missing.module", "foo") == []

    def mock_load_metrics(path: str) -> MockModule:
        """Mock load for metrics.

        Args:
            path: Module path.

        Returns:
            MockModule: Mocked module.

        Raises:
            Exception: If path is unknown.
        """
        if path == "keras.metrics":
            return MockModule(
                {
                    "Metric": MockMember(is_class=True),
                    "_priv": MockMember(is_class=True),
                    "neither": MockMember(is_class=False, is_function=False),
                }
            )
        raise Exception("Unknown path")  # pragma: no cover

    mocker.patch.object(keras_fw.griffe, "load", mock_load_metrics)  # type: ignore[attr-defined]
    mets2 = keras_fw.collect_api(SemanticTier.METRIC)
    assert not any(x.name == "Metric" for x in mets2)

    # Trigger dummy func for coverage
    dummy = keras_fw._make_dummy_obj("test_func", "function")
    dummy()

    # Exception branch coverage
    mocker.patch.object(keras_fw.griffe, "load", side_effect=Exception)  # type: ignore[attr-defined]
    assert keras_fw.collect_api(SemanticTier.LAYER) == []

    # Missing griffe branch coverage
    keras_fw.griffe = None  # type: ignore[attr-defined, assignment]
    assert keras_fw._scan_griffe_module("keras.losses", "keras.losses") == []
    assert keras_fw.collect_api(SemanticTier.LOSS) == []


def test_mlx_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import mlx as mlx_fw

    class Dense:
        """Class docstring."""

        pass

    def relu() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def mse_loss() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    class Adam:
        """Class docstring."""

        pass

    fake_losses = create_module("losses", {"mse_loss": mse_loss})

    fake_nn = create_module(
        "mlx.nn", {"Dense": Dense, "relu": relu, "losses": fake_losses}
    )
    fake_optims = create_module("mlx.optimizers", {"Adam": Adam})

    fake_core = create_module("mlx.core", {"abs": lambda: None})

    fake_mlx = create_module(
        "mlx", {"nn": fake_nn, "optimizers": fake_optims, "core": fake_core}
    )

    mocker.patch.object(mlx_fw, "mlx", fake_mlx)

    assert any(x.name == "mse_loss" for x in mlx_fw.collect_api(SemanticTier.LOSS))
    assert any(x.name == "Adam" for x in mlx_fw.collect_api(SemanticTier.OPTIMIZER))
    assert any(x.name == "relu" for x in mlx_fw.collect_api(SemanticTier.ACTIVATION))
    assert any(x.name == "Dense" for x in mlx_fw.collect_api(SemanticTier.LAYER))
    assert any("abs" in x.api_path for x in mlx_fw.collect_api(SemanticTier.ARRAY_API))
    assert mlx_fw.collect_api("unknown") == []

    # Test mlx.nn.losses coverage gaps (not a function/class, not containing 'loss')
    fake_losses_gap = create_module(
        "losses", {"not_a_callable": 123, "other_func": lambda x: x}
    )
    mocker.patch.object(
        mlx_fw,
        "mlx",
        create_module(
            "mlx",
            {"nn": create_module("mlx.nn", {"losses": fake_losses_gap})},
        ),
    )
    assert mlx_fw.collect_api(SemanticTier.LOSS) == []

    empty_nn = create_module("mlx.nn", {})

    mocker.patch.object(mlx_fw, "mlx", create_module("mlx", {"nn": empty_nn}))

    assert mlx_fw.collect_api(SemanticTier.LOSS) == []

    mocker.patch.object(mlx_fw, "mlx", None)
    assert mlx_fw.collect_api(SemanticTier.LOSS) == []

    # Exception branch coverage
    mocker.patch(
        "ml_framework_snapshots.frameworks.tensorflow.get_all_members",
        side_effect=Exception,
    )
    assert mlx_fw.collect_api(SemanticTier.LAYER) == []


def test_jax_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import jax as jax_fw
    from ml_framework_snapshots.frameworks.optax_shim import OptaxScanner

    def relu() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def _priv() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def glorot() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    mocker.patch.object(jax_fw, "jax", True)
    fake_jax_nn = create_module(
        "jax.nn", {"relu": relu, "_priv": _priv, "not_a_func": 123}
    )
    fake_jax_init = create_module(
        "jax.nn.initializers", {"glorot": glorot, "_priv": _priv, "not_a_func": 123}
    )

    mocker.patch.dict(
        "sys.modules",
        {
            "jax": create_module("jax", {}),
            "jax.nn": fake_jax_nn,
            "jax.nn.initializers": fake_jax_init,
            "jax.numpy": create_module(
                "jax.numpy", {"abs": lambda: None, "transpose": lambda: None}
            ),
            "numpy": create_module("numpy", {"float32": 1}),
        },
    )

    mocker.patch.object(
        OptaxScanner,
        "scan_losses",
        return_value=[GhostRef(name="mse", api_path="o.mse", kind="function")],
    )
    mocker.patch.object(
        OptaxScanner,
        "scan_optimizers",
        return_value=[GhostRef(name="adam", api_path="o.a", kind="function")],
    )
    mocker.patch.object(
        OptaxScanner,
        "scan_schedulers",
        return_value=[GhostRef(name="cosine", api_path="o.c", kind="function")],
    )

    assert len(jax_fw.collect_api(SemanticTier.LOSS)) == 1
    assert len(jax_fw.collect_api(SemanticTier.OPTIMIZER)) == 1
    assert len(jax_fw.collect_api(SemanticTier.SCHEDULER)) == 1

    acts = jax_fw.collect_api(SemanticTier.ACTIVATION)
    assert any(x.name == "relu" for x in acts)

    inits = jax_fw.collect_api(SemanticTier.INITIALIZER)
    assert any(x.name == "glorot" for x in inits)

    arrays = jax_fw.collect_api(SemanticTier.ARRAY_API)
    assert any("abs" in x.api_path for x in arrays)

    import sys

    del sys.modules["jax.numpy"].transpose
    arrays2 = jax_fw.collect_api(SemanticTier.ARRAY_API)
    assert not any("transpose" in x.api_path for x in arrays2)

    assert jax_fw.collect_api("unknown") == []

    # Exception branch coverage
    mocker.patch(
        "ml_framework_snapshots.frameworks.jax.get_all_members", side_effect=Exception
    )
    assert jax_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert jax_fw.collect_api(SemanticTier.INITIALIZER) == []

    mocker.patch.object(jax_fw, "jax", None)
    assert jax_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert jax_fw.collect_api(SemanticTier.INITIALIZER) == []
    import unittest.mock as mock

    with mock.patch.dict("sys.modules", {"jax.numpy": None}):
        res = jax_fw.collect_api(SemanticTier.ARRAY_API)
        assert len(res) == 3


def test_flax_nnx_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import flax_nnx as flax_fw

    class Module:
        """Class docstring."""

        pass

    class Dense(Module):
        """Class docstring."""

        pass

    class _Priv(Module):
        """Class docstring."""

        pass

    class NotAModule:
        """Class docstring."""

        pass

    fake_nnx = create_module(
        "flax.nnx",
        {"Module": Module, "Dense": Dense, "_Priv": _Priv, "NotAModule": NotAModule},
    )

    mocker.patch.object(flax_fw, "nnx", fake_nnx)
    mocker.patch(
        "ml_framework_snapshots.frameworks.flax_nnx.jax_collect_api",
        return_value=["delegated"],
    )

    layers = flax_fw.collect_api(SemanticTier.LAYER)
    assert any(x.name == "Dense" for x in layers)
    assert not any(x.name == "Module" for x in layers)
    assert flax_fw.collect_api(SemanticTier.LOSS) == ["delegated"]
    assert flax_fw.collect_api("unknown") == []

    # Exception branch coverage
    mocker.patch(
        "ml_framework_snapshots.frameworks.flax_nnx.get_all_members",
        side_effect=Exception,
    )
    assert flax_fw.collect_api(SemanticTier.LAYER) == []

    # TypeError branch coverage
    fake_nnx.Module = 1  # type: ignore[attr-defined] # Force TypeError
    assert flax_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(flax_fw, "nnx", None)
    assert flax_fw.collect_api(SemanticTier.LAYER) == []


def test_optax_shim_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.optax_shim as o_shim

    def adam() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def sgd() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def my_optimizer() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def _priv() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def mse_loss() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def other_error() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def kl_entropy() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def _priv_loss() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def some_other_func() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    def cosine() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    fake_losses = create_module(
        "losses",
        {
            "mse_loss": mse_loss,
            "other_error": other_error,
            "kl_entropy": kl_entropy,
            "_priv_loss": _priv_loss,
            "not_a_func": 123,
            "some_other_func": some_other_func,
        },
    )

    fake_scheds = create_module(
        "schedules", {"cosine": cosine, "_priv": _priv, "not_a_func": 123}
    )

    def not_a_thing() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    fake_optax = create_module(
        "optax",
        {
            "adam": adam,
            "sgd": sgd,
            "my_optimizer": my_optimizer,
            "_priv": _priv,
            "losses": fake_losses,
            "schedules": fake_scheds,
            "not_a_thing": not_a_thing,
            "not_a_func": 123,
        },
    )

    mocker.patch.object(o_shim, "optax", fake_optax)

    optims = o_shim.OptaxScanner.scan_optimizers()
    names = [x.name for x in optims]
    assert "adam" in names
    assert "sgd" in names
    assert "my_optimizer" in names

    losses = o_shim.OptaxScanner.scan_losses()
    lnames = [x.name for x in losses]
    assert "mse_loss" in lnames
    assert "other_error" in lnames

    scheds = o_shim.OptaxScanner.scan_schedulers()
    snames = [x.name for x in scheds]
    assert "cosine" in snames

    # Exception branch coverage
    mocker.patch(
        "ml_framework_snapshots.models.GhostInspector.inspect", side_effect=Exception
    )
    assert o_shim.OptaxScanner.scan_optimizers() == []
    assert o_shim.OptaxScanner.scan_losses() == []
    assert o_shim.OptaxScanner.scan_schedulers() == []

    mocker.patch.object(o_shim, "optax", None)
    assert o_shim.OptaxScanner.scan_optimizers() == []
    assert o_shim.OptaxScanner.scan_losses() == []
    assert o_shim.OptaxScanner.scan_schedulers() == []


def test_sklearn_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import sklearn as sklearn_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import types

    class BaseEstimator:
        """Class docstring."""

        pass

    class RandomForestClassifier(BaseEstimator):
        """Class docstring."""

        def __init__(self, n_estimators: int = 100) -> Any:  # type: ignore
            """Function docstring.

            Args:
                n_estimators: description
            """
            pass  # pragma: no cover

    class NotAnEstimator:
        """Class docstring."""

        pass

    def accuracy_score(y_true: Any, y_pred: Any) -> Any:
        """Function docstring.

        Args:
            y_true: description
            y_pred: description
        """
        pass  # pragma: no cover

    mock_sklearn = types.ModuleType("sklearn")
    mock_sklearn.base = types.ModuleType("sklearn.base")  # type: ignore
    mock_sklearn.base.BaseEstimator = BaseEstimator  # type: ignore[attr-defined]

    mock_ensemble = types.ModuleType("sklearn.ensemble")
    mock_ensemble.RandomForestClassifier = RandomForestClassifier  # type: ignore[attr-defined]
    mock_ensemble.NotAnEstimator = NotAnEstimator  # type: ignore

    mock_metrics = types.ModuleType("sklearn.metrics")
    mock_metrics.accuracy_score = accuracy_score  # type: ignore

    sys_modules = {
        "sklearn": mock_sklearn,
        "sklearn.base": mock_sklearn.base,
        "sklearn.ensemble": mock_ensemble,
        "sklearn.metrics": mock_metrics,
    }
    for name in [
        "linear_model",
        "tree",
        "svm",
        "neighbors",
        "cluster",
        "decomposition",
        "manifold",
        "preprocessing",
        "feature_extraction",
        "pipeline",
        "compose",
    ]:
        sys_modules[f"sklearn.{name}"] = types.ModuleType(f"sklearn.{name}")

    mocker.patch.dict("sys.modules", sys_modules)
    mocker.patch.object(sklearn_fw, "sklearn", mock_sklearn)
    mocker.patch.object(sklearn_fw, "BaseEstimator", BaseEstimator)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER, include_nonpublic=False)
    assert len(layers) == 1
    assert layers[0].name == "RandomForestClassifier"

    metrics = sklearn_fw.collect_api(SemanticTier.METRIC, include_nonpublic=False)
    assert len(metrics) == 1
    assert metrics[0].name == "accuracy_score"

    empty = sklearn_fw.collect_api(SemanticTier.LOSS)
    assert empty == []


def test_sklearn_import_error(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import sklearn as sklearn_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    mocker.patch.object(sklearn_fw, "sklearn", None)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER)
    assert layers == []


def test_sklearn_module_import_error(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import sklearn as sklearn_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import types
    import builtins

    mock_sklearn = types.ModuleType("sklearn")
    mocker.patch.object(sklearn_fw, "sklearn", mock_sklearn)

    original_import = builtins.__import__

    def mock_import(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Function docstring.

        Args:
            name: description
            args: description
            kwargs: description


        Raises:
            ImportError: Exception.

        Returns:
            Return value.
        """
        if name.startswith("sklearn."):  # pragma: no branch
            raise ImportError("Mocked import error")
        return original_import(name, *args, **kwargs)  # pragma: no cover

    mocker.patch("builtins.__import__", side_effect=mock_import)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER)
    assert layers == []

    metrics = sklearn_fw.collect_api(SemanticTier.METRIC)
    assert metrics == []


def test_sklearn_scan_module_edge_cases(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks.sklearn import _scan_module
    import types

    # 43: not module
    assert _scan_module(None, "prefix") == []

    # 50-51: Exception during getmembers
    mock_mod = types.ModuleType("mock_mod")

    # To cause getmembers to fail, we can make an attribute raise Exception
    class BadObj:
        """Class docstring."""

        @property
        def bad(self) -> Any:
            """Function docstring.

            Raises:
                RuntimeError: Exception.
            """
            raise RuntimeError("Bad")  # pragma: no cover

    mock_mod.bad_obj = BadObj()  # type: ignore
    mocker.patch(
        "ml_framework_snapshots.frameworks.tensorflow.get_all_members",
        side_effect=Exception("mocked"),
    )
    assert _scan_module(mock_mod, "prefix") == []

    # 55: private or block_list
    mock_mod2 = types.ModuleType("mock_mod2")

    class ValidObj:
        """Class docstring."""

        pass

    mock_mod2._private = ValidObj  # type: ignore
    mock_mod2.blocked = ValidObj  # type: ignore

    mocker.patch(
        "ml_framework_snapshots.frameworks.tensorflow.get_all_members",
        return_value=[("_private", ValidObj), ("blocked", ValidObj)],
    )
    mocker.patch("inspect.isclass", return_value=True)
    assert _scan_module(mock_mod2, "prefix", block_list={"blocked"}) == []


def test_sklearn_scan_module_branches(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks.sklearn import _scan_module
    import types

    mock_mod = types.ModuleType("mock_mod")

    class ValidObj:
        """Class docstring."""

        pass

    def valid_func() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    # 1. kind="class", is_estimator=False
    mock_mod.ValidObj = ValidObj  # type: ignore
    mock_mod.valid_func = valid_func  # type: ignore

    mocker.patch(
        "ml_framework_snapshots.frameworks.tensorflow.get_all_members",
        return_value=[
            ("ValidObj", ValidObj),
            ("valid_func", valid_func),
            ("other", 123),
        ],
    )

    res = _scan_module(mock_mod, "prefix", kind="class", is_estimator=False)
    assert len(res) == 1

    res = _scan_module(mock_mod, "prefix", kind="function")
    assert len(res) == 1


def test_numpy_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.numpy as np_shim
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    from ml_framework_snapshots.models import GhostInspector

    def tanh() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    def exp() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    fake_np = create_module(
        "numpy",
        {
            "tanh": tanh,
            "exp": exp,
            "maximum": 123,
            "minimum": lambda: None,
            "not_callable": 123,
        },
    )
    mocker.patch.object(np_shim, "np", fake_np)

    original_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, api_path: Any, **kwargs: Any) -> Any:
        """Docstring.

        Args:
            **kwargs: Parameter.
            api_path: Parameter.
            obj: Parameter.

        Raises:
            ValueError: Exception.

        Returns:
            Return value.
        """
        if "minimum" in api_path:
            raise ValueError("mock error")
        return original_inspect(obj, api_path, **kwargs)

    mocker.patch(
        "ml_framework_snapshots.frameworks.numpy.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = np_shim.collect_api(SemanticTier.ACTIVATION)
    names = [x.name for x in res]
    assert "tanh" in names
    assert "exp" in names
    assert "not_callable" not in names

    # Empty
    res2 = np_shim.collect_api(SemanticTier.LAYER)
    assert not res2

    # Import error
    mocker.patch.object(np_shim, "np", None)
    assert not np_shim.collect_api(SemanticTier.ACTIVATION)


def test_orbax_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.orbax_checkpoint as ocp_shim
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    from ml_framework_snapshots.models import GhostInspector

    def checkpoint() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    def _priv() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    class Checkpointer:
        """Docstring."""

        pass

    def error_func() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    fake_ocp = create_module(
        "orbax.checkpoint",
        {
            "checkpoint": checkpoint,
            "_priv": _priv,
            "Checkpointer": Checkpointer,
            "error_func": error_func,
            "not_callable": 123,
        },
    )
    mocker.patch.object(ocp_shim, "ocp", fake_ocp)

    original_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, api_path: Any, **kwargs: Any) -> Any:
        """Docstring.

        Args:
            **kwargs: Parameter.
            api_path: Parameter.
            obj: Parameter.

        Raises:
            ValueError: Exception.

        Returns:
            Return value.
        """
        if "error_func" in api_path:
            raise ValueError("mock error")
        return original_inspect(obj, api_path, **kwargs)

    mocker.patch(
        "ml_framework_snapshots.frameworks.orbax_checkpoint.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = ocp_shim.collect_api(SemanticTier.ARRAY_API)
    names = [x.name for x in res]
    assert "checkpoint" in names
    assert "Checkpointer" in names
    assert "_priv" not in names

    # Include non-public
    res_all = ocp_shim.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    names_all = [x.name for x in res_all]
    assert "_priv" in names_all

    # Empty
    res2 = ocp_shim.collect_api(SemanticTier.LAYER)
    assert not res2

    # Import error
    mocker.patch.object(ocp_shim, "ocp", None)
    assert not ocp_shim.collect_api(SemanticTier.ARRAY_API)


def test_pax_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.pax as pax_shim
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    from ml_framework_snapshots.models import GhostInspector

    class Linear:
        """Docstring."""

        pass

    class _PrivLayer:
        """Docstring."""

        pass

    def not_a_class() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    class ErrorClass:
        """Docstring."""

        pass

    fake_layers = create_module(
        "praxis.layers",
        {
            "Linear": Linear,
            "_PrivLayer": _PrivLayer,
            "not_a_class": not_a_class,
            "ErrorClass": ErrorClass,
            "num": 123,
        },
    )
    mocker.patch.object(pax_shim, "layers", fake_layers)

    original_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, api_path: Any, **kwargs: Any) -> Any:
        """Docstring.

        Args:
            **kwargs: Parameter.
            api_path: Parameter.
            obj: Parameter.

        Raises:
            ValueError: Exception.

        Returns:
            Return value.
        """
        if "ErrorClass" in api_path:
            raise ValueError("mock error")
        return original_inspect(obj, api_path, **kwargs)

    mocker.patch(
        "ml_framework_snapshots.frameworks.pax.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = pax_shim.collect_api(SemanticTier.NEURAL)
    names = [x.name for x in res]
    assert "Linear" in names
    assert "_PrivLayer" not in names

    # Empty
    res2 = pax_shim.collect_api(SemanticTier.ACTIVATION)
    assert not res2

    # Import error
    mocker.patch.object(pax_shim, "layers", None)
    assert not pax_shim.collect_api(SemanticTier.NEURAL)


def test_triton_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.triton as triton_shim
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import importlib

    def cdiv() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    fake_triton = create_module("triton", {"cdiv": cdiv, "_priv": lambda: None})

    fake_tl = create_module("triton.language", {"cdiv": cdiv, "_priv": lambda: None})

    original_import = importlib.import_module

    def mock_import(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Docstring.

        Args:
            **kwargs: Parameter.
            *args: Parameter.
            name: Parameter.

        Raises:
            ImportError: Exception.

        Returns:
            Return value.
        """
        if name == "triton":
            return fake_triton
        elif name == "triton.language":  # pragma: no branch
            return fake_tl
        elif name == "triton_fail":  # pragma: no cover
            raise ImportError("mock error")  # pragma: no cover
        return original_import(name, *args, **kwargs)  # pragma: no cover

    mocker.patch("importlib.import_module", side_effect=mock_import)

    res = triton_shim.collect_api(SemanticTier.UTIL)
    assert "cdiv" in [x.name for x in res]

    # Import error
    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not triton_shim.collect_api(SemanticTier.UTIL)


def test_deepspeed_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.deepspeed as ds_shim
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import importlib

    def initialize() -> Any:
        """Docstring."""
        pass  # pragma: no cover

    fake_ds = create_module("deepspeed", {"initialize": initialize})

    original_import = importlib.import_module

    def mock_import(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Docstring.

        Args:
            **kwargs: Parameter.
            *args: Parameter.
            name: Parameter.

        Returns:
            Return value.
        """
        if name == "deepspeed":  # pragma: no branch
            return fake_ds
        return original_import(name, *args, **kwargs)  # pragma: no cover

    mocker.patch("importlib.import_module", side_effect=mock_import)
    res = ds_shim.collect_api(SemanticTier.MODEL)
    assert "initialize" in [x.name for x in res]

    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not ds_shim.collect_api(SemanticTier.MODEL)


def test_onnxruntime_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.onnxruntime as ort_shim
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import importlib

    class InferenceSession:
        """Docstring."""

        pass

    fake_ort = create_module("onnxruntime", {"InferenceSession": InferenceSession})

    original_import = importlib.import_module

    def mock_import(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Docstring.

        Args:
            **kwargs: Parameter.
            *args: Parameter.
            name: Parameter.

        Returns:
            Return value.
        """
        if name == "onnxruntime":  # pragma: no branch
            return fake_ort
        return original_import(name, *args, **kwargs)  # pragma: no cover

    mocker.patch("importlib.import_module", side_effect=mock_import)
    res = ort_shim.collect_api(SemanticTier.MODEL)
    assert "InferenceSession" in [x.name for x in res]

    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not ort_shim.collect_api(SemanticTier.MODEL)


def test_huggingface_collect(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_framework_snapshots.frameworks.huggingface as hf_shim
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import importlib

    class PreTrainedModel:
        """Docstring."""

        pass

    class Trainer:
        """Docstring."""

        pass

    fake_transformers = create_module(
        "transformers", {"PreTrainedModel": PreTrainedModel, "Trainer": Trainer}
    )

    original_import = importlib.import_module

    def mock_import(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Docstring.

        Args:
            **kwargs: Parameter.
            *args: Parameter.
            name: Parameter.

        Returns:
            Return value.
        """
        if name == "transformers":  # pragma: no branch
            return fake_transformers
        return original_import(name, *args, **kwargs)  # pragma: no cover

    mocker.patch("importlib.import_module", side_effect=mock_import)

    res = hf_shim.collect_transformers(SemanticTier.MODEL)
    names = [x.name for x in res]
    assert "PreTrainedModel" in names

    res2 = hf_shim.collect_transformers(SemanticTier.UTIL)
    assert "Trainer" in [x.name for x in res2]

    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not hf_shim.collect_transformers(SemanticTier.MODEL)


def test_maxtext_collect(mocker: Any) -> None:
    """Test the maxtext API collector.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import maxtext as maxtext_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    class FakeMaxtext:
        """Fake Maxtext module for testing."""

        __path__ = ["fake/path"]

    mocker.patch.object(maxtext_fw, "maxtext", FakeMaxtext())

    import textwrap

    mocker.patch("glob.glob", return_value=["fake/path/models/fake_model.py"])

    mock_file_content = textwrap.dedent("""
        class FakeModel:
            def __init__(self, arg1, *, kwarg1=1):
                pass
        class NotAModel:
            pass
    """)

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))

    models = maxtext_fw.collect_api(SemanticTier.MODEL)
    assert len(models) == 2
    names = [x.name for x in models]
    assert "FakeModel" in names
    assert "NotAModel" in names

    assert maxtext_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(maxtext_fw, "maxtext", None)
    assert maxtext_fw.collect_api(SemanticTier.MODEL) == []

    # test parser exception
    mocker.patch.object(maxtext_fw, "maxtext", FakeMaxtext())
    mocker.patch("builtins.open", side_effect=Exception)
    assert maxtext_fw.collect_api(SemanticTier.MODEL) == []

    # test missing __path__
    class FakeMaxtextNoPath:
        """Fake Maxtext module without __path__."""

        pass

    mocker.patch.object(maxtext_fw, "maxtext", FakeMaxtextNoPath())
    assert maxtext_fw.collect_api(SemanticTier.MODEL) == []


def test_mlir_collect(mocker: Any) -> None:
    """Test the mlir API collector.

    Args:
        mocker: Parameter.
    """
    from ml_framework_snapshots.frameworks import mlir as mlir_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    class FakeMlir:
        """Fake MLIR module for testing."""

        __path__ = ["fake/path"]

    mocker.patch.object(mlir_fw, "mlir_dialects", FakeMlir())

    import textwrap

    mocker.patch("glob.glob", return_value=["fake/path/_arith_ops_gen.py"])

    mock_file_content = textwrap.dedent("""
        class AddFOp:
            OPERATION_NAME = "arith.addf"
            def __init__(self, lhs, *, loc=None):
                pass
        class NotAnOp:
            pass
        class PartialOp:
            OPERATION_NAME = "arith.partial"
            # missing __init__
        class WeirdOp:
            # Operation name is not a constant
            OPERATION_NAME = get_name()
            # Operation target is something else
            some_other_var = "value"
    """)

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))
    mocker.patch(
        "glob.glob",
        return_value=[
            "fake/path/_arith_ops_gen.py",
            "fake/path/not_ops_gen.py",
            "fake/path/not_started_with_us_ops_gen.py",
        ],
    )

    ops = mlir_fw.collect_api(SemanticTier.UTIL)
    assert len(ops) == 2
    names = [x.name for x in ops]
    assert "AddFOp" in names
    assert "PartialOp" in names

    assert mlir_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(mlir_fw, "mlir_dialects", None)
    assert mlir_fw.collect_api(SemanticTier.UTIL) == []

    # test parser exception
    mocker.patch.object(mlir_fw, "mlir_dialects", FakeMlir())
    mocker.patch("builtins.open", side_effect=Exception)
    assert mlir_fw.collect_api(SemanticTier.UTIL) == []

    # test missing __path__
    class FakeMlirNoPath:
        """Fake MLIR module without __path__."""

        pass

    mocker.patch.object(mlir_fw, "mlir_dialects", FakeMlirNoPath())
    assert mlir_fw.collect_api(SemanticTier.UTIL) == []


def test_static_dsl_extractors() -> None:
    """Test static DSL extractors."""
    from ml_framework_snapshots.frameworks import html_dsl, latex_dsl, tikz, nvidia_sass
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    html_refs = html_dsl.collect_api(SemanticTier.UTIL)
    assert len(html_refs) > 0
    assert any(r.name == "div" for r in html_refs)
    assert html_dsl.collect_api(SemanticTier.LAYER) == []

    latex_refs = latex_dsl.collect_api(SemanticTier.UTIL)
    assert len(latex_refs) > 0
    assert any(r.name == "frac" for r in latex_refs)
    assert latex_dsl.collect_api(SemanticTier.LAYER) == []

    tikz_refs = tikz.collect_api(SemanticTier.UTIL)
    assert len(tikz_refs) > 0
    assert any(r.name == "draw" for r in tikz_refs)
    assert any(r.name == "circle" for r in tikz_refs)
    assert tikz.collect_api(SemanticTier.LAYER) == []

    sass_refs = nvidia_sass.collect_api(SemanticTier.UTIL)
    assert len(sass_refs) > 0
    assert any(r.name == "FADD" for r in sass_refs)
    assert any(r.name == "MOV" for r in sass_refs)
    assert nvidia_sass.collect_api(SemanticTier.LAYER) == []
