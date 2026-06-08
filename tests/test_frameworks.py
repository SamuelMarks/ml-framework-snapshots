"""Module docstring."""

import types
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostRef


def create_module(name, attrs):
    """Function docstring.

    Args:
        name: description
        attrs: description
    """
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_torch_collect(mocker):
    """Function docstring."""
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

    def xavier_uniform_():
        """Function docstring."""
        pass

    def _private_init():
        """Function docstring."""
        pass

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

    assert torch_fw.collect_api("unknown") == []


def test_torch_import_error(mocker):
    """Function docstring."""
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


def test_torch_typeerror(mocker):
    """Function docstring."""
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


def test_tensorflow_collect(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks import tensorflow as tf_fw

    def relu():
        """Function docstring."""
        pass

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


def test_keras_collect(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks import keras as keras_fw

    class MSELoss:
        """Class docstring."""

        pass

    class Adam:
        """Class docstring."""

        pass

    def relu():
        """Function docstring."""
        pass

    class DenseLayer:
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

    fake_keras = create_module(
        "keras",
        {
            "losses": create_module("losses", {"MSELoss": MSELoss}),
            "optimizers": create_module(
                "optimizers",
                {
                    "Adam": Adam,
                    "schedules": create_module(
                        "schedules", {"CosineDecay": CosineDecay}
                    ),
                },
            ),
            "activations": create_module("activations", {"relu": relu}),
            "layers": create_module("layers", {"DenseLayer": DenseLayer}),
            "initializers": create_module(
                "initializers", {"GlorotUniform": GlorotUniform}
            ),
            "metrics": create_module("metrics", {"Accuracy": Accuracy}),
        },
    )

    mocker.patch.object(keras_fw, "keras", fake_keras)

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
    assert keras_fw._scan_module(None, "foo") == []

    class Metric:
        """Class docstring."""

        pass

    fake_metrics2 = create_module("metrics", {"Metric": Metric, "_priv": Metric})
    mocker.patch.object(
        keras_fw, "keras", create_module("keras", {"metrics": fake_metrics2})
    )
    mets2 = keras_fw.collect_api(SemanticTier.METRIC)
    assert not any(x.name == "Metric" for x in mets2)

    # Exception branch coverage
    mocker.patch(
        "ml_framework_snapshots.frameworks.tensorflow.get_all_members",
        side_effect=Exception,
    )
    assert keras_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(keras_fw, "keras", None)
    assert keras_fw.collect_api(SemanticTier.LOSS) == []


def test_mlx_collect(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks import mlx as mlx_fw

    class Dense:
        """Class docstring."""

        pass

    def relu():
        """Function docstring."""
        pass

    def mse_loss():
        """Function docstring."""
        pass

    class Adam:
        """Class docstring."""

        pass

    fake_losses = create_module("losses", {"mse_loss": mse_loss})
    fake_nn = create_module(
        "mlx.nn", {"Dense": Dense, "relu": relu, "losses": fake_losses}
    )
    fake_optims = create_module("mlx.optimizers", {"Adam": Adam})
    fake_mlx = create_module("mlx", {"nn": fake_nn, "optimizers": fake_optims})

    mocker.patch.object(mlx_fw, "mlx", fake_mlx)

    assert any(x.name == "mse_loss" for x in mlx_fw.collect_api(SemanticTier.LOSS))
    assert any(x.name == "Adam" for x in mlx_fw.collect_api(SemanticTier.OPTIMIZER))
    assert any(x.name == "relu" for x in mlx_fw.collect_api(SemanticTier.ACTIVATION))
    assert any(x.name == "Dense" for x in mlx_fw.collect_api(SemanticTier.LAYER))
    assert mlx_fw.collect_api("unknown") == []

    # Test mlx.nn.losses coverage gaps (not a function/class, not containing 'loss')
    fake_losses_gap = create_module(
        "losses", {"not_a_callable": 123, "other_func": lambda x: x}
    )
    mocker.patch.object(
        mlx_fw,
        "mlx",
        create_module(
            "mlx", {"nn": create_module("mlx.nn", {"losses": fake_losses_gap})}
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


def test_jax_collect(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks import jax as jax_fw
    from ml_framework_snapshots.frameworks.optax_shim import OptaxScanner

    def relu():
        """Function docstring."""
        pass

    def _priv():
        """Function docstring."""
        pass

    def glorot():
        """Function docstring."""
        pass

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


def test_flax_nnx_collect(mocker):
    """Function docstring."""
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
    fake_nnx.Module = 1  # Force TypeError
    assert flax_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(flax_fw, "nnx", None)
    assert flax_fw.collect_api(SemanticTier.LAYER) == []


def test_optax_shim_collect(mocker):
    """Function docstring."""
    import ml_framework_snapshots.frameworks.optax_shim as o_shim

    def adam():
        """Function docstring."""
        pass

    def sgd():
        """Function docstring."""
        pass

    def my_optimizer():
        """Function docstring."""
        pass

    def _priv():
        """Function docstring."""
        pass

    def mse_loss():
        """Function docstring."""
        pass

    def other_error():
        """Function docstring."""
        pass

    def kl_entropy():
        """Function docstring."""
        pass

    def _priv_loss():
        """Function docstring."""
        pass

    def some_other_func():
        """Function docstring."""
        pass

    def cosine():
        """Function docstring."""
        pass

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

    def not_a_thing():
        """Function docstring."""
        pass

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


def test_sklearn_collect(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks import sklearn as sklearn_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import types

    class BaseEstimator:
        """Class docstring."""

        pass

    class RandomForestClassifier(BaseEstimator):
        """Class docstring."""

        def __init__(self, n_estimators: int = 100):
            """Function docstring.

            Args:
                n_estimators: description
            """
            pass

    class NotAnEstimator:
        """Class docstring."""

        pass

    def accuracy_score(y_true, y_pred):
        """Function docstring.

        Args:
            y_true: description
            y_pred: description
        """
        pass

    mock_sklearn = types.ModuleType("sklearn")
    mock_sklearn.base = types.ModuleType("sklearn.base")
    mock_sklearn.base.BaseEstimator = BaseEstimator

    mock_ensemble = types.ModuleType("sklearn.ensemble")
    mock_ensemble.RandomForestClassifier = RandomForestClassifier
    mock_ensemble.NotAnEstimator = NotAnEstimator

    mock_metrics = types.ModuleType("sklearn.metrics")
    mock_metrics.accuracy_score = accuracy_score

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


def test_sklearn_import_error(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks import sklearn as sklearn_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    mocker.patch.object(sklearn_fw, "sklearn", None)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER)
    assert layers == []


def test_sklearn_module_import_error(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks import sklearn as sklearn_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    import types
    import builtins

    mock_sklearn = types.ModuleType("sklearn")
    mocker.patch.object(sklearn_fw, "sklearn", mock_sklearn)

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Function docstring.

        Args:
            name: description
            args: description
            kwargs: description
        """
        if name.startswith("sklearn."):
            raise ImportError("Mocked import error")
        return original_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER)
    assert layers == []

    metrics = sklearn_fw.collect_api(SemanticTier.METRIC)
    assert metrics == []


def test_sklearn_scan_module_edge_cases(mocker):
    """Function docstring."""
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
        def bad(self):
            """Function docstring."""
            raise RuntimeError("Bad")

    mock_mod.bad_obj = BadObj()
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

    mock_mod2._private = ValidObj
    mock_mod2.blocked = ValidObj

    mocker.patch(
        "ml_framework_snapshots.frameworks.tensorflow.get_all_members",
        return_value=[("_private", ValidObj), ("blocked", ValidObj)],
    )
    mocker.patch("inspect.isclass", return_value=True)
    assert _scan_module(mock_mod2, "prefix", block_list={"blocked"}) == []


def test_sklearn_scan_module_branches(mocker):
    """Function docstring."""
    from ml_framework_snapshots.frameworks.sklearn import _scan_module
    import types

    mock_mod = types.ModuleType("mock_mod")

    class ValidObj:
        """Class docstring."""

        pass

    def valid_func():
        """Function docstring."""
        pass

    # 1. kind="class", is_estimator=False
    mock_mod.ValidObj = ValidObj
    mock_mod.valid_func = valid_func

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
