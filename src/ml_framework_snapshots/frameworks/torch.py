"""PyTorch API Snapshot Extractor.

Provides functions to dynamically introspect the PyTorch library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List, Set
from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

import typing

try:
    import torch.nn as _nn
    import torch.optim as _optim  # pragma: no cover
    import torch.utils.data as _data  # pragma: no cover

    nn: typing.Any = _nn  # pragma: no cover
    optim: typing.Any = _optim  # pragma: no cover
    data: typing.Any = _data  # pragma: no cover
except ImportError:  # pragma: no cover
    nn = None
    optim = None
    data = None


def _scan_losses(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn` for loss classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found losses.

    """
    if not nn:
        return []
    found = []
    for name, obj in get_all_members(nn):
        if inspect.isclass(obj) and name.endswith("Loss") and name != "_Loss":
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _scan_optimizers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.optim` for optimizer classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found optimizers.

    """
    if not optim:
        return []
    found = []
    for name, obj in get_all_members(optim):
        if inspect.isclass(obj) and name != "Optimizer":
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, optim.Optimizer):
                    found.append(GhostInspector.inspect(obj, f"torch.optim.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _get_activation_names() -> Set[str]:
    """Retrieve all activation module names from torch.nn.modules.activation with fallback."""
    names: Set[str] = set()
    try:
        import torch.nn.modules.activation as act_mod

        for name, obj in get_all_members(act_mod):
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                names.add(name)
    except Exception:
        pass
    if not names:
        names = {
            "ReLU",
            "Sigmoid",
            "Tanh",
            "GELU",
            "SiLU",
            "Softmax",
            "LeakyReLU",
        }
    return names


def _scan_activations(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn` for activation functions.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found activations.

    """
    if not nn:
        return []
    found = []
    target_activations = _get_activation_names()

    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module) and name in target_activations:
                    found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _scan_layers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn` for layer classes (excluding losses and activations).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found layers.

    """
    if not nn:
        return []
    found = []
    target_activations = _get_activation_names()

    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    if not name.endswith("Loss") and name not in target_activations:
                        found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _scan_schedulers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.optim.lr_scheduler` for schedulers.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found schedulers.

    """
    if not optim or not hasattr(optim, "lr_scheduler"):
        return []
    found = []
    for name, obj in get_all_members(optim.lr_scheduler):
        if inspect.isclass(obj) and name != "LRScheduler":
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(
                GhostInspector.inspect(obj, f"torch.optim.lr_scheduler.{name}")
            )
    return found


def _scan_initializers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn.init` for initializers.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found initializers.

    """
    if not nn or not hasattr(nn, "init"):
        return []
    found = []
    for name, obj in get_all_members(nn.init):
        if inspect.isfunction(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(GhostInspector.inspect(obj, f"torch.nn.init.{name}"))
    return found


def _scan_metrics(include_nonpublic: bool) -> List[GhostRef]:
    """Scan for metrics (delegates to torchmetrics if installed, otherwise returns empty).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found metrics.

    """
    found = []
    try:
        import torchmetrics

        for name, obj in get_all_members(torchmetrics):
            if inspect.isclass(obj) and (include_nonpublic or not name.startswith("_")):
                try:
                    found.append(GhostInspector.inspect(obj, f"torchmetrics.{name}"))
                except Exception:  # pragma: no cover
                    pass
    except ImportError:
        pass
    return found


def _scan_dataloaders(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.utils.data` for dataloader-related classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found dataloaders.

    """
    if not data:
        return []
    found = []
    for name, obj in get_all_members(data):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(GhostInspector.inspect(obj, f"torch.utils.data.{name}"))
    return found


def _scan_array_api(include_nonpublic: bool) -> List[GhostRef]:
    """Scan top-level PyTorch module for array API functions and submodules.

    Exhaustively discovers:
        - All public operations in `torch.*`
        - `torch.linalg.*`
        - `torch.special.*`
        - `torch.fft.*`
        - `torch.nn.functional.*` (convolutions, pooling, linear, activations, attention, loss)

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found array API functions.
    """
    found = []
    try:
        import torch

        # 1. Top-level array API functions
        for name, obj in get_all_members(torch):
            if not include_nonpublic and name.startswith("_"):
                continue
            if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj):
                try:
                    found.append(GhostInspector.inspect(obj, f"torch.{name}"))
                except Exception:  # pragma: no cover
                    pass

        # 2. linalg module
        if hasattr(torch, "linalg"):
            for name, obj in get_all_members(torch.linalg):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and callable(obj)
                    and not inspect.isclass(obj)
                ):
                    try:
                        found.append(
                            GhostInspector.inspect(obj, f"torch.linalg.{name}")
                        )
                    except Exception:  # pragma: no cover
                        pass

        # 3. special module
        if hasattr(torch, "special"):
            for name, obj in get_all_members(torch.special):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and callable(obj)
                    and not inspect.isclass(obj)
                ):
                    try:
                        found.append(
                            GhostInspector.inspect(obj, f"torch.special.{name}")
                        )
                    except Exception:  # pragma: no cover
                        pass

        # 4. fft module
        if hasattr(torch, "fft"):
            for name, obj in get_all_members(torch.fft):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(obj, f"torch.fft.{name}")
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 5. Exhaustive nn.functional module
        if hasattr(torch, "nn") and hasattr(torch.nn, "functional"):
            for name, obj in get_all_members(torch.nn.functional):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(
                                    obj, f"torch.nn.functional.{name}"
                                )
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 6. autograd functions
        if hasattr(torch, "autograd"):
            for name, obj in get_all_members(torch.autograd):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(obj, f"torch.autograd.{name}")
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 7. distributed functions
        if hasattr(torch, "distributed"):
            for name, obj in get_all_members(torch.distributed):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(obj, f"torch.distributed.{name}")
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 8. torch.Tensor instance methods
        if hasattr(torch, "Tensor"):
            for name, obj in inspect.getmembers(torch.Tensor):
                if not include_nonpublic and name.startswith("_"):
                    continue
                if callable(obj) and not inspect.isclass(obj):
                    try:
                        found.append(
                            GhostInspector.inspect(
                                obj, f"torch.Tensor.{name}", kind="method"
                            )
                        )
                    except Exception:  # pragma: no cover
                        pass

    except ImportError:
        pass

    return found


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the PyTorch API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    if category == SemanticTier.LOSS:
        return _scan_losses(include_nonpublic)
    elif category == SemanticTier.OPTIMIZER:
        return _scan_optimizers(include_nonpublic)
    elif category == SemanticTier.ACTIVATION:
        return _scan_activations(include_nonpublic)
    elif category == SemanticTier.LAYER:
        return _scan_layers(include_nonpublic)
    elif category == SemanticTier.SCHEDULER:
        return _scan_schedulers(include_nonpublic)
    elif category == SemanticTier.INITIALIZER:
        return _scan_initializers(include_nonpublic)
    elif category == SemanticTier.METRIC:
        return _scan_metrics(include_nonpublic)
    elif category == SemanticTier.DATALOADER:
        return _scan_dataloaders(include_nonpublic)
    elif category == SemanticTier.ARRAY_API:
        return _scan_array_api(include_nonpublic)
    return []
