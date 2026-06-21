"""PyTorch API Snapshot Extractor.

Provides functions to dynamically introspect the PyTorch library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List
from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

try:
    import torch.nn as nn
    import torch.optim as optim  # pragma: no cover
    import torch.utils.data as data  # pragma: no cover
except ImportError:  # pragma: no cover
    nn = None  # type: ignore
    optim = None  # type: ignore
    data = None  # type: ignore


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
    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    if name in [
                        "ReLU",
                        "Sigmoid",
                        "Tanh",
                        "GELU",
                        "SiLU",
                        "Softmax",
                        "LeakyReLU",
                    ]:
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
    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    if not name.endswith("Loss") and name not in [
                        "ReLU",
                        "Sigmoid",
                        "Tanh",
                        "GELU",
                        "SiLU",
                        "Softmax",
                        "LeakyReLU",
                    ]:
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
    """Scan for metrics (returns empty for base PyTorch).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found metrics.

    """
    return []


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
    return []
