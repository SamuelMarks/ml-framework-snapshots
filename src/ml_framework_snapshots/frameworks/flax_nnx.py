"""Flax (NNX) API Snapshot Extractor.

Provides functions to dynamically introspect the Flax NNX library and
generate GhostRefs for layers. Defers to core JAX extractors for
losses, optimizers, and activations.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.frameworks.jax import collect_api as jax_collect_api

try:
    import flax.nnx as nnx
except ImportError:  # pragma: no cover
    nnx = None  # type: ignore


def _scan_nnx_layers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `flax.nnx` module for classes inheriting from `nnx.Module`.

    Excludes the base `Module` class itself.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found NNX layers.

    """
    if not nnx:
        return []

    found = []
    try:
        for name, obj in get_all_members(nnx):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isclass(obj) and name != "Module":
                # Check inheritance gracefully (some objects might not be types)
                try:
                    if issubclass(obj, nnx.Module):
                        found.append(GhostInspector.inspect(obj, f"flax.nnx.{name}"))
                except TypeError:  # pragma: no cover
                    pass
    except Exception:  # pragma: no cover
        pass

    return found


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the Flax API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    # Use core JAX scanning for losses, optimizers, activations
    if category in [
        SemanticTier.LOSS,
        SemanticTier.OPTIMIZER,
        SemanticTier.ACTIVATION,
        SemanticTier.SCHEDULER,
        SemanticTier.INITIALIZER,
        SemanticTier.METRIC,
        SemanticTier.DATALOADER,
    ]:
        return jax_collect_api(category, include_nonpublic)

    # Add Flax-specific neural layers
    if category == SemanticTier.LAYER:
        return _scan_nnx_layers(include_nonpublic)

    return []
