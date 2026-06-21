"""JAX API Snapshot Extractor.

Provides functions to dynamically introspect the JAX and Optax libraries and
generate GhostRefs for activations, losses, and optimizers.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.frameworks.optax_shim import OptaxScanner

try:
    import jax
except ImportError:  # pragma: no cover
    jax = None  # type: ignore


def _scan_jax_activations(include_nonpublic: bool) -> List[GhostRef]:
    """Dynamically scans jax.nn for activation-like functions.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found activation functions.

    """
    if jax is None:
        return []
    found = []
    try:
        import jax.nn as jax_nn

        for name, obj in get_all_members(jax_nn):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isfunction(obj):
                found.append(GhostInspector.inspect(obj, f"jax.nn.{name}"))
    except Exception:  # pragma: no cover
        pass
    return found


def _scan_jax_initializers(include_nonpublic: bool) -> List[GhostRef]:
    """Dynamically scans jax.nn.initializers for initializer functions.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found initializer functions.

    """
    if jax is None:
        return []
    found = []
    try:
        import jax.nn.initializers as jax_init

        for name, obj in get_all_members(jax_init):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isfunction(obj):
                found.append(GhostInspector.inspect(obj, f"jax.nn.initializers.{name}"))
    except Exception:  # pragma: no cover
        pass
    return found


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> List[GhostRef]:
    """Scan the live JAX library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.

    """
    results = []
    if category == SemanticTier.LOSS:
        results.extend(OptaxScanner.scan_losses(include_nonpublic))
    elif category == SemanticTier.OPTIMIZER:
        results.extend(OptaxScanner.scan_optimizers(include_nonpublic))
    elif category == SemanticTier.ACTIVATION:
        results.extend(_scan_jax_activations(include_nonpublic))
    elif category == SemanticTier.SCHEDULER:
        results.extend(OptaxScanner.scan_schedulers(include_nonpublic))
    elif category == SemanticTier.INITIALIZER:
        results.extend(_scan_jax_initializers(include_nonpublic))
    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the JAX API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    return _collect_live(category, include_nonpublic)
