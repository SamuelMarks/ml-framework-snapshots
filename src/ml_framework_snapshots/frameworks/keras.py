"""Keras API Snapshot Extractor.

Provides functions to dynamically introspect the Keras library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List, Any, Optional, Set

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

try:
    import keras
except ImportError:  # pragma: no cover
    keras = None


def _scan_module(
    module: Any,
    prefix: str,
    kind: str = "class",
    block_list: Optional[Set[str]] = None,
    include_nonpublic: bool = False,
) -> List[GhostRef]:
    """Reflectively scans a Keras module for members of a specific kind.

    Args:
        module: The live Python module to inspect.
        prefix: The import prefix for the generated API path.
        kind: Expected kind ("class" or "function").
        block_list: A set of names to exclude from the scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing the discovered API members.

    """
    if not module:
        return []
    block_list = block_list or set()
    found = []

    try:
        for name, obj in get_all_members(module):
            if (not include_nonpublic and name.startswith("_")) or name in block_list:
                continue

            if kind == "class" and inspect.isclass(obj):
                found.append(GhostInspector.inspect(obj, f"{prefix}.{name}"))
            elif kind == "function" and inspect.isfunction(obj):
                found.append(GhostInspector.inspect(obj, f"{prefix}.{name}"))
    except Exception:  # pragma: no cover
        pass
    return found


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> List[GhostRef]:
    """Scan the live Keras library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.

    """
    if not keras:
        return []

    results = []
    if category == SemanticTier.LOSS:
        results.extend(
            _scan_module(
                getattr(keras, "losses", None),
                "keras.losses",
                kind="class",
                block_list={"Loss", "Container"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.OPTIMIZER:
        results.extend(
            _scan_module(
                getattr(keras, "optimizers", None),
                "keras.optimizers",
                kind="class",
                block_list={"Optimizer", "TFOptimizer"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.ACTIVATION:
        results.extend(
            _scan_module(
                getattr(keras, "activations", None),
                "keras.activations",
                kind="function",
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.LAYER:
        results.extend(
            _scan_module(
                getattr(keras, "layers", None),
                "keras.layers",
                kind="class",
                block_list={"Layer"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.SCHEDULER:
        results.extend(
            _scan_module(
                getattr(getattr(keras, "optimizers", None), "schedules", None),
                "keras.optimizers.schedules",
                kind="class",
                block_list={"LearningRateSchedule"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.INITIALIZER:
        results.extend(
            _scan_module(
                getattr(keras, "initializers", None),
                "keras.initializers",
                kind="class",
                block_list={"Initializer"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.METRIC:
        results.extend(
            _scan_module(
                getattr(keras, "metrics", None),
                "keras.metrics",
                kind="class",
                block_list={"Metric"},
                include_nonpublic=include_nonpublic,
            )
        )

    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the Keras API signature for a given category.

    Args:
        category: The category of API to collect (e.g., LOSS).
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    return _collect_live(category, include_nonpublic)
