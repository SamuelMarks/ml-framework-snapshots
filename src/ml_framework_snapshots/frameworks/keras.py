"""Keras API Snapshot Extractor.

Provides functions to statically introspect the Keras library using Griffe and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from typing import List, Any, Optional, Set

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

try:
    import griffe
except ImportError:  # pragma: no cover
    griffe = None  # type: ignore[assignment]


def _make_dummy_obj(name: str, kind: str) -> Any:
    if kind == "class":
        return type(name, (), {})
    else:

        def dummy_func() -> None:
            """Dummy function."""
            pass

        dummy_func.__name__ = name
        return dummy_func


def _scan_griffe_module(
    module_path: str,
    prefix: str,
    kind: str = "class",
    block_list: Optional[Set[str]] = None,
    include_nonpublic: bool = False,
) -> List[GhostRef]:
    """Statically scans a Keras module for members of a specific kind.

    Args:
        module_path: The griffe module path to inspect (e.g. 'keras.losses').
        prefix: The import prefix for the generated API path.
        kind: Expected kind ("class" or "function").
        block_list: A set of names to exclude from the scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing the discovered API members.
    """
    if not griffe:
        return []

    block_list = block_list or set()
    found = []

    try:
        mod = griffe.load(module_path)
    except Exception:
        return []

    for name, member in mod.members.items():
        if (not include_nonpublic and name.startswith("_")) or name in block_list:
            continue

        is_class = member.is_class
        is_function = member.is_function

        if kind == "class" and is_class:
            obj = _make_dummy_obj(name, "class")
            found.append(GhostInspector.inspect(obj, f"{prefix}.{name}"))
        elif kind == "function" and is_function:
            obj = _make_dummy_obj(name, "function")
            found.append(GhostInspector.inspect(obj, f"{prefix}.{name}"))

    return found


def _collect_static(category: SemanticTier, include_nonpublic: bool) -> List[GhostRef]:
    """Scan the Keras library for a specific API category via Griffe.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.
    """
    if not griffe:
        return []

    results = []
    if category == SemanticTier.LOSS:
        results.extend(
            _scan_griffe_module(
                "keras.losses",
                "keras.losses",
                kind="class",
                block_list={"Loss", "Container"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.OPTIMIZER:
        results.extend(
            _scan_griffe_module(
                "keras.optimizers",
                "keras.optimizers",
                kind="class",
                block_list={"Optimizer", "TFOptimizer"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.ACTIVATION:
        results.extend(
            _scan_griffe_module(
                "keras.activations",
                "keras.activations",
                kind="function",
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.LAYER:
        results.extend(
            _scan_griffe_module(
                "keras.layers",
                "keras.layers",
                kind="class",
                block_list={"Layer"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.SCHEDULER:
        results.extend(
            _scan_griffe_module(
                "keras.optimizers.schedules",
                "keras.optimizers.schedules",
                kind="class",
                block_list={"LearningRateSchedule"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.INITIALIZER:
        results.extend(
            _scan_griffe_module(
                "keras.initializers",
                "keras.initializers",
                kind="class",
                block_list={"Initializer"},
                include_nonpublic=include_nonpublic,
            )
        )
    elif category == SemanticTier.METRIC:
        results.extend(
            _scan_griffe_module(
                "keras.metrics",
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
    return _collect_static(category, include_nonpublic)
