"""MLX API Snapshot Extractor.

Provides functions to dynamically introspect the MLX library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

try:
    import mlx.core  # pragma: no cover
    import mlx.nn  # pragma: no cover
    import mlx.optimizers  # pragma: no cover
except ImportError:  # pragma: no cover
    mlx = None


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> List[GhostRef]:
    """Scan the live MLX library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.

    """
    results: list[GhostRef] = []
    if not mlx:
        return results

    try:
        if category == SemanticTier.LAYER:
            for name, obj in get_all_members(mlx.nn):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and inspect.isclass(obj)
                    and name[0].isupper()
                ):
                    results.append(GhostInspector.inspect(obj, f"mlx.nn.{name}"))

        elif category == SemanticTier.ACTIVATION:
            target_names = {
                "relu",
                "gelu",
                "silu",
                "sigmoid",
                "tanh",
                "softmax",
                "elu",
            }
            for name, obj in get_all_members(mlx.nn):
                if (
                    include_nonpublic or not name.startswith("_")
                ) and name.lower() in target_names:
                    results.append(GhostInspector.inspect(obj, f"mlx.nn.{name}"))

        elif category == SemanticTier.LOSS:
            if hasattr(mlx.nn, "losses"):
                for name, obj in get_all_members(mlx.nn.losses):
                    if not include_nonpublic and name.startswith("_"):
                        continue
                    if inspect.isfunction(obj) or inspect.isclass(obj):
                        if "loss" in name.lower():
                            results.append(
                                GhostInspector.inspect(obj, f"mlx.nn.losses.{name}")
                            )

        elif category == SemanticTier.OPTIMIZER:
            for name, obj in get_all_members(mlx.optimizers):
                if (
                    inspect.isclass(obj)
                    and (include_nonpublic or not name.startswith("_"))
                    and name[0].isupper()
                ):
                    results.append(
                        GhostInspector.inspect(obj, f"mlx.optimizers.{name}")
                    )
    except Exception:  # pragma: no cover
        pass

    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the MLX API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    return _collect_live(category, include_nonpublic)
