"""TensorFlow API Snapshot Extractor.

Provides functions to dynamically introspect the TensorFlow library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> List[GhostRef]:
    """Scan the live TensorFlow library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.

    """
    results: list[GhostRef] = []
    if not tf:
        return results

    try:
        if category == SemanticTier.ACTIVATION:
            target_names = {
                "relu",
                "sigmoid",
                "tanh",
                "softmax",
                "leaky_relu",
                "elu",
                "selu",
            }
            if hasattr(tf, "nn"):
                for name in target_names:
                    if hasattr(tf.nn, name):
                        results.append(
                            GhostInspector.inspect(
                                getattr(tf.nn, name), f"tf.nn.{name}"
                            )
                        )

        elif category == SemanticTier.LAYER:
            is_keras_available = hasattr(tf, "keras") and hasattr(tf.keras, "layers")
            if is_keras_available:
                for name, obj in get_all_members(tf.keras.layers):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.layers.{name}")
                        )

        elif category == SemanticTier.LOSS:
            is_keras_available = hasattr(tf, "keras") and hasattr(tf.keras, "losses")
            if is_keras_available:
                for name, obj in get_all_members(tf.keras.losses):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.losses.{name}")
                        )

        elif category == SemanticTier.OPTIMIZER:
            is_keras_available = hasattr(tf, "keras") and hasattr(
                tf.keras, "optimizers"
            )
            if is_keras_available:
                for name, obj in get_all_members(tf.keras.optimizers):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.optimizers.{name}")
                        )

        elif category == SemanticTier.SCHEDULER:
            is_keras_available = (
                hasattr(tf, "keras")
                and hasattr(tf.keras, "optimizers")
                and hasattr(tf.keras.optimizers, "schedules")
            )
            if is_keras_available:
                for name, obj in get_all_members(tf.keras.optimizers.schedules):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(
                                obj, f"tf.keras.optimizers.schedules.{name}"
                            )
                        )

        elif category == SemanticTier.INITIALIZER:
            is_keras_available = hasattr(tf, "keras") and hasattr(
                tf.keras, "initializers"
            )
            if is_keras_available:
                for name, obj in get_all_members(tf.keras.initializers):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.initializers.{name}")
                        )

        elif category == SemanticTier.METRIC:
            is_keras_available = hasattr(tf, "keras") and hasattr(tf.keras, "metrics")
            if is_keras_available:
                for name, obj in get_all_members(tf.keras.metrics):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.metrics.{name}")
                        )

        elif category == SemanticTier.DATALOADER:
            if hasattr(tf, "data"):
                for name, obj in get_all_members(tf.data):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(GhostInspector.inspect(obj, f"tf.data.{name}"))

    except Exception:  # pragma: no cover
        pass

    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the TensorFlow API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    return _collect_live(category, include_nonpublic)
