"""JAX API Snapshot Extractor.

Provides functions to dynamically introspect the JAX and Optax libraries and
generate GhostRefs for activations, losses, and optimizers.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import Any, List
import typing

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.frameworks.optax_shim import OptaxScanner

try:
    import jax as _jax

    jax: typing.Any = _jax  # pragma: no cover
except Exception:  # pragma: no cover
    jax = None


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


def _attach_jax_static_arg_metadata(ref: GhostRef, obj: Any) -> GhostRef:
    """Attach static argument numbers and names from JIT-compiled functions as metadata.

    Args:
        ref: GhostRef representation of function.
        obj: Live JAX function or JIT wrapper.

    Returns:
        Augmented GhostRef with static argument metadata.
    """
    static_nums = getattr(obj, "_static_argnums", None) or getattr(
        obj, "static_argnums", None
    )
    static_names = getattr(obj, "_static_argnames", None) or getattr(
        obj, "static_argnames", None
    )

    if static_nums:
        ref.environment_tags.append(
            f"static_argnums:{','.join(map(str, sorted(list(static_nums))))}"
        )
        for idx, p in enumerate(ref.params):
            if idx in static_nums:
                p.standardized_name = "static_arg"

    if static_names:
        ref.environment_tags.append(
            f"static_argnames:{','.join(map(str, sorted(list(static_names))))}"
        )
        for p in ref.params:
            if p.name in static_names:
                p.standardized_name = "static_arg"

    return ref


def _scan_array_api(include_nonpublic: bool) -> List[GhostRef]:
    """Scan JAX for array API functions, primitives, random generation, and transforms.

    Args:
        include_nonpublic: Include non-public APIs.

    Returns:
        List of GhostRefs.
    """
    found = []
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        # 1. Exhaustive jax.numpy.*
        for name, obj in get_all_members(jnp):
            if not include_nonpublic and name.startswith("_"):
                continue
            if callable(obj) and not inspect.isclass(obj):
                try:
                    ref = GhostInspector.inspect(obj, f"jax.numpy.{name}")
                    found.append(_attach_jax_static_arg_metadata(ref, obj))
                except Exception:  # pragma: no cover
                    pass

        # 2. Core primitive operations in jax.lax.*
        if hasattr(jax, "lax"):
            for name, obj in get_all_members(jax.lax):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and callable(obj)
                    and not inspect.isclass(obj)
                ):
                    try:
                        ref = GhostInspector.inspect(obj, f"jax.lax.{name}")
                        found.append(_attach_jax_static_arg_metadata(ref, obj))
                    except Exception:  # pragma: no cover
                        pass

        # 3. Random generation APIs in jax.random.*
        if hasattr(jax, "random"):
            for name, obj in get_all_members(jax.random):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and callable(obj)
                    and not inspect.isclass(obj)
                ):
                    try:
                        found.append(GhostInspector.inspect(obj, f"jax.random.{name}"))
                    except Exception:  # pragma: no cover
                        pass

        # 4. Transformations (jax.jit, jax.grad, jax.vmap, jax.pmap, jax.checkpoint)
        for transform_name in ["jit", "grad", "vmap", "pmap", "checkpoint"]:
            obj = getattr(jax, transform_name, None)
            if obj and callable(obj):
                try:
                    found.append(GhostInspector.inspect(obj, f"jax.{transform_name}"))
                except Exception:  # pragma: no cover
                    pass

        obj = getattr(jnp, "transpose", None)
        if obj and not any(r.name == "transpose" for r in found):
            found.append(GhostInspector.inspect(obj, "jnp.transpose"))

        found.append(GhostInspector.inspect(np.float32, "jax.numpy.float32"))

        # 5. jax.Array member methods
        if hasattr(jax, "Array"):
            for name, obj in inspect.getmembers(jax.Array):
                if not include_nonpublic and name.startswith("_"):
                    continue
                if callable(obj) and not inspect.isclass(obj):
                    try:
                        found.append(
                            GhostInspector.inspect(
                                obj, f"jax.Array.{name}", kind="method"
                            )
                        )
                    except Exception:  # pragma: no cover
                        pass
    except ImportError:
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
    elif category == SemanticTier.ARRAY_API:
        results.extend(_scan_array_api(include_nonpublic))
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
