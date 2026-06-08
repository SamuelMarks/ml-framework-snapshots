"""Optax Scanner Logic.

This module provides introspection for the Optax library to power the "Ghost Protocol"
discovery. Optax uses a functional API where optimizers and losses are functions
returning named tuples or callables, rather than Classes.

Capabilities:
1.  Scan `optax.losses` for loss functions.
2.  Scans root `optax` for optimizer factory functions.
3.  Filters internal utilities to provide clean Abstract Standard candidates.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List

try:
    import optax
except ImportError:  # pragma: no cover
    optax = None

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef


class OptaxScanner:
    """Helper to inspect Optax APIs for the discovery system."""

    @staticmethod
    def scan_optimizers(include_nonpublic: bool = False) -> List[GhostRef]:
        """Scan the Optax public API for optimizer functions.

        Optax optimizers are typically factory functions (e.g., `adam`, `sgd`)
        that return a `GradientTransformation`.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of GhostRef objects representing found optimizers.

        """
        if not optax:
            return []

        results = []
        known_optimizers = {
            "adam",
            "adamw",
            "sgd",
            "rmsprop",
            "adagrad",
            "lamb",
            "lion",
            "novograd",
            "fromage",
            "yogi",
            "adabelief",
        }

        for name, obj in get_all_members(optax):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj) or inspect.isclass(obj):
                is_known = name.lower() in known_optimizers
                is_suffixed = name.endswith("_optimizer")

                if is_known or is_suffixed:
                    try:
                        ref = GhostInspector.inspect(obj, f"optax.{name}")
                        results.append(ref)
                    except Exception:  # pragma: no cover
                        pass

        return results

    @staticmethod
    def scan_losses(include_nonpublic: bool = False) -> List[GhostRef]:
        """Scan `optax.losses` for loss functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of GhostRef objects representing found losses.

        """
        if not optax or not hasattr(optax, "losses"):
            return []

        results = []

        for name, obj in get_all_members(optax.losses):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj):
                name_lower = name.lower()
                if (
                    "loss" in name_lower
                    or "error" in name_lower
                    or "entropy" in name_lower
                ):
                    try:
                        ref = GhostInspector.inspect(obj, f"optax.losses.{name}")
                        results.append(ref)
                    except Exception:  # pragma: no cover
                        pass

        return results

    @staticmethod
    def scan_schedulers(include_nonpublic: bool = False) -> List[GhostRef]:
        """Scan `optax.schedules` for scheduler functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of GhostRef objects representing found schedulers.

        """
        if not optax or not hasattr(optax, "schedules"):
            return []

        results = []
        for name, obj in get_all_members(optax.schedules):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj):
                try:
                    ref = GhostInspector.inspect(obj, f"optax.schedules.{name}")
                    results.append(ref)
                except Exception:  # pragma: no cover
                    pass

        return results
