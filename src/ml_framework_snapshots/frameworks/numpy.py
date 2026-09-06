"""Numpy API Snapshot Extractor."""

import inspect
from typing import List
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
from ml_framework_snapshots.models import GhostInspector

import typing

try:
    import numpy as _np

    np: typing.Any = _np
except ImportError:  # pragma: no cover
    np = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect numpy API.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRefs.
    """
    results: List[GhostRef] = []
    if not np:
        return results

    if category == SemanticTier.ACTIVATION:
        # Numpy doesn't have an "activation" module per se, but we can capture basic math ops
        for name in ["tanh", "exp", "maximum", "minimum"]:  # pragma: no branch
            if hasattr(np, name):  # pragma: no branch
                obj = getattr(np, name)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"numpy.{name}", is_public=True
                        )
                        results.append(res)
                    except Exception:
                        pass
    elif category == SemanticTier.ARRAY_API:
        array_ops = [
            "abs",
            "add",
            "all",
            "any",
            "arange",
            "argmax",
            "argmin",
            "clip",
            "concatenate",
            "cos",
            "divide",
            "dot",
            "empty",
            "exp",
            "eye",
            "full",
            "linspace",
            "log",
            "matmul",
            "max",
            "maximum",
            "mean",
            "min",
            "minimum",
            "multiply",
            "ones",
            "prod",
            "reshape",
            "round",
            "sin",
            "split",
            "sqrt",
            "squeeze",
            "stack",
            "subtract",
            "sum",
            "tan",
            "tanh",
            "transpose",
            "where",
            "zeros",
        ]
        for name in array_ops:
            if hasattr(np, name):
                obj = getattr(np, name)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"numpy.{name}", is_public=True
                        )
                        results.append(res)
                    except Exception:  # pragma: no cover
                        pass

        # Introspect numpy.linalg.*
        if hasattr(np, "linalg"):
            for name in dir(np.linalg):
                if not name.startswith("_"):
                    obj = getattr(np.linalg, name)
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj, f"numpy.linalg.{name}", is_public=True
                                )
                            )
                        except Exception:  # pragma: no cover
                            pass

        # Introspect numpy.fft.*
        if hasattr(np, "fft"):
            for name in dir(np.fft):
                if not name.startswith("_"):
                    obj = getattr(np.fft, name)
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj, f"numpy.fft.{name}", is_public=True
                                )
                            )
                        except Exception:  # pragma: no cover
                            pass
    return results
