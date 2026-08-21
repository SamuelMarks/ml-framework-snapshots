"""CuPy API Snapshot Extractor."""

from typing import List
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
from ml_framework_snapshots.models import GhostInspector

import typing

try:
    import cupy as _cp

    cp: typing.Any = _cp
except ImportError:  # pragma: no cover
    cp = None  # pragma: no cover


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect cupy API.

    Args:
        category: Parameter.
        include_nonpublic: Parameter.
    """
    results: List[GhostRef] = []
    if not cp:
        return results

    if category == SemanticTier.ACTIVATION:
        for name in ["tanh", "exp", "maximum", "minimum"]:
            if hasattr(cp, name):
                obj = getattr(cp, name)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"cupy.{name}", is_public=True
                        )
                        results.append(res)
                    except Exception:
                        pass
    elif category == SemanticTier.ARRAY_API:
        for name in dir(cp):
            if not include_nonpublic and name.startswith("_"):
                continue
            obj = getattr(cp, name)
            if callable(obj):
                try:
                    res = GhostInspector.inspect(
                        obj, f"cupy.{name}", is_public=not name.startswith("_")
                    )
                    results.append(res)
                except Exception:
                    pass

    return results
