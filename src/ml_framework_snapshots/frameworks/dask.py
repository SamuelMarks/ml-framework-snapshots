"""Dask API Snapshot Extractor."""

from typing import List
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostRef
from ..models import GhostInspector

import typing

try:
    import dask.array as _da

    da: typing.Any = _da  # pragma: no cover
except ImportError:
    da = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect dask API.

    Args:
        category: Parameter.
        include_nonpublic: Parameter.
    """
    results: List[GhostRef] = []
    if not da:
        return results

    if category == SemanticTier.ARRAY_API:
        for name in dir(da):
            if not include_nonpublic and name.startswith("_"):
                continue
            obj = getattr(da, name)
            if callable(obj):
                try:
                    res = GhostInspector.inspect(
                        obj, f"dask.array.{name}", is_public=not name.startswith("_")
                    )
                    results.append(res)
                except Exception:
                    pass

    return results
