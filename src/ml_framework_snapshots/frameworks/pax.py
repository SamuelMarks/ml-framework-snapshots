"""Pax API Snapshot Extractor."""

from typing import List
import inspect
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
from ml_framework_snapshots.models import GhostInspector
from ml_framework_snapshots.utils import get_all_members

try:
    import paxml
    import praxis  # pragma: no cover
    from praxis import layers  # pragma: no cover
except ImportError:  # pragma: no cover
    paxml = None  # type: ignore # pragma: no cover
    praxis = None  # type: ignore # pragma: no cover
    layers = None  # type: ignore # pragma: no cover


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect pax API.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRefs.
    """
    results: List[GhostRef] = []
    if not layers:
        return results

    if category == SemanticTier.NEURAL:
        for name, obj in get_all_members(layers):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isclass(obj):
                try:
                    results.append(GhostInspector.inspect(obj, f"praxis.layers.{name}"))
                except Exception:
                    pass
    return results
