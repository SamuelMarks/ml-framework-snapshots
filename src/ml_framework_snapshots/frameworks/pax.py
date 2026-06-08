import inspect
from typing import List
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
from ml_framework_snapshots.models import GhostInspector
from ml_framework_snapshots.utils import get_all_members

try:
    import paxml
    import praxis
    from praxis import layers
except ImportError:
    paxml = None
    praxis = None
    layers = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    results: List[GhostRef] = []
    if not layers:
        return results

    if category == SemanticTier.LAYER:
        try:
            for name, obj in get_all_members(layers):
                if not include_nonpublic and name.startswith("_"):
                    continue
                if inspect.isclass(obj):
                    results.append(GhostInspector.inspect(obj, f"praxis.layers.{name}"))
        except Exception:
            pass
    return results
