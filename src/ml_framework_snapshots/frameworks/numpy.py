from typing import List
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
from ml_framework_snapshots.models import GhostInspector

try:
    import numpy as np
except ImportError:
    np = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    results: List[GhostRef] = []
    if not np:
        return results

    if category == SemanticTier.ACTIVATION:
        # Numpy doesn't have an "activation" module per se, but we can capture basic math ops
        for name in ["tanh", "exp", "maximum", "minimum"]:
            if hasattr(np, name):
                obj = getattr(np, name)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"numpy.{name}", is_public=True
                        )
                        results.append(res)
                    except Exception:
                        pass
    return results
