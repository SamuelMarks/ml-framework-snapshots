"""Onnxruntime module."""

import importlib
from typing import List
from ml_switcheroo.enums import SemanticTier
from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo.core.ghost import GhostRef


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect onnxruntime API.

    Args:
        category: description
        include_nonpublic: description

    Returns:
        List of GhostRefs.
    """
    results: List[GhostRef] = []

    # We mainly target models/InferenceSession and Utils
    target_cats = {SemanticTier.MODEL, SemanticTier.UTIL}
    if category not in target_cats:
        return results

    try:
        mod = importlib.import_module("onnxruntime")
    except ImportError:
        return results

    inspector = GhostInspector()

    for name in dir(mod):
        if not include_nonpublic and name.startswith("_"):
            continue

        obj = getattr(mod, name, None)
        if obj is None:  # pragma: no cover
            continue

        obj_cat = SemanticTier.MODEL if "Session" in name else SemanticTier.UTIL

        if obj_cat == category:
            try:
                ref = inspector.inspect(obj, f"onnxruntime.{name}")
                if ref:  # pragma: no branch
                    # Introspect execution providers from the signature or dynamically
                    # Usually providers is a kwarg in InferenceSession.__init__
                    if name == "InferenceSession":  # pragma: no branch
                        # Ensure 'providers' is in parameters
                        has_providers = any(p.name == "providers" for p in ref.params)
                        if not has_providers:  # pragma: no branch
                            from ml_switcheroo.core.ghost import GhostParam

                            ref.params.append(
                                GhostParam(
                                    name="providers",
                                    kind="POSITIONAL_OR_KEYWORD",
                                    default="None",
                                    annotation="Sequence[str | tuple[str, dict[str, Any]]] | None",
                                )
                            )
                    results.append(ref)
            except Exception:  # pragma: no cover
                pass

    return results
