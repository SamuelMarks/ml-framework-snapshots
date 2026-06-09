"""Deepspeed module."""

import importlib
from typing import List
from ml_switcheroo.enums import SemanticTier
from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo.core.ghost import GhostRef


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect deepspeed API.

    Args:
        category: description
        include_nonpublic: description

    Returns:
        List of GhostRefs.
    """
    results: List[GhostRef] = []

    # DeepSpeed mainly provides training, inference, and utilities.
    # We categorize 'initialize' and config-related things to MODEL or UTIL.
    if category not in (SemanticTier.MODEL, SemanticTier.UTIL):
        return results

    try:
        mod = importlib.import_module("deepspeed")
    except ImportError:
        return results

    inspector = GhostInspector()

    for name in dir(mod):
        if not include_nonpublic and name.startswith("_"):
            continue

        obj = getattr(mod, name, None)
        if obj is None:  # pragma: no cover
            continue

        obj_cat = (
            SemanticTier.MODEL
            if name in ("initialize", "DeepSpeedEngine")
            else SemanticTier.UTIL
        )

        if obj_cat == category:
            try:
                ref = inspector.inspect(obj, f"deepspeed.{name}")
                if ref:  # pragma: no branch
                    if name == "initialize":  # pragma: no branch
                        # Map distributed configuration dictionaries into structured elements.
                        has_config = any(p.name == "config_params" for p in ref.params)
                        if not has_config:  # pragma: no branch
                            from ml_switcheroo.core.ghost import GhostParam

                            ref.params.append(
                                GhostParam(
                                    name="config_params",
                                    kind="POSITIONAL_OR_KEYWORD",
                                    default="None",
                                    annotation="dict | str | None",
                                )
                            )
                    results.append(ref)
            except Exception:  # pragma: no cover
                pass

    return results
