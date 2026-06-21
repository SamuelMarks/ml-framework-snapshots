"""Scikit-Learn API Snapshot Extractor.

Provides functions to dynamically introspect the Scikit-Learn library and generate
GhostRefs for estimators, transformers, metrics, and other components.
"""

import inspect
from ml_framework_snapshots.utils import get_all_members
from typing import List, Any, Optional, Set

from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

try:
    import sklearn
    from sklearn.base import BaseEstimator
except ImportError:  # pragma: no cover
    sklearn = None  # type: ignore
    BaseEstimator = None  # type: ignore


def _scan_module(
    module: Any,
    prefix: str,
    kind: str = "class",
    block_list: Optional[Set[str]] = None,
    include_nonpublic: bool = False,
    is_estimator: bool = False,
) -> List[GhostRef]:
    """Reflectively scans a Scikit-Learn module.

    Args:
        module: The live Python module to inspect.
        prefix: The import prefix for the generated API path.
        kind: Expected kind ("class" or "function").
        block_list: A set of names to exclude from the scan.
        include_nonpublic: Whether to include non-public APIs.
        is_estimator: If True, only subclasses of BaseEstimator are collected.

    Returns:
        A list of GhostRef objects representing the discovered API members.

    """
    if not module:
        return []
    block_list = block_list or set()
    found = []

    try:
        # Some sklearn modules can be missing attributes or lazy loaded
        members = get_all_members(module)
    except Exception:
        return []

    for name, obj in members:
        if (not include_nonpublic and name.startswith("_")) or name in block_list:
            continue

        if kind == "class" and inspect.isclass(obj):
            if is_estimator and BaseEstimator is not None:
                if not issubclass(obj, BaseEstimator):
                    continue
            found.append(
                GhostInspector.inspect(
                    obj, f"{prefix}.{name}", is_public=not name.startswith("_")
                )
            )
        elif kind == "function" and inspect.isfunction(obj):
            found.append(
                GhostInspector.inspect(
                    obj, f"{prefix}.{name}", is_public=not name.startswith("_")
                )
            )

    return found


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect all API signatures for a given category from Scikit-Learn.

    Args:
        category: The SemanticTier to extract.
        include_nonpublic: Whether to include non-public components.

    Returns:
        A list of populated GhostRef objects.

    """
    global sklearn
    if not sklearn:  # pragma: no cover
        return []

    results = []

    if category == SemanticTier.LAYER:
        try:
            import sklearn.ensemble as sk_ensemble
            import sklearn.linear_model as sk_linear_model
            import sklearn.tree as sk_tree
            import sklearn.svm as sk_svm
            import sklearn.neighbors as sk_neighbors
            import sklearn.cluster as sk_cluster
            import sklearn.decomposition as sk_decomposition
            import sklearn.manifold as sk_manifold
            import sklearn.preprocessing as sk_preprocessing
            import sklearn.feature_extraction as sk_feature_extraction
            import sklearn.pipeline as sk_pipeline
            import sklearn.compose as sk_compose
        except ImportError:
            return []

        modules_to_scan = [
            (sk_ensemble, "sklearn.ensemble"),
            (sk_linear_model, "sklearn.linear_model"),
            (sk_tree, "sklearn.tree"),
            (sk_svm, "sklearn.svm"),
            (sk_neighbors, "sklearn.neighbors"),
            (sk_cluster, "sklearn.cluster"),
            (sk_decomposition, "sklearn.decomposition"),
            (sk_manifold, "sklearn.manifold"),
            (sk_preprocessing, "sklearn.preprocessing"),
            (sk_feature_extraction, "sklearn.feature_extraction"),
            (sk_pipeline, "sklearn.pipeline"),
            (sk_compose, "sklearn.compose"),
        ]
        for mod, prefix in modules_to_scan:
            results.extend(
                _scan_module(
                    mod,
                    prefix,
                    "class",
                    include_nonpublic=include_nonpublic,
                    is_estimator=True,
                )
            )

    elif category == SemanticTier.METRIC:
        try:
            import sklearn.metrics as sk_metrics
        except ImportError:
            return []

        results.extend(
            _scan_module(
                sk_metrics,
                "sklearn.metrics",
                "function",
                include_nonpublic=include_nonpublic,
            )
        )

    return results
