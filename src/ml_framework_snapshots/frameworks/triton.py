"""Triton module."""

import importlib
from typing import List, Any
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef


def _extract_triton_kernel(
    obj: Any, name: str, module_name: str, inspector: GhostInspector
) -> GhostRef:
    """Extract triton kernel.

    Args:
        obj: description
        name: description
        module_name: description
        inspector: description

    Returns:
        The GhostRef.
    """
    # A triton JIT kernel often wraps the original function in `obj.fn` or `obj.src`
    fn = obj
    if hasattr(obj, "fn"):  # pragma: no branch
        fn = obj.fn
    else:  # pragma: no cover
        pass

    ref = inspector.inspect(fn, f"{module_name}.{name}")
    if ref:
        # Check for constexpr hints. Sometimes it's in annotations.
        for param in ref.params:
            if (
                hasattr(fn, "__annotations__") and param.name in fn.__annotations__
            ):  # pragma: no branch
                anno = fn.__annotations__[param.name]
                if "constexpr" in str(anno):
                    param.annotation = "tl.constexpr"
    return ref


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect triton API.

    Args:
        category: description
        include_nonpublic: description

    Returns:
        List of GhostRefs.
    """
    results: List[GhostRef] = []
    if category != SemanticTier.UTIL:
        return results

    try:
        importlib.import_module("triton")
    except ImportError:  # pragma: no cover
        return results

    inspector = GhostInspector()

    # Introspect triton language (tl) and kernels
    try:
        tl = importlib.import_module("triton.language")
        for name in dir(tl):
            if not include_nonpublic and name.startswith("_"):
                continue
            obj = getattr(tl, name, None)
            if obj and callable(obj):  # pragma: no branch
                ref = _extract_triton_kernel(obj, name, "triton.language", inspector)
                if ref:
                    results.append(ref)
    except ImportError:  # pragma: no cover
        pass

    return results
