"""Huggingface module."""

import inspect
from typing import Dict, List, Any
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostRef, GhostParam
from ml_framework_snapshots.models import GhostInspector, sanitize_type_str


def _extract_generation_kwargs(obj: Any, ref: GhostRef) -> None:
    """Extract generation kwargs.

    Args:
        obj: description
        ref: description
    """
    gen_method = getattr(obj, "generate")
    try:
        sig = inspect.signature(gen_method)
        for param in sig.parameters.values():
            if param.name == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if not any(p.name == param.name for p in ref.params):
                default_str = (
                    str(param.default)
                    if param.default is not inspect.Parameter.empty
                    else None
                )
                anno_str = (
                    str(param.annotation)
                    if param.annotation is not inspect.Parameter.empty
                    else None
                )
                anno_str = sanitize_type_str(anno_str) if anno_str else None
                ref.params.append(
                    GhostParam(
                        name=param.name,
                        kind="KEYWORD_ONLY",
                        default=default_str,
                        annotation=anno_str,
                    )
                )
    except Exception:  # pragma: no cover
        pass


def _parse_pretrained_config(obj: Any, ref: GhostRef) -> None:
    """Parse pretrained config.

    Args:
        obj: description
        ref: description
    """
    if not hasattr(obj, "__annotations__"):  # pragma: no cover
        return

    for (
        attr_name,
        attr_type,
    ) in obj.__annotations__.items():  # pragma: no branch  # pragma: no branch
        if not any(p.name == attr_name for p in ref.params):  # pragma: no branch
            anno_str = str(attr_type) if not isinstance(attr_type, str) else attr_type
            final_anno = sanitize_type_str(anno_str) if anno_str else None
            ref.params.append(
                GhostParam(
                    name=attr_name,
                    kind="POSITIONAL_OR_KEYWORD",
                    default=None,
                    annotation=final_anno,
                )
            )


def _handle_automodel_factory(obj: Any, name: str, ref: GhostRef) -> None:
    """Handle auto model factory.

    Args:
        obj: description
        name: description
        ref: description
    """
    if not any(p.name == "config" for p in ref.params):  # pragma: no branch
        ref.params.append(
            GhostParam(
                name="config",
                kind="POSITIONAL_OR_KEYWORD",
                default=None,
                annotation="PreTrainedConfig",
            )
        )


def collect_huggingface(
    module_name: str,
    category_mapping: Dict[SemanticTier, List[str]],
    category: SemanticTier,
    include_nonpublic: bool = False,
) -> List[GhostRef]:
    """Collect huggingface APIs.

    Args:
        module_name: description
        category_mapping: description
        category: description
        include_nonpublic: description

    Returns:
        List of GhostRefs.
    """
    import importlib

    results: List[GhostRef] = []

    # Check if this category is handled
    if category not in category_mapping:
        return results

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return results

    inspector = GhostInspector()

    for name in dir(mod):
        if not include_nonpublic and name.startswith("_"):
            continue

        try:
            obj = getattr(mod, name)
        except Exception:
            continue

        if obj is None:
            continue

        obj_cat = None
        if "Config" in name:
            obj_cat = SemanticTier.MODEL
        elif "Model" in name or "Pipeline" in name or "Tokenizer" in name:
            obj_cat = SemanticTier.MODEL
        elif "Scheduler" in name:
            obj_cat = SemanticTier.OPTIMIZER
        else:
            obj_cat = SemanticTier.UTIL

        if obj_cat == category:
            try:
                ref = inspector.inspect(obj, f"{module_name}.{name}")
                if ref:
                    if "Config" in name:
                        _parse_pretrained_config(obj, ref)
                    if name.startswith("AutoModel"):
                        _handle_automodel_factory(obj, name, ref)
                    if hasattr(obj, "generate"):
                        _extract_generation_kwargs(obj, ref)

                    results.append(ref)
            except Exception:
                pass

    return results


def collect_transformers(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect transformers API.

    Args:
        category: description
        include_nonpublic: description

    Returns:
        List of GhostRefs.
    """
    mapping = {
        SemanticTier.MODEL: ["Config", "Model", "Pipeline", "Tokenizer"],
        SemanticTier.UTIL: ["other"],
    }
    return collect_huggingface("transformers", mapping, category, include_nonpublic)


def collect_diffusers(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect diffusers API.

    Args:
        category: description
        include_nonpublic: description

    Returns:
        List of GhostRefs.
    """
    mapping = {
        SemanticTier.MODEL: ["Model", "Pipeline"],
        SemanticTier.OPTIMIZER: ["Scheduler"],
        SemanticTier.UTIL: ["other"],
    }
    return collect_huggingface("diffusers", mapping, category, include_nonpublic)


def collect_tokenizers(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect tokenizers API.

    Args:
        category: description
        include_nonpublic: description

    Returns:
        List of GhostRefs.
    """
    mapping = {
        SemanticTier.MODEL: ["Tokenizer", "Model"],
        SemanticTier.UTIL: ["other"],
    }
    return collect_huggingface("tokenizers", mapping, category, include_nonpublic)
