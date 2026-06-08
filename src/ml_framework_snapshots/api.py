"""ML Framework Snapshots SDK.

Provides a programmatic interface for capturing and saving API signatures
from various machine learning frameworks.
"""

import concurrent.futures
import importlib.metadata
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.frameworks.torch import collect_api as torch_collect
from ml_framework_snapshots.frameworks.jax import collect_api as jax_collect
from ml_framework_snapshots.frameworks.keras import collect_api as keras_collect
from ml_framework_snapshots.frameworks.tensorflow import collect_api as tf_collect
from ml_framework_snapshots.frameworks.mlx import collect_api as mlx_collect
from ml_framework_snapshots.frameworks.flax_nnx import collect_api as flax_collect
from ml_framework_snapshots.frameworks.sklearn import collect_api as sklearn_collect
from ml_framework_snapshots.frameworks.huggingface import (
    collect_transformers,
    collect_diffusers,
    collect_tokenizers,
)
from ml_framework_snapshots.frameworks.triton import collect_api as triton_collect
from ml_framework_snapshots.frameworks.onnxruntime import (
    collect_api as onnxruntime_collect,
)
from ml_framework_snapshots.frameworks.deepspeed import collect_api as deepspeed_collect
from ml_switcheroo_ir.schema.ghost import GhostRef


def get_available_frameworks() -> Dict[str, Any]:
    """Get a dictionary of available framework collectors.

    Returns:
        A dictionary mapping framework identifiers to their collection functions.
    """
    import pkgutil
    import importlib
    import ml_framework_snapshots.frameworks

    collectors = {}

    # Iterate over modules in the frameworks package
    package = ml_framework_snapshots.frameworks
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        try:
            mod = importlib.import_module(
                f"ml_framework_snapshots.frameworks.{module_name}"
            )

            # Find all functions starting with 'collect_'
            for name, obj in vars(mod).items():
                if callable(obj) and name.startswith("collect_"):
                    # Derive a reasonable identifier based on the module or function name
                    if name == "collect_api":
                        identifier = module_name
                    elif name.startswith("collect_"):  # pragma: no branch
                        identifier = f"{module_name}_{name[8:]}"
                    else:  # pragma: no cover
                        continue

                    collectors[identifier] = obj
        except Exception:  # pragma: no branch
            pass

    # Legacy mapping mapping shortnames to correct functions
    legacy = {
        "torch": torch_collect,
        "jax": jax_collect,
        "keras": keras_collect,
        "tensorflow": tf_collect,
        "mlx": mlx_collect,
        "flax_nnx": flax_collect,
        "sklearn": sklearn_collect,
        "transformers": collect_transformers,
        "diffusers": collect_diffusers,
        "tokenizers": collect_tokenizers,
        "triton": triton_collect,
        "onnxruntime": onnxruntime_collect,
        "deepspeed": deepspeed_collect,
    }

    collectors.update(legacy)
    return collectors


FRAMEWORK_COLLECTORS = get_available_frameworks()


def get_pkg_version(package_name: str) -> str:
    """Safely retrieves the installed version of a python package.

    Args:
        package_name: The name of the pip package.

    Returns:
        The version string, or "unknown" if not installed.

    """
    try:
        if package_name == "flax_nnx":
            package_name = "flax"
        elif package_name == "sklearn":
            package_name = "scikit-learn"
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def _consolidate_aliases(refs: List[GhostRef]) -> List[GhostRef]:
    """Consolidates identical GhostRefs into a single reference with aliases.

    Args:
        refs: A list of GhostRef objects.

    Returns:
        A consolidated list of GhostRef objects.

    """
    consolidated = {}
    for ref in refs:
        # Use name, kind, params, and docstring to identify identical references.
        # Convert params to a comparable tuple.
        param_sigs = tuple(
            (
                p.name,
                p.kind,
                p.default,
                p.annotation,
                p.description,
                p.standardized_name,
            )
            for p in ref.params
        )
        key = (
            ref.name,
            ref.kind,
            param_sigs,
            ref.docstring,
            tuple(ref.raises),
            ref.returns_type,
            ref.returns_description,
        )

        if key not in consolidated:
            consolidated[key] = ref
        else:
            existing = consolidated[key]
            # Keep the shorter api_path as primary
            if len(ref.api_path) < len(existing.api_path):
                ref.aliases.append(existing.api_path)
                ref.aliases.extend(existing.aliases)
                ref.is_public = ref.is_public or existing.is_public
                consolidated[key] = ref
            else:
                existing.aliases.append(ref.api_path)
                existing.aliases.extend(ref.aliases)
                existing.is_public = existing.is_public or ref.is_public

    # Deduplicate and sort aliases
    for ref in consolidated.values():
        ref.aliases = sorted(list(set(ref.aliases)))

    return list(consolidated.values())


def extract_snapshot(
    framework_name: str, include_nonpublic: bool = False
) -> Dict[str, Any]:
    """Extract a snapshot dictionary for a specific framework.

    Args:
        framework_name: The framework identifier (e.g., "torch", "jax").
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A dictionary containing the version and category signatures.
        Returns an empty dict if the framework is unknown or not installed.

    """
    if framework_name not in FRAMEWORK_COLLECTORS:
        return {}

    version = get_pkg_version(framework_name)
    if version == "unknown":
        return {}

    collect_func = FRAMEWORK_COLLECTORS[framework_name]
    snapshot_data: Dict[str, Any] = {"version": version, "categories": {}}
    found_any = False

    def _process_category(cat: SemanticTier) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a single category.

        Args:
            cat: category

        Returns:
            tuple
        """
        try:
            refs = collect_func(cat, include_nonpublic)
            if refs:
                refs = _consolidate_aliases(refs)
                refs.sort(key=lambda x: x.name)
                return cat.value, [r.model_dump(exclude_unset=True) for r in refs]
        except Exception:  # pragma: no branch
            pass
        return cat.value, []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_cat = {
            executor.submit(_process_category, cat): cat for cat in SemanticTier
        }
        for future in concurrent.futures.as_completed(future_to_cat):
            cat_val, dumped_refs = future.result()
            if dumped_refs:
                found_any = True
                snapshot_data["categories"][cat_val] = dumped_refs

    if not found_any:
        return {}
    return snapshot_data


def extract_all_snapshots(include_nonpublic: bool = False) -> Dict[str, Dict[str, Any]]:
    """Extract snapshots for all supported and installed frameworks.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A dictionary mapping framework identifiers to their snapshot data.

    """
    results = {}
    for fw in FRAMEWORK_COLLECTORS:
        data = extract_snapshot(fw, include_nonpublic)
        if data:
            results[fw] = data
    return results


def write_snapshot(
    framework_name: str, snapshot_data: Dict[str, Any], output_dir: str
) -> str:
    """Write a snapshot dictionary to a JSON file on disk.

    Args:
        framework_name: The framework identifier.
        snapshot_data: The snapshot dictionary.
        output_dir: The directory path where the file should be written.

    Returns:
        The path to the generated JSON file.

    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    version = snapshot_data.get("version", "unknown")
    safe_ver = version.replace("+", "_").replace(" ", "_")

    file_path = out_path / f"{framework_name}_v{safe_ver}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2, sort_keys=True)

    return str(file_path)
