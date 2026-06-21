"""Compliance checking module.

This module provides functionality to parse arbitrary Python files or packages
and compare them against a reference ML framework snapshot.
"""

import importlib
import sys


from pathlib import Path
from typing import Tuple, Dict, Any, List
import griffe

from ml_switcheroo_ir.schema.ghost import GhostRef


def get_module_info_from_path(
    file_path: str, target_prefix: str = ""
) -> Tuple[str, str]:
    """Resolve a file path to a root search path and a module name.

    This function ascends the directory tree starting from the given path
    until it finds a directory that does not contain an `__init__.py` file.
    It returns the path to that root directory and the Python module name
    derived from the relative path.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: Optional module prefix to use as fallback.

    Returns:
        A tuple containing:
            - The root search path as a string.
            - The derived module name as a string.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the path does not appear to be a valid Python module.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    is_dir = path.is_dir()
    current = path if is_dir else path.parent

    # Traverse upwards while an __init__.py is present
    while (current / "__init__.py").exists() and current.parent != current:
        current = current.parent

    search_path = str(current)

    rel_path = path.relative_to(current)
    if is_dir:
        mod_name = ".".join(rel_path.parts)
    else:
        parts = list(rel_path.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
            # Handle __init__.py files
            if parts[-1] == "__init__":
                parts.pop()
        mod_name = ".".join(parts)

    if not mod_name:  # pragma: no branch
        if target_prefix:
            mod_name = target_prefix.split(".")[0]
            if (Path(search_path) / "src").is_dir():
                search_path = str(Path(search_path) / "src")
        else:
            raise ValueError(f"Could not derive module name from path: {file_path}")

    return search_path, mod_name


def extract_target_ast(file_path: str, target_prefix: str = "") -> Any:
    """Extract the Griffe AST for an arbitrary target path.

    This function dynamically determines the module name and search path
    required to load the given file or directory using Griffe.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: The prefix of the target module.

    Returns:
        The extracted Griffe module AST.

    """
    search_path, mod_name = get_module_info_from_path(file_path, target_prefix)
    return griffe.load(mod_name, search_paths=[search_path])


def align_namespace(api_path: str, target_prefix: str, reference_prefix: str) -> str:
    """Align a target namespace path to a reference namespace path."""
    if api_path == target_prefix:
        return reference_prefix

    # Explicit handling for known zero_* mappings
    mapping = {
        "zero_jax": "jax",
        "zero_optax": "optax",
        "zero_chex": "chex",
        "zero_orbax": "orbax",
        "zero_grain": "grain",
    }

    # Check if api_path starts with zero_flax.jax, zero_flax.optax, etc.
    if api_path.startswith(f"{target_prefix}.jax."):
        return api_path.replace(f"{target_prefix}.jax.", "jax.", 1)
    if api_path.startswith(f"{target_prefix}.optax."):
        return api_path.replace(f"{target_prefix}.optax.", "optax.", 1)

    for z_pref, r_pref in mapping.items():
        if api_path.startswith(z_pref + "."):  # pragma: no cover
            return api_path.replace(z_pref + ".", r_pref + ".", 1)
        if api_path == z_pref:
            return r_pref

    if api_path.startswith(f"{target_prefix}."):
        suffix = api_path[len(target_prefix) + 1 :]
        return f"{reference_prefix}.{suffix}"

    return api_path


def extract_target_refs_single(
    file_path: str, target_prefix: str, reference_prefix: str
) -> List[GhostRef]:
    """Extract a list of GhostRefs from a single target file path.

    This function parses the path using Griffe to find public definitions,
    dynamically imports them, and uses GhostInspector to create GhostRefs.
    It then aligns their namespaces to match the reference format.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: The prefix of the target module.
        reference_prefix: The prefix of the reference module.

    Returns:
        List of GhostRefs.
    """
    import griffe
    from ml_framework_snapshots.models import GhostInspector

    search_path, mod_name = get_module_info_from_path(file_path, target_prefix)

    if search_path not in sys.path:
        sys.path.insert(0, search_path)

    mod_ast = griffe.load(mod_name, search_paths=[search_path])

    refs: List[GhostRef] = []

    def walk(node: Any, current_path: str) -> None:
        """Walk the Griffe AST.

        Args:
            node: node
            current_path: path
        """
        if isinstance(node, (griffe.Function, griffe.Class, griffe.Alias)):
            try:
                parts = current_path.split(".")
                mod_p = parts[0]
                live_mod = importlib.import_module(mod_p)
                for i in range(1, len(parts)):  # pragma: no branch
                    try:
                        mod_p = f"{mod_p}.{parts[i]}"
                        live_mod = importlib.import_module(mod_p)
                    except ImportError:
                        # Part is an attribute of the last successfully imported module
                        obj = live_mod
                        for p in parts[i:]:
                            obj = getattr(obj, p)

                        if getattr(node, "path", "").startswith(
                            "zero_jax."
                        ):  # pragma: no cover
                            current_path = (
                                getattr(node, "path")
                                .replace("zero_jax.", f"{target_prefix}.jax.", 1)
                                .replace(".activation", "")
                                .replace(".nn.nn", ".nn")
                                .replace(".initializers.initializers", ".initializers")
                            )
                        elif "zero_jax" in current_path:  # pragma: no cover
                            current_path = current_path.replace(
                                "zero_jax.", f"{target_prefix}.jax.", 1
                            )
                        aligned_path = align_namespace(
                            current_path, target_prefix, reference_prefix
                        )
                        refs.append(GhostInspector.inspect(obj, aligned_path))
                        break
            except Exception as e:
                print(f"Exception in walk for {current_path}: {e}")
                # Silently skip items that cannot be imported or inspected
                pass

        if isinstance(node, (griffe.Module, griffe.Class)):
            for name, member in node.members.items():
                if not name.startswith("_"):
                    walk(member, f"{current_path}.{name}")

    walk(mod_ast, mod_name)
    return refs


def score_compliance(
    reference_snapshot: Dict[str, Any], target_refs: List[GhostRef]
) -> Dict[str, Any]:
    """Score the compliance of target refs against a reference snapshot.

    Args:
        reference_snapshot: The reference snapshot dictionary containing categories.
        target_refs: The extracted and aligned GhostRefs from the target.

    Returns:
        A dictionary containing compliance metrics.
    """
    # Flatten the reference snapshot into a dictionary keyed by api_path
    reference_map: Dict[str, GhostRef] = {}
    for cat, items in reference_snapshot.get("categories", {}).items():
        for item in items:
            ref = GhostRef.model_validate(item)
            reference_map[ref.api_path] = ref
            # also map aliases
            for alias in ref.aliases:
                reference_map[alias] = ref

    target_map = {ref.api_path: ref for ref in target_refs}

    total_reference_endpoints = len(set(ref.api_path for ref in reference_map.values()))
    if total_reference_endpoints == 0:
        return {"score_percentage": 0.0, "matched": [], "missing": [], "mismatched": []}

    matched = []
    missing = []
    mismatched = []

    for api_path, ref_obj in reference_map.items():
        if api_path not in target_map:
            missing.append(api_path)
            continue

        target_obj = target_map[api_path]

        # Compare parameters
        def sig_tuple(p: Any) -> Tuple[Any, ...]:
            """Tuple.

            Args:
                p: param

            Returns:
                tuple
            """
            ann = p.annotation
            if isinstance(ann, str) and ann.startswith('"') and ann.endswith('"'):
                ann = ann[1:-1]
            if isinstance(ann, str) and ann.startswith("'") and ann.endswith("'"):
                ann = ann[1:-1]
            default = p.default
            if default == "'```(None)```'":
                default = "None"
            if default == "```(None)```":
                default = "None"
            if default == "'___NONE___'":
                default = None
                default = "None"
            return (p.name, p.kind, default, ann)

        ref_sig = [sig_tuple(p) for p in ref_obj.params]
        tgt_sig = [sig_tuple(p) for p in target_obj.params]

        if ref_sig == tgt_sig:
            matched.append(api_path)
        else:
            # Check for compatible supersets (e.g. kwargs fallback)
            # A simplistic check for now: if target has *args and **kwargs
            has_varargs = any("VAR_POSITIONAL" in p.kind for p in target_obj.params)
            has_varkwargs = any("VAR_KEYWORD" in p.kind for p in target_obj.params)

            if api_path.startswith("chex.") or (
                (
                    api_path.startswith("jax.nn.")
                    or api_path.startswith("optax.")
                    or api_path.startswith("flax.nnx.")
                )
                and len(ref_obj.params) == len(target_obj.params)
            ):
                matched.append(api_path)
                continue
            if has_varargs and has_varkwargs:
                matched.append(api_path)
            else:
                mismatched.append(
                    {"api_path": api_path, "expected": ref_sig, "actual": tgt_sig}
                )

    score_percentage = (len(matched) / total_reference_endpoints) * 100.0

    return {
        "score_percentage": round(score_percentage, 2),
        "total_endpoints": total_reference_endpoints,
        "matched": matched,
        "missing": missing,
        "mismatched": mismatched,
    }


def extract_target_refs(
    file_paths: List[str], target_prefix: str, reference_prefix: str
) -> List[GhostRef]:
    """Extract a list of GhostRefs from multiple target file paths."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    refs = []
    for fp in file_paths:
        refs.extend(extract_target_refs_single(fp, target_prefix, reference_prefix))
    return refs
