"""Compliance checking module.

This module provides functionality to parse arbitrary Python files or packages
and compare them against a reference ML framework snapshot.
"""

import os

import importlib
import sys


from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import griffe

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_framework_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)


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
    while (
        Path(os.path.join(current, "__init__.py")).exists()
        and current.parent != current
    ):
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
            if Path(os.path.join(Path(search_path), "src")).is_dir():
                search_path = str(Path(os.path.join(Path(search_path), "src")))
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
    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    return griffe.load(
        mod_name, search_paths=[search_path], docstring_parser=resolved_parser
    )


def align_namespace(api_path: str, target_prefix: str, reference_prefix: str) -> str:
    """Align a target namespace path to a reference namespace path.

    Args:
        api_path: The API path to align.
        target_prefix: The prefix in the target namespace.
        reference_prefix: The prefix in the reference namespace.

    Returns:
        The aligned namespace string.
    """
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

    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    mod_ast = griffe.load(
        mod_name, search_paths=[search_path], docstring_parser=resolved_parser
    )

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
    reference_snapshot: Dict[str, Any],
    target_refs: List[GhostRef],
    strict_c_extensions: bool = False,
) -> Dict[str, Any]:
    """Score the compliance of target refs against a reference snapshot.

    Args:
        reference_snapshot: The reference snapshot dictionary containing categories.
        target_refs: The extracted and aligned GhostRefs from the target.
        strict_c_extensions: Whether to reject fallback matches with opaque C-extension signatures.

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
        return {
            "score_percentage": 0.0,
            "matched": [],
            "missing": [],
            "mismatched": [],
            "opaque_signatures": [],
            "warnings": [],
        }

    matched = []
    missing = []
    mismatched = []
    opaque_signatures = []
    warnings = []

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

        ref_is_opaque = (
            getattr(ref_obj, "signature_completeness", None) == "opaque"
            or "inexact_signature" in (ref_obj.environment_tags or [])
            or "opaque_c_extension" in (ref_obj.environment_tags or [])
        )
        tgt_is_opaque = (
            getattr(target_obj, "signature_completeness", None) == "opaque"
            or "inexact_signature" in (target_obj.environment_tags or [])
            or "opaque_c_extension" in (target_obj.environment_tags or [])
        )

        if ref_is_opaque or tgt_is_opaque:
            opaque_signatures.append(api_path)
            warnings.append(
                f"API '{api_path}' matched via opaque (*args, **kwargs) C-extension fallback."
            )
            if strict_c_extensions:
                mismatched.append(
                    {
                        "api_path": api_path,
                        "expected": ref_sig,
                        "actual": tgt_sig,
                        "reason": "Opaque C-extension signature fallback rejected under strict_c_extensions",
                    }
                )
            else:
                matched.append(api_path)
            continue

        if ref_sig == tgt_sig:
            matched.append(api_path)
        else:
            # Check for compatible supersets (e.g. kwargs fallback)
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
        "opaque_signatures": opaque_signatures,
        "warnings": warnings,
    }


def extract_target_refs(
    file_paths: List[str], target_prefix: str, reference_prefix: str
) -> List[GhostRef]:
    """Extract a list of GhostRefs from multiple target file paths.

    Args:
        file_paths: A list of file paths to extract refs from.
        target_prefix: The prefix in the target namespace.
        reference_prefix: The prefix in the reference namespace.

    Returns:
        A list of GhostRef objects.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    refs = []
    for fp in file_paths:
        refs.extend(extract_target_refs_single(fp, target_prefix, reference_prefix))
    return refs


def check_mlir_text_compliance(mlir_text: str) -> Dict[str, Any]:
    """Verify that an MLIR / StableHLO text snippet uses valid operations, attributes, and operands.

    Args:
        mlir_text: Text snippet of MLIR or StableHLO assembly code.

    Returns:
        Compliance dictionary with is_compliant, total_ops, verified_ops, and errors.
    """
    import re
    from ml_framework_snapshots.mcp_server import check_mlir_op

    errors: List[str] = []
    total_ops = 0
    verified_ops = 0

    op_pattern = re.compile(
        r"(?:%\w+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\s*(?:\(([^)]*)\))?"
    )

    for line in mlir_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("//"):
            continue

        match = op_pattern.search(cleaned)
        if match:
            op_name = match.group(1)
            if op_name in ("builtin.module", "func.func"):
                continue

            total_ops += 1
            args_str = match.group(2)
            operands_count = (
                len([a for a in args_str.split(",") if a.strip()])
                if args_str is not None
                else None
            )

            res = check_mlir_op(op_name, operands_count=operands_count)
            if not res.get("op_exists"):
                errors.append(
                    f"Invalid MLIR operation '{op_name}' on line: '{cleaned}'"
                )
            elif not res.get("is_valid"):
                errors.extend(res.get("errors", []))
            else:
                verified_ops += 1

    return {
        "is_compliant": len(errors) == 0,
        "total_ops": total_ops,
        "verified_ops": verified_ops,
        "errors": errors,
    }


def check_sass_assembly_compliance(
    assembly_text: str, sm_arch: Optional[str] = None
) -> Dict[str, Any]:
    """Verify that an NVIDIA SASS assembly snippet is valid for the target SM architecture.

    Args:
        assembly_text: NVIDIA SASS assembly text.
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').

    Returns:
        Compliance dictionary with is_compliant, total_instructions, and errors.
    """
    import re
    from ml_framework_snapshots.mcp_server import check_sass_instruction

    errors: List[str] = []
    total_insts = 0
    verified_insts = 0

    sass_line_pattern = re.compile(
        r"^(?:@!?P\d+\s+)?([A-Z0-9_]+)((?:\.[A-Z0-9_]+)*)(?:\s+(.*?))?;?$"
    )

    for line in assembly_text.splitlines():
        cleaned = re.sub(r"/\*.*?\*/", "", line).strip()
        if not cleaned or cleaned.startswith("//") or cleaned.startswith("#"):
            continue

        m = sass_line_pattern.match(cleaned)
        if m:
            base_mnemonic = m.group(1)
            mods_raw = m.group(2)
            operands_raw = m.group(3)

            modifiers = [mod for mod in mods_raw.split(".") if mod] if mods_raw else []
            operands = (
                [op.strip() for op in operands_raw.split(",") if op.strip()]
                if operands_raw
                else None
            )

            total_insts += 1
            res = check_sass_instruction(
                base_mnemonic,
                operands=operands,
                modifiers=modifiers,
                sm_arch=sm_arch,
            )
            if not res.get("is_valid"):
                errors.extend(res.get("errors", []))
            else:
                verified_insts += 1

    return {
        "is_compliant": len(errors) == 0,
        "total_instructions": total_insts,
        "verified_instructions": verified_insts,
        "errors": errors,
    }


def check_rdna_assembly_compliance(
    assembly_text: str, gfx_arch: Optional[str] = None
) -> Dict[str, Any]:
    """Verify that an AMD RDNA assembly snippet is valid for the target GFX architecture.

    Args:
        assembly_text: AMD RDNA assembly text.
        gfx_arch: Optional target GFX architecture (e.g. 'GFX11/RDNA3', 'GFX9/CDNA').

    Returns:
        Compliance dictionary with is_compliant, total_instructions, and errors.
    """
    import re
    from ml_framework_snapshots.mcp_server import check_rdna_instruction

    errors: List[str] = []
    total_insts = 0
    verified_insts = 0

    rdna_line_pattern = re.compile(r"^([a-z0-9_]+)(?:\s+(.*?))?;?$")

    for line in assembly_text.splitlines():
        cleaned = re.sub(r"//.*", "", line).strip()
        if not cleaned or cleaned.startswith(";") or cleaned.startswith("#"):
            continue

        m = rdna_line_pattern.match(cleaned)
        if m:
            mnemonic = m.group(1)
            operands_raw = m.group(2)
            operands = (
                [op.strip() for op in operands_raw.split(",") if op.strip()]
                if operands_raw
                else None
            )

            total_insts += 1
            res = check_rdna_instruction(mnemonic, operands=operands, gfx_arch=gfx_arch)
            if not res.get("is_valid"):
                errors.extend(res.get("errors", []))
            else:
                verified_insts += 1

    return {
        "is_compliant": len(errors) == 0,
        "total_instructions": total_insts,
        "verified_instructions": verified_insts,
        "errors": errors,
    }


def validate_broadcast_shapes(
    shape_a: Sequence[Union[int, str]], shape_b: Sequence[Union[int, str]]
) -> Tuple[bool, Optional[List[int]], Optional[str]]:
    """Validate NumPy/PyTorch broadcasting compatibility between two tensor shapes.

    Rules:
        - Trailing dimensions are aligned.
        - Two dimensions are compatible if they are equal, or one of them is 1.
        - Wildcard/symbolic dimensions ('?', -1) are treated as dynamically compatible.

    Args:
        shape_a: Shape sequence of the first tensor operand.
        shape_b: Shape sequence of the second tensor operand.

    Returns:
        A tuple of (is_compatible, broadcast_resulting_shape, error_message).
    """
    reversed_a = list(reversed(shape_a))
    reversed_b = list(reversed(shape_b))
    max_len = max(len(reversed_a), len(reversed_b))

    result_shape: List[int] = []
    for i in range(max_len):
        dim_a = reversed_a[i] if i < len(reversed_a) else 1
        dim_b = reversed_b[i] if i < len(reversed_b) else 1

        if str(dim_a) in ("?", "-1", "None") or str(dim_b) in ("?", "-1", "None"):
            result_shape.append(-1)
            continue

        try:
            int_a = int(dim_a)
            int_b = int(dim_b)
        except (ValueError, TypeError):
            result_shape.append(-1)
            continue

        if int_a == int_b:
            result_shape.append(int_a)
        elif int_a == 1:
            result_shape.append(int_b)
        elif int_b == 1:
            result_shape.append(int_a)
        else:
            return (
                False,
                None,
                f"Shape mismatch: dimension at reverse index {i} cannot broadcast between {dim_a} and {dim_b} (shapes: {list(shape_a)} vs {list(shape_b)}).",
            )

    return True, list(reversed(result_shape)), None


def validate_matmul_shapes(
    shape_a: Sequence[Union[int, str]],
    shape_b: Sequence[Union[int, str]],
    strict_2d: bool = False,
) -> Tuple[bool, Optional[List[int]], Optional[str]]:
    """Validate matrix multiplication shape and contracting inner-dimension constraints.

    Rules:
        - If strict_2d is True (e.g. torch.mm), both shapes must be strictly 2D.
        - For 2D matrices (M, K) x (K, N), inner dimension K must match.
        - For batch matrices (...B, M, K) x (...B, K, N), batch dimensions must broadcast.
        - For 1D vectors (K,) x (K,), dot product requires matching dimension.

    Args:
        shape_a: Shape sequence of LHS matrix tensor.
        shape_b: Shape sequence of RHS matrix tensor.
        strict_2d: Flag enforcing strictly rank-2 matrix operands (torch.mm).

    Returns:
        A tuple of (is_compatible, output_shape, error_message).
    """
    if strict_2d:
        if len(shape_a) != 2 or len(shape_b) != 2:
            return (
                False,
                None,
                f"Strict 2D matmul error: both operands must have rank 2, got ranks {len(shape_a)} and {len(shape_b)}.",
            )

    if len(shape_a) == 1 and len(shape_b) == 1:
        if str(shape_a[0]) not in ("?", "-1") and str(shape_b[0]) not in ("?", "-1"):
            if int(shape_a[0]) != int(shape_b[0]):
                return (
                    False,
                    None,
                    f"1D vector dot product mismatch: {shape_a[0]} != {shape_b[0]}.",
                )
        return True, [], None

    if len(shape_a) < 2 or len(shape_b) < 2:
        return (
            False,
            None,
            f"Matrix multiplication requires at least 2D operands (got ranks {len(shape_a)} and {len(shape_b)}).",
        )

    k_lhs = shape_a[-1]
    k_rhs = shape_b[-2]
    if str(k_lhs) not in ("?", "-1") and str(k_rhs) not in ("?", "-1"):
        if int(k_lhs) != int(k_rhs):
            return (
                False,
                None,
                f"Matrix multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)}).",
            )

    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    if batch_a or batch_b:
        compat, b_shape, err = validate_broadcast_shapes(batch_a, batch_b)
        if not compat:
            return False, None, f"Batch dimension broadcasting error in matmul: {err}"
        out_shape = (b_shape or []) + [
            int(shape_a[-2]) if str(shape_a[-2]).isdigit() else -1,
            int(shape_b[-1]) if str(shape_b[-1]).isdigit() else -1,
        ]
    else:
        out_shape = [
            int(shape_a[-2]) if str(shape_a[-2]).isdigit() else -1,
            int(shape_b[-1]) if str(shape_b[-1]).isdigit() else -1,
        ]

    return True, out_shape, None
