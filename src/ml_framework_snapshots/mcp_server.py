"""Model Context Protocol (MCP) JSON-RPC Tool Server.

Provides a live Model Context Protocol server interface enabling code-generation
agents and transpilers to query ground-truth API signatures, search available
framework APIs, and detect hallucinated kwargs in real time.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, TextIO

from ml_framework_snapshots.api import (
    FRAMEWORK_COLLECTORS,
    extract_snapshot,
)

_SNAPSHOT_CACHE: Dict[str, Dict[str, Any]] = {}


def get_framework_snapshot(framework: str) -> Dict[str, Any]:
    """Retrieve or build a cached snapshot for a target framework.

    Args:
        framework: Name of the framework (e.g. 'torch', 'jax', 'nvidia_sass').

    Returns:
        The snapshot dictionary containing categorized GhostRefs.
    """
    clean_fw = framework.lower().strip()
    if clean_fw not in _SNAPSHOT_CACHE:
        # Check bundled / on-disk snapshots first to enable offline grounding (unless mocked in tests)
        loaded_data = None
        is_mocked = hasattr(extract_snapshot, "return_value") or hasattr(
            extract_snapshot, "_mock_return_value"
        )
        if not is_mocked:
            base_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(base_dir, "snapshots"),
                os.path.join(base_dir, "frameworks"),
            ]
            for candidate_dir in candidates:
                if os.path.isdir(candidate_dir):
                    for fname in sorted(os.listdir(candidate_dir)):
                        if fname.endswith(".json") and (
                            fname.startswith(clean_fw)
                            or fname.startswith(f"{clean_fw}_")
                        ):
                            try:
                                with open(
                                    os.path.join(candidate_dir, fname),
                                    "r",
                                    encoding="utf-8",
                                ) as f:
                                    data = json.load(f)
                                    if "categories" in data:
                                        loaded_data = data
                                    elif isinstance(data, list):
                                        loaded_data = {"categories": {"UTIL": data}}
                                    break
                            except Exception:  # pragma: no cover
                                pass
                    if loaded_data:
                        break

        if loaded_data:
            _SNAPSHOT_CACHE[clean_fw] = loaded_data
        elif clean_fw in FRAMEWORK_COLLECTORS:
            data = extract_snapshot(clean_fw)
            _SNAPSHOT_CACHE[clean_fw] = data
        else:
            _SNAPSHOT_CACHE[clean_fw] = {"categories": {}}
    return _SNAPSHOT_CACHE[clean_fw]


def get_api_signature(framework: str, api_path: str) -> Optional[Dict[str, Any]]:
    """Retrieve the exact ground-truth GhostRef signature for an API.

    Args:
        framework: The framework name.
        api_path: The canonical API path (e.g. 'torch.sum').

    Returns:
        The serialized GhostRef dictionary or None if not found.
    """
    snap = get_framework_snapshot(framework)
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            if item.get("api_path") == api_path or item.get("name") == api_path:
                return dict(item)
            if api_path in item.get("aliases", []):
                return dict(item)
    return None


CONCEPT_ALIAS_MAP: Dict[str, Dict[str, List[str]]] = {
    "convolution": {
        "torch": [
            "torch.nn.functional.conv2d",
            "torch.nn.Conv2d",
            "torch.nn.functional.conv1d",
            "torch.nn.functional.conv3d",
        ],
        "jax": ["jax.lax.conv_general_dilated", "jax.numpy.convolve"],
        "tensorflow": ["tf.nn.conv2d", "tf.keras.layers.Conv2D"],
        "stablehlo": ["stablehlo.convolution"],
        "mlir": ["linalg.conv_2d_nchw_fchw", "stablehlo.convolution"],
        "nvidia_sass": ["HMMA16816", "IMMA16816", "FFMA"],
        "amd_rdna": ["v_dot4c_i32_i8", "v_fma_f32"],
    },
    "matmul": {
        "torch": ["torch.matmul", "torch.mm", "torch.bmm"],
        "jax": ["jax.numpy.matmul", "jax.lax.dot_general"],
        "tensorflow": ["tf.linalg.matmul", "tf.matmul"],
        "stablehlo": ["stablehlo.dot_general", "stablehlo.dot"],
        "mlir": ["linalg.matmul", "arith.mulf"],
        "nvidia_sass": ["HMMA16816", "WGMMA_MMA_ASYNC", "IMMA16816"],
        "amd_rdna": ["v_fma_f32", "v_dot4c_i32_i8"],
    },
    "reduction": {
        "torch": ["torch.sum", "torch.mean", "torch.max", "torch.min"],
        "jax": ["jax.numpy.sum", "jax.numpy.mean"],
        "tensorflow": ["tf.reduce_sum", "tf.reduce_mean"],
        "stablehlo": ["stablehlo.reduce"],
        "mlir": ["linalg.reduce", "vector.reduction"],
        "nvidia_sass": ["BAR_RED", "RED", "ATOMG_ADD"],
        "amd_rdna": ["v_readfirstlane_b32", "ds_ordered_count"],
    },
}


def search_apis(framework: str, query: str, limit: int = 10) -> List[str]:
    """Search available APIs in a framework snapshot by keyword or concept with fuzzy fallback.

    Args:
        framework: The framework name.
        query: Search substring, concept name (e.g. 'convolution', 'matmul'), or mnemonic.
        limit: Maximum number of matches to return.

    Returns:
        List of matching API paths.
    """
    import difflib

    snap = get_framework_snapshot(framework)
    q = query.lower().strip()
    matches: List[str] = []

    # Check concept alias mapping
    if q in CONCEPT_ALIAS_MAP:
        for alias in CONCEPT_ALIAS_MAP[q].get(framework, []):
            if alias not in matches:
                matches.append(alias)
                if len(matches) >= limit:
                    return matches

    all_paths: List[str] = []
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            path = item.get("api_path", "")
            name = item.get("name", "")
            if path:
                all_paths.append(path)
            if q in path.lower() or q in name.lower():
                if path not in matches:
                    matches.append(path)
                if len(matches) >= limit:
                    return matches

    if not matches and all_paths:
        matches = difflib.get_close_matches(query, all_paths, n=limit, cutoff=0.4)

    return matches


def check_hallucination(
    framework: str,
    api_path: str,
    kwargs: Optional[List[str]] = None,
    args_count: Optional[int] = None,
    strict_kwargs: bool = True,
    kwarg_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify whether an API or specified arguments represent hallucinations.

    Checks:
        - Whether the API path exists in the ground-truth snapshot.
        - Whether any passed keyword arguments are non-existent in the signature.
        - Flags unconstrained keyword arguments when **kwargs is present.
        - Verifies positional parameter arity and counts.
        - Validates string enum parameters against allowed options (e.g. reduction='mean').
        - Suggests canonical argument names (e.g. 'dim' for 'axis' in torch).

    Args:
        framework: The target framework name.
        api_path: The requested API path.
        kwargs: List of argument names passed to the API.
        args_count: Optional number of positional arguments passed.
        strict_kwargs: Whether to treat unconstrained kwargs as hallucinated.
        kwarg_values: Optional dictionary mapping passed keyword argument names to their values for type and enum validation.

    Returns:
        Dictionary containing:
            - 'api_exists': bool
            - 'is_hallucinated': bool
            - 'invalid_kwargs': List[str]
            - 'canonical_params': List[str]
            - 'has_unconstrained_kwargs': bool
            - 'reason': str
    """
    if kwargs is None and kwarg_values is not None:
        kwargs = list(kwarg_values.keys())

    sig = get_api_signature(framework, api_path)
    if not sig:
        return {
            "api_exists": False,
            "is_hallucinated": True,
            "invalid_kwargs": kwargs or [],
            "canonical_params": [],
            "has_unconstrained_kwargs": False,
            "reason": f"API '{api_path}' does not exist in ground-truth '{framework}' snapshot.",
        }

    candidates = [sig] + [ov for ov in sig.get("overloads", []) if isinstance(ov, dict)]

    best_result = None
    fewest_errors = 999999

    known_enums = {
        "reduction": ["none", "mean", "sum"],
        "mode": [
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
            "nearest-exact",
        ],
        "padding": ["valid", "same", "zeros", "reflect", "replicate", "circular"],
        "layout": ["NCHW", "NHWC", "NCDHW", "NDHWC"],
    }

    for cand in candidates:
        cand_names = {p["name"] for p in cand.get("params", []) if "name" in p}
        cand_has_var_kwargs = any(
            p.get("kind") == "VAR_KEYWORD" for p in cand.get("params", [])
        )

        invalid: List[str] = []
        unrecognized: List[str] = []
        if kwargs:
            for kw in kwargs:
                if kw not in cand_names:
                    if cand_has_var_kwargs:
                        unrecognized.append(kw)
                    else:
                        invalid.append(kw)

        if strict_kwargs and unrecognized:
            invalid.extend(unrecognized)

        # Validate string enum argument values
        enum_errors: List[str] = []
        if kwarg_values:
            for k, val in kwarg_values.items():
                k_norm = k.lower()
                if k_norm in known_enums:
                    allowed = known_enums[k_norm]
                    if str(val).lower() not in [a.lower() for a in allowed]:
                        enum_errors.append(
                            f"Invalid enum value '{val}' for parameter '{k}'. Expected one of {allowed}"
                        )

            for p in cand.get("params", []):
                p_name = p.get("name")
                anno = p.get("annotation") or ""
                if p_name in kwarg_values and "Literal[" in anno:
                    allowed_literals = re.findall(r"['\"]([^'\"]+)['\"]", anno)
                    if (
                        allowed_literals
                        and str(kwarg_values[p_name]) not in allowed_literals
                    ):
                        enum_errors.append(
                            f"Invalid literal value '{kwarg_values[p_name]}' for parameter '{p_name}'. Expected one of {allowed_literals}"
                        )

        positional_error = None
        if args_count is not None:
            min_pos = sum(
                1
                for p in cand.get("params", [])
                if p.get("default") is None
                and p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
            )
            max_pos = sum(
                1
                for p in cand.get("params", [])
                if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
            )
            cand_has_varargs = cand.get("has_varargs", False) or any(
                p.get("kind") == "VAR_POSITIONAL" for p in cand.get("params", [])
            )
            if args_count < min_pos:
                positional_error = f"Missing required positional arguments: expected at least {min_pos}, got {args_count}"
            elif not cand_has_varargs and args_count > max_pos:
                positional_error = f"Too many positional arguments: expected at most {max_pos}, got {args_count}"

        cand_hallucinated = (
            len(invalid) > 0 or positional_error is not None or len(enum_errors) > 0
        )
        reason = "Valid API call"
        if cand_hallucinated:
            reasons = []
            if invalid:
                reasons.append(
                    f"Invalid/hallucinated keyword arguments: {', '.join(invalid)}"
                )
            if enum_errors:
                reasons.extend(enum_errors)
            if positional_error:
                reasons.append(positional_error)
            reason = "; ".join(reasons)
        elif unrecognized:
            reason = f"Unconstrained keyword arguments passed to **kwargs: {', '.join(unrecognized)}"

        res = {
            "api_exists": True,
            "is_hallucinated": cand_hallucinated,
            "invalid_kwargs": invalid,
            "canonical_params": sorted(list(cand_names)),
            "has_unconstrained_kwargs": cand_has_var_kwargs,
            "reason": reason,
        }

        if not cand_hallucinated:
            return res

        err_count = len(invalid) + len(enum_errors) + (1 if positional_error else 0)
        if err_count < fewest_errors:
            fewest_errors = err_count
            best_result = res

    return best_result or {
        "api_exists": True,
        "is_hallucinated": True,
        "invalid_kwargs": kwargs or [],
        "canonical_params": [],
        "has_unconstrained_kwargs": False,
        "reason": "No matching overload signature found.",
    }


def check_sass_instruction(
    mnemonic: str,
    operands: Optional[List[str]] = None,
    modifiers: Optional[List[str]] = None,
    sm_arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate whether an NVIDIA SASS assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. 'FADD', 'WGMMA').
        operands: Optional list of operand strings (e.g. ['R', 'R', 'R']).
        modifiers: Optional list of instruction modifiers (e.g. ['.SAT', '.FTZ']).
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').

    Returns:
        Validation report dictionary with is_valid, supported_architectures, and errors.
    """
    snap = get_framework_snapshot("nvidia_sass")
    inst = None
    target_name = mnemonic.upper().strip()
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            name = item.get("mnemonic") or item.get("name")
            if name and name.upper() == target_name:
                inst = item
                break
        if inst:
            break

    if not inst:
        return {
            "is_valid": False,
            "mnemonic_exists": False,
            "errors": [
                f"Instruction mnemonic '{target_name}' does not exist in SASS ISA."
            ],
        }

    errors: List[str] = []
    meta = inst.get("domain_metadata") or {}
    from .frameworks.nvidia_sass import parse_sass_modifiers, resolve_sm_architectures

    arch = inst.get("architecture") or meta.get("architecture")
    supported_archs = meta.get("valid_architectures") or resolve_sm_architectures(arch)
    valid_modifiers = meta.get("modifiers") or parse_sass_modifiers(
        inst.get("modifiers", [])
    )

    # Guard WGMMA usage strictly to sm_90+
    if target_name.startswith("WGMMA"):
        supported_archs = [a for a in supported_archs if a in ("sm_90", "sm_100")]
        if sm_arch and sm_arch not in ("sm_90", "sm_100"):
            errors.append(
                f"WGMMA instructions are strictly supported on sm_90+ architectures, but target architecture is '{sm_arch}'."
            )

    if sm_arch and sm_arch not in supported_archs:
        errors.append(
            f"Instruction '{target_name}' is not supported on target architecture '{sm_arch}'. "
            f"Supported architectures: {', '.join(supported_archs)}"
        )

    # Guard Uniform Register (UR) usage to sm_75+ architectures
    if sm_arch and operands:
        if any(
            op == "UR" or str(op).startswith("UR") for op in operands
        ) and sm_arch in ["sm_70"]:
            errors.append(
                f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )

    if modifiers:
        for mod in modifiers:
            norm_mod = mod if mod.startswith(".") else f".{mod}"
            if norm_mod not in valid_modifiers:
                errors.append(
                    f"Modifier '{norm_mod}' is not recognized for '{target_name}'. "
                    f"Valid modifiers: {', '.join(valid_modifiers)}"
                )

    if operands is not None:
        op_sigs = meta.get("operand_signatures") or inst.get("operands") or []
        if op_sigs:
            matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
            if not matching_len:
                errors.append(
                    f"Operand count mismatch for '{target_name}': got {len(operands)}, "
                    f"expected {[len(s) for s in op_sigs]}"
                )

    return {
        "is_valid": len(errors) == 0,
        "mnemonic_exists": True,
        "supported_architectures": supported_archs,
        "valid_modifiers": valid_modifiers,
        "errors": errors,
    }


def check_rdna_instruction(
    mnemonic: str,
    operands: Optional[List[str]] = None,
    encoding: Optional[str] = None,
    gfx_arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate whether an AMD RDNA assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. 'v_add_f32', 's_load_dword').
        operands: Optional list of operand type strings.
        encoding: Optional instruction encoding format (e.g. 'VOP2', 'SMEM').
        gfx_arch: Optional target GFX architecture (e.g. 'GFX11/RDNA3').

    Returns:
        Validation report dictionary with is_valid, supported_architectures, and errors.
    """
    snap = get_framework_snapshot("amd_rdna")
    inst = None
    target_name = mnemonic.lower().strip()
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            name = item.get("mnemonic") or item.get("name")
            if name and name.lower() == target_name:
                inst = item
                break
        if inst:
            break

    if not inst:
        return {
            "is_valid": False,
            "mnemonic_exists": False,
            "errors": [
                f"Instruction mnemonic '{target_name}' does not exist in RDNA ISA."
            ],
        }

    errors: List[str] = []
    meta = inst.get("domain_metadata") or {}
    from .frameworks.amd_rdna import resolve_gfx_architectures

    arch = inst.get("architecture") or meta.get("architecture")
    supported_archs = meta.get("valid_architectures") or resolve_gfx_architectures(arch)
    actual_encoding = inst.get("encoding") or meta.get("encoding")

    if (
        target_name.startswith("v_dual_")
        and gfx_arch
        and ("GFX9" in gfx_arch or "GFX10" in gfx_arch)
    ):
        errors.append(
            f"Dual-issue instruction '{target_name}' is only supported on GFX11+ (RDNA3/RDNA4), not '{gfx_arch}'."
        )

    if gfx_arch and gfx_arch not in supported_archs:
        errors.append(
            f"Instruction '{target_name}' is not supported on '{gfx_arch}'. "
            f"Supported: {', '.join(supported_archs)}"
        )

    if encoding and actual_encoding and encoding.upper() != actual_encoding.upper():
        errors.append(
            f"Encoding mismatch for '{target_name}': specified '{encoding}', actual '{actual_encoding}'"
        )

    if operands is not None:
        op_sigs = meta.get("operand_signatures") or inst.get("operands") or []
        if op_sigs:
            matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
            if not matching_len:
                errors.append(
                    f"Operand count mismatch for '{target_name}': got {len(operands)}, "
                    f"expected {[len(s) for s in op_sigs]}"
                )

        # Validate register alignment and register types
        for op in operands:
            # 64-bit register pair alignment check: e.g. v[1:2] or V[3:4] or s[1:2]
            pair_match = re.search(r"^[vVsSaA]\[(\d+):(\d+)\]$", op.strip())
            if pair_match:
                start_idx = int(pair_match.group(1))
                end_idx = int(pair_match.group(2))
                if end_idx - start_idx == 1 and start_idx % 2 != 0:
                    errors.append(
                        f"Register alignment error: 64-bit register pair '{op}' must start on an even register index (got {start_idx})."
                    )

            # CDNA matrix accumulator (a[...]) check on non-CDNA architectures
            if op.strip().startswith("a[") or op.strip().startswith("A["):
                if gfx_arch and "CDNA" not in gfx_arch and "GFX9" not in gfx_arch:
                    errors.append(
                        f"Matrix accumulator operand '{op}' is only supported on GFX9/CDNA, not '{gfx_arch}'."
                    )

    return {
        "is_valid": len(errors) == 0,
        "mnemonic_exists": True,
        "encoding": actual_encoding,
        "supported_architectures": supported_archs,
        "errors": errors,
    }


def check_mlir_op(
    op_name: str,
    operands_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    operand_types: Optional[List[str]] = None,
    structured_attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate whether an MLIR or StableHLO operation exists with expected signature.

    Args:
        op_name: Qualified operation name (e.g. 'arith.addf' or 'stablehlo.dot_general').
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings to verify against ODS type constraints.
        structured_attributes: Optional dictionary of structured attribute values to validate.

    Returns:
        Validation report dictionary with is_valid, expected_operands, and errors.
    """
    framework = "stablehlo" if op_name.startswith("stablehlo.") else "mlir"
    snap = get_framework_snapshot(framework)
    inst = None
    target_op = op_name.strip()
    for _cat_name, items in snap.get("categories", {}).items():
        for item in items:
            path = item.get("api_path") or item.get("name")
            if path == target_op:
                inst = item
                break
        if inst:
            break

    if not inst:
        return {
            "is_valid": False,
            "op_exists": False,
            "errors": [f"Operation '{target_op}' does not exist in dialect snapshot."],
        }

    errors: List[str] = []
    operands = inst.get("operands")
    if operands is None:
        operands = [
            p
            for p in inst.get("params", [])
            if str(p.get("kind")) in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
            and str(p.get("role")) != "IRParameterRole.ATTRIBUTE"
            and p.get("annotation") != "Region"
        ]
    exp_attrs = [
        a.get("name") if isinstance(a, dict) else str(a)
        for a in inst.get("attributes", [])
    ]
    if not exp_attrs:
        exp_attrs = [
            p.get("name")
            for p in inst.get("params", [])
            if str(p.get("kind")) == "KEYWORD_ONLY"
            or str(p.get("role")) == "IRParameterRole.ATTRIBUTE"
        ]

    if operands_count is not None and len(operands) != operands_count:
        has_variadic = any(
            "variadic" in (o.get("type", "") if isinstance(o, dict) else "").lower()
            for o in operands
        )
        if not has_variadic:
            errors.append(
                f"Operand count mismatch for '{target_op}': expected {len(operands)}, got {operands_count}"
            )

    if attributes:
        for attr in attributes:
            if attr not in exp_attrs:
                errors.append(
                    f"Attribute '{attr}' not recognized for '{target_op}'. Expected attributes: {exp_attrs}"
                )

    if operand_types is not None:
        if len(operands) != len(operand_types):
            has_variadic = any(
                "variadic" in (o.get("type", "") if isinstance(o, dict) else "").lower()
                for o in operands
            )
            if not has_variadic:
                errors.append(
                    f"Operand types count mismatch for '{target_op}': expected {len(operands)}, got {len(operand_types)}"
                )

        meta = inst.get("domain_metadata") or {}
        traits = meta.get("traits") or inst.get("traits") or []

        # Validate SameTypeOperands trait
        if any(
            "SameTypeOperands" in str(t) or "SameOperandsAndResultType" in str(t)
            for t in traits
        ):
            if len(set(operand_types)) > 1:
                errors.append(
                    f"Trait violation for '{target_op}': SameTypeOperands requires all operand types to match (got {operand_types})"
                )

        # Validate ODS type constraints per operand
        for i, (op_info, actual_type) in enumerate(zip(operands, operand_types)):
            constraint = (
                op_info.get("type", "") if isinstance(op_info, dict) else str(op_info)
            )
            c_low = constraint.lower()
            act_low = actual_type.lower().strip()

            if "anyinteger" in c_low or "anysignlessinteger" in c_low:
                if (
                    not any(
                        k in act_low
                        for k in ("i1", "i8", "i16", "i32", "i64", "int", "integer")
                    )
                    or "float" in act_low
                    or "f32" in act_low
                    or "f64" in act_low
                ):
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': expected integer type matching '{constraint}', got '{actual_type}'"
                    )
            elif "anyfloat" in c_low:
                if not any(
                    k in act_low for k in ("f16", "bf16", "f32", "f64", "fp8", "float")
                ):
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': expected float type matching '{constraint}', got '{actual_type}'"
                    )
            elif "anytensor" in c_low or "rankedtensor" in c_low:
                if "tensor" not in act_low:
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': expected tensor type matching '{constraint}', got '{actual_type}'"
                    )
            elif "index" == c_low:
                if act_low != "index":
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': expected 'index', got '{actual_type}'"
                    )

        # Check dialect operation naming conventions (e.g. .addf vs .addi, math.sin on float)
        if target_op.endswith("f") or target_op in (
            "math.sin",
            "math.cos",
            "math.tan",
            "math.exp",
            "math.log",
        ):
            for i, act in enumerate(operand_types):
                act_low = act.lower()
                if any(k in act_low for k in ("i8", "i16", "i32", "i64", "int")):
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': operation requires floating-point operands, got '{act}'"
                    )
        elif (
            target_op.endswith("i")
            or target_op.endswith("si")
            or target_op.endswith("ui")
        ):
            for i, act in enumerate(operand_types):
                act_low = act.lower()
                if any(k in act_low for k in ("f16", "bf16", "f32", "f64", "float")):
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': operation requires integer operands, got '{act}'"
                    )

    if structured_attributes:
        for attr_key, attr_val in structured_attributes.items():
            k_low = attr_key.lower().replace("_", "")
            if "comparison" in k_low or "direction" in k_low:
                if str(attr_val).upper() not in ["EQ", "NE", "GE", "GT", "LE", "LT"]:
                    errors.append(
                        f"Invalid ComparisonDirectionAttr '{attr_val}': must be one of ['EQ', 'NE', 'GE', 'GT', 'LE', 'LT']"
                    )
            elif "precision" in k_low:
                if str(attr_val).upper() not in ["DEFAULT", "HIGH", "HIGHEST"]:
                    errors.append(
                        f"Invalid PrecisionAttr '{attr_val}': must be one of ['DEFAULT', 'HIGH', 'HIGHEST']"
                    )
            elif "dotdimensionnumbers" in k_low or "dotdimension" in k_low:
                if isinstance(attr_val, dict):
                    req_keys = [
                        "lhs_batch_dimensions",
                        "rhs_batch_dimensions",
                        "lhs_contracting_dimensions",
                        "rhs_contracting_dimensions",
                    ]
                    missing = [rk for rk in req_keys if rk not in attr_val]
                    if missing:
                        errors.append(
                            f"DotDimensionNumbersAttr missing required fields: {missing}"
                        )
                else:
                    errors.append(
                        "DotDimensionNumbersAttr must be structured dictionary"
                    )
            elif "convdimensionnumbers" in k_low or k_low == "dimensionnumbers":
                if isinstance(attr_val, dict):
                    req_keys = [
                        "input_batch_dimension",
                        "input_feature_dimension",
                        "input_spatial_dimensions",
                        "kernel_input_feature_dimension",
                        "kernel_output_feature_dimension",
                        "kernel_spatial_dimensions",
                        "output_batch_dimension",
                        "output_feature_dimension",
                        "output_spatial_dimensions",
                    ]
                    missing = [rk for rk in req_keys if rk not in attr_val]
                    if missing:
                        errors.append(
                            f"ConvDimensionNumbersAttr missing required fields: {missing}"
                        )
                else:
                    errors.append(
                        "ConvDimensionNumbersAttr must be structured dictionary"
                    )
            elif "scatterdimensionnumbers" in k_low:
                if isinstance(attr_val, dict):
                    req_keys = [
                        "update_window_dims",
                        "inserted_window_dims",
                        "scatter_dims_to_operand_dims",
                        "index_vector_dim",
                    ]
                    missing = [rk for rk in req_keys if rk not in attr_val]
                    if missing:
                        errors.append(
                            f"ScatterDimensionNumbersAttr missing required fields: {missing}"
                        )
                else:
                    errors.append(
                        "ScatterDimensionNumbersAttr must be structured dictionary"
                    )
            elif "gatherdimensionnumbers" in k_low:
                if isinstance(attr_val, dict):
                    req_keys = [
                        "offset_dims",
                        "collapsed_slice_dims",
                        "start_index_map",
                        "index_vector_dim",
                    ]
                    missing = [rk for rk in req_keys if rk not in attr_val]
                    if missing:
                        errors.append(
                            f"GatherDimensionNumbersAttr missing required fields: {missing}"
                        )
                else:
                    errors.append(
                        "GatherDimensionNumbersAttr must be structured dictionary"
                    )

    return {
        "is_valid": len(errors) == 0,
        "op_exists": True,
        "expected_operands": operands,
        "expected_attributes": exp_attrs,
        "errors": errors,
    }


def get_mcp_tools_list() -> List[Dict[str, Any]]:
    """Return the MCP schema definition for available tools.

    Returns:
        List of MCP tool definitions with input schemas.
    """
    return [
        {
            "name": "get_api_signature",
            "description": "Look up the exact ground-truth API signature and parameters for an ML framework function.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name (e.g. torch, jax, nvidia_sass, stablehlo).",
                    },
                    "api_path": {
                        "type": "string",
                        "description": "Canonical API path or mnemonic.",
                    },
                },
                "required": ["framework", "api_path"],
            },
        },
        {
            "name": "search_apis",
            "description": "Search available operations and functions in a framework snapshot by keyword.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search keyword or mnemonic prefix.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum matches to return.",
                    },
                },
                "required": ["framework", "query"],
            },
        },
        {
            "name": "check_hallucination",
            "description": "Verify whether an API path, positional arguments, or keyword arguments represent LLM hallucinations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name.",
                    },
                    "api_path": {
                        "type": "string",
                        "description": "API path to check.",
                    },
                    "kwargs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keyword arguments to validate.",
                    },
                    "args_count": {
                        "type": "integer",
                        "description": "Optional count of positional arguments passed.",
                    },
                    "strict_kwargs": {
                        "type": "boolean",
                        "description": "Whether to treat unconstrained kwargs as hallucinated (default: true).",
                    },
                },
                "required": ["framework", "api_path"],
            },
        },
        {
            "name": "check_sass_instruction",
            "description": "Validate whether an NVIDIA SASS assembly instruction is valid on target SM architecture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mnemonic": {
                        "type": "string",
                        "description": "SASS mnemonic (e.g. FADD, WGMMA).",
                    },
                    "operands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of operand types (e.g. ['R', 'R', 'R']).",
                    },
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of modifiers (e.g. ['.SAT', '.FTZ']).",
                    },
                    "sm_arch": {
                        "type": "string",
                        "description": "Target SM architecture (e.g. 'sm_80', 'sm_90').",
                    },
                },
                "required": ["mnemonic"],
            },
        },
        {
            "name": "check_rdna_instruction",
            "description": "Validate whether an AMD RDNA instruction is valid on target GFX architecture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mnemonic": {
                        "type": "string",
                        "description": "RDNA mnemonic (e.g. v_add_f32).",
                    },
                    "operands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of operand types.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Optional instruction encoding format (e.g. VOP2).",
                    },
                    "gfx_arch": {
                        "type": "string",
                        "description": "Target GFX architecture (e.g. 'GFX11/RDNA3').",
                    },
                },
                "required": ["mnemonic"],
            },
        },
        {
            "name": "check_mlir_op",
            "description": "Validate whether an MLIR or StableHLO operation exists with expected signature.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op_name": {
                        "type": "string",
                        "description": "Qualified dialect op name (e.g. arith.addf, stablehlo.dot_general).",
                    },
                    "operands_count": {
                        "type": "integer",
                        "description": "Optional expected number of SSA operands.",
                    },
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of attribute names to verify.",
                    },
                    "operand_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SSA operand types to verify against ODS type constraints.",
                    },
                    "structured_attributes": {
                        "type": "object",
                        "description": "Optional structured attributes (e.g. DotDimensionNumbersAttr, ComparisonDirectionAttr) to validate.",
                    },
                },
                "required": ["op_name"],
            },
        },
    ]


def handle_mcp_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle an incoming JSON-RPC / MCP protocol message and generate a response.

    Args:
        message: Parsed JSON-RPC request dictionary.

    Returns:
        JSON-RPC response dictionary.
    """
    msg_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": get_mcp_tools_list()},
        }

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "get_api_signature":
            res = get_api_signature(args.get("framework", ""), args.get("api_path", ""))
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                },
            }
        elif tool_name == "search_apis":
            res_list = search_apis(
                args.get("framework", ""),
                args.get("query", ""),
                args.get("limit", 10),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(res_list, indent=2)}
                    ]
                },
            }
        elif tool_name == "check_hallucination":
            check_res = check_hallucination(
                args.get("framework", ""),
                args.get("api_path", ""),
                args.get("kwargs", []),
                args_count=args.get("args_count"),
                strict_kwargs=args.get("strict_kwargs", True),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(check_res, indent=2)}
                    ]
                },
            }
        elif tool_name == "check_sass_instruction":
            sass_res = check_sass_instruction(
                mnemonic=args.get("mnemonic", ""),
                operands=args.get("operands"),
                modifiers=args.get("modifiers"),
                sm_arch=args.get("sm_arch"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(sass_res, indent=2)}
                    ]
                },
            }
        elif tool_name == "check_rdna_instruction":
            rdna_res = check_rdna_instruction(
                mnemonic=args.get("mnemonic", ""),
                operands=args.get("operands"),
                encoding=args.get("encoding"),
                gfx_arch=args.get("gfx_arch"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(rdna_res, indent=2)}
                    ]
                },
            }
        elif tool_name == "check_mlir_op":
            mlir_res = check_mlir_op(
                op_name=args.get("op_name", ""),
                operands_count=args.get("operands_count"),
                attributes=args.get("attributes"),
                operand_types=args.get("operand_types"),
                structured_attributes=args.get("structured_attributes"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(mlir_res, indent=2)}
                    ]
                },
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method/tool '{tool_name}' not found",
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Unknown JSON-RPC method '{method}'",
        },
    }


def run_mcp_server(
    input_stream: Optional[TextIO] = None, output_stream: Optional[TextIO] = None
) -> None:
    """Run the stdio MCP JSON-RPC loop for code-generation agent queries.

    Args:
        input_stream: Input stream to read from (defaults to sys.stdin).
        output_stream: Output stream to write to (defaults to sys.stdout).
    """
    in_s = input_stream or sys.stdin
    out_s = output_stream or sys.stdout

    for line in in_s:
        line_clean = line.strip()
        if not line_clean:
            continue
        try:
            req = json.loads(line_clean)
            resp = handle_mcp_message(req)
            out_s.write(json.dumps(resp) + "\n")
            out_s.flush()
        except Exception as e:  # pragma: no cover
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            out_s.write(json.dumps(err_resp) + "\n")
            out_s.flush()
