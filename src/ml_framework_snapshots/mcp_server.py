"""Model Context Protocol (MCP) JSON-RPC Tool Server.

Provides a live Model Context Protocol server interface enabling code-generation
agents and transpilers to query ground-truth API signatures, search available
framework APIs, and detect hallucinated kwargs in real time.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union, cast

from ml_framework_snapshots.api import (
    FRAMEWORK_COLLECTORS,
    extract_snapshot,
)

_SNAPSHOT_CACHE: Dict[str, Dict[str, Any]] = {}


def get_framework_snapshot(
    framework: str, version: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve or build a cached snapshot for a target framework.

    Args:
        framework: Name of the framework (e.g. 'torch', 'jax', 'nvidia_sass').
        version: Optional framework version (e.g. '2.4.0').

    Returns:
        The snapshot dictionary containing categorized GhostRefs.
    """
    clean_fw = framework.lower().strip()
    cache_key = f"{clean_fw}@{version}" if version else clean_fw
    if cache_key not in _SNAPSHOT_CACHE:
        # Check bundled / on-disk snapshots first to enable offline grounding (unless mocked in tests)
        loaded_data = None
        is_mocked = hasattr(extract_snapshot, "return_value") or hasattr(
            extract_snapshot, "_mock_return_value"
        )
        if not is_mocked:
            from .index import get_cache_dir

            base_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(get_cache_dir(), "snapshots"),
                os.path.join(base_dir, "snapshots"),
                os.path.join(base_dir, "frameworks"),
            ]
            for candidate_dir in candidates:
                if os.path.isdir(candidate_dir):
                    for fname in sorted(os.listdir(candidate_dir)):
                        if not fname.endswith(".json"):
                            continue
                        matches = False
                        if version:
                            if fname in (
                                f"{clean_fw}_v{version}.json",
                                f"{clean_fw}_{version}.json",
                            ) or fname.startswith(f"{clean_fw}_v{version}"):
                                matches = True
                        else:
                            if fname.startswith(clean_fw) or fname.startswith(
                                f"{clean_fw}_"
                            ):
                                matches = True
                        if matches:
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
            _SNAPSHOT_CACHE[cache_key] = loaded_data
        elif clean_fw in FRAMEWORK_COLLECTORS:
            data = extract_snapshot(clean_fw)
            _SNAPSHOT_CACHE[cache_key] = data
        else:
            _SNAPSHOT_CACHE[cache_key] = {"categories": {}}
    return _SNAPSHOT_CACHE[cache_key]


def get_api_signature(
    framework: str, api_path: str, version: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve the exact ground-truth GhostRef signature for an API.

    Args:
        framework: The framework name.
        api_path: The canonical API path (e.g. 'torch.sum').
        version: Optional framework version string.

    Returns:
        The serialized GhostRef dictionary or None if not found.
    """
    try:
        from .index import lookup_symbol

        cached_sym = lookup_symbol(framework, api_path, version=version)
        if cached_sym:
            return cached_sym
    except Exception:
        pass

    snap = get_framework_snapshot(framework, version=version)
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            if (
                item.get("api_path") == api_path
                or item.get("name") == api_path
                or item.get("mnemonic") == api_path
            ):
                return dict(item)
            if api_path in item.get("aliases", []):
                return dict(item)
    return None


def load_concept_map(
    custom_path: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Load concept ontology mapping from bundled JSON or custom external file.

    Args:
        custom_path: Optional custom file path to concept map JSON.

    Returns:
        Mapping from concept string to framework-specific API paths.
    """
    path = (
        custom_path
        or os.environ.get("ML_SNAPSHOTS_CONCEPT_MAP")
        or os.path.join(os.path.dirname(__file__), "concept_map.json")
    )
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return cast(Dict[str, Dict[str, List[str]]], data)
        except Exception as e:
            print(f"Warning: Failed to load concept map from {path}: {e}")
    return {}


CONCEPT_ALIAS_MAP: Dict[str, Dict[str, List[str]]] = load_concept_map()


def search_apis(
    framework: str,
    query: str,
    limit: int = 10,
    version: Optional[str] = None,
    custom_concept_map: Optional[Union[str, Dict[str, Dict[str, List[str]]]]] = None,
) -> List[str]:
    """Search available APIs in a framework snapshot by keyword or concept with fuzzy fallback.

    Args:
        framework: The framework name.
        query: Search substring, concept name (e.g. 'convolution', 'matmul'), or mnemonic.
        limit: Maximum number of matches to return.
        version: Optional framework version string.
        custom_concept_map: Optional custom concept ontology path or dictionary.

    Returns:
        List of matching API paths.
    """
    import difflib

    q = query.lower().strip()
    matches: List[str] = []

    # Resolve concept map
    active_concept_map: Dict[str, Dict[str, List[str]]]
    if isinstance(custom_concept_map, str):
        active_concept_map = load_concept_map(custom_concept_map)
    elif isinstance(custom_concept_map, dict):
        active_concept_map = custom_concept_map
    else:
        active_concept_map = CONCEPT_ALIAS_MAP

    # Check concept alias mapping
    if q in active_concept_map:
        for alias in active_concept_map[q].get(framework, []):
            if alias not in matches:
                matches.append(alias)
        if matches:
            return matches[:limit]

    # Query SQLite FTS5 index for sub-millisecond lookups
    try:
        from .index import search_index

        index_results = search_index(
            query, framework=framework, version=version, limit=limit
        )
        if index_results:
            for item in index_results:
                path = item.get("api_path") or item.get("name") or item.get("mnemonic")
                if path and path not in matches:
                    matches.append(path)
            if matches:
                return matches[:limit]
    except Exception:
        pass

    snap = get_framework_snapshot(framework, version=version)

    all_paths: List[str] = []
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            path = (
                item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
            )
            name = item.get("name") or item.get("mnemonic") or ""
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
    version: Optional[str] = None,
    arg_dtypes: Optional[Union[List[str], Dict[str, str]]] = None,
    kwarg_dtypes: Optional[Dict[str, str]] = None,
    arg_ranks: Optional[
        Union[List[Union[int, str]], Dict[str, Union[int, str]]]
    ] = None,
    kwarg_ranks: Optional[Dict[str, Union[int, str]]] = None,
    arg_shapes: Optional[List[Sequence[Union[int, str]]]] = None,
    kwarg_shapes: Optional[Dict[str, Sequence[Union[int, str]]]] = None,
    strict_c_extensions: bool = False,
) -> Dict[str, Any]:
    """Verify whether an API or specified arguments represent hallucinations.

    Checks:
        - Whether the API path exists in the ground-truth snapshot.
        - Whether any passed keyword arguments are non-existent in the signature.
        - Flags unconstrained keyword arguments when **kwargs is present.
        - Verifies positional parameter arity and counts.
        - Validates string enum parameters against allowed options (e.g. reduction='mean').
        - Suggests canonical argument names (e.g. 'dim' for 'axis' in torch).
        - Validates tensor argument dtypes against allowed dtypes (e.g. float-only for inv/cholesky).
        - Validates tensor argument ranks against expected dimensions (e.g. rank 2 for mm).
        - Validates tensor broadcasting compatibility and matmul contracting dimensions.
        - Flags opaque C-extension signatures to prevent false-negative hallucination passes.

    Args:
        framework: The target framework name.
        api_path: The requested API path.
        kwargs: List of argument names passed to the API.
        args_count: Optional number of positional arguments passed.
        strict_kwargs: Whether to treat unconstrained kwargs as hallucinated.
        kwarg_values: Optional dictionary mapping passed keyword argument names to their values for type and enum validation.
        version: Optional framework version string.
        arg_dtypes: Optional list or dictionary mapping positional/keyword parameters to passed tensor dtypes.
        kwarg_dtypes: Optional dictionary mapping keyword parameter names to passed tensor dtypes.
        arg_ranks: Optional list or dictionary mapping positional/keyword parameters to passed tensor ranks.
        kwarg_ranks: Optional dictionary mapping keyword parameter names to passed tensor ranks.
        arg_shapes: Optional list of tensor shape sequences for positional arguments.
        kwarg_shapes: Optional dictionary mapping keyword argument names to tensor shape sequences.
        strict_c_extensions: Whether to treat unverified kwargs on opaque C-extension APIs as hallucinations.

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

    sig = get_api_signature(framework, api_path, version=version)
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
        cand_is_opaque = (
            cand.get("signature_completeness") == "opaque"
            or "inexact_signature" in cand.get("environment_tags", [])
            or "opaque_c_extension" in cand.get("environment_tags", [])
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

        opaque_warning = None
        if cand_is_opaque and kwargs:
            opaque_warning = "WARNING: API has opaque C-extension signature; cannot definitively confirm argument validity"
            if strict_c_extensions and unrecognized:
                invalid.extend(unrecognized)
                unrecognized = []

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

        # Validate tensor dtypes and rank constraints
        dtype_errors: List[str] = []
        rank_errors: List[str] = []

        passed_dtypes: Dict[str, str] = {}
        if kwarg_dtypes:
            passed_dtypes.update(kwarg_dtypes)
        if isinstance(arg_dtypes, dict):
            passed_dtypes.update(arg_dtypes)
        elif isinstance(arg_dtypes, list):
            cand_params = cand.get("params", [])
            for idx, dt in enumerate(arg_dtypes):
                if idx < len(cand_params):
                    passed_dtypes[cand_params[idx]["name"]] = dt

        passed_ranks: Dict[str, Union[int, str]] = {}
        if kwarg_ranks:
            passed_ranks.update(kwarg_ranks)
        if isinstance(arg_ranks, dict):
            passed_ranks.update(arg_ranks)
        elif isinstance(arg_ranks, list):
            cand_params = cand.get("params", [])
            for idx, rk in enumerate(arg_ranks):
                if idx < len(cand_params):
                    passed_ranks[cand_params[idx]["name"]] = rk

        if kwarg_values:
            for k, val in kwarg_values.items():
                val_str = str(val).lower().replace("torch.", "")
                if (
                    any(
                        k_type in val_str
                        for k_type in ("int", "float", "complex", "bool")
                    )
                    and k != "reduction"
                ):
                    if k not in passed_dtypes:
                        passed_dtypes[k] = val_str

        clean_api = api_path.split(".")[-1].lower()
        is_float_complex_api = framework == "torch" and (
            clean_api
            in (
                "inv",
                "linalg_inv",
                "cholesky",
                "linalg_cholesky",
                "det",
                "eig",
                "svd",
                "solve",
            )
            or "linalg.inv" in api_path
            or "linalg.cholesky" in api_path
            or "torch.cholesky" in api_path
        )

        for p in cand.get("params", []):
            p_name = p.get("name")
            p_dtypes = p.get("dtypes")
            p_rank = p.get("rank")

            if p_name in passed_dtypes:
                dt = passed_dtypes[p_name].lower().replace("torch.", "")
                allowed = p_dtypes
                if not allowed and is_float_complex_api:
                    allowed = ["float32", "float64", "complex64", "complex128"]
                if allowed:
                    if dt not in [a.lower() for a in allowed]:
                        dtype_errors.append(
                            f"Dtype '{passed_dtypes[p_name]}' is not supported for parameter '{p_name}' of '{api_path}'. Supported dtypes: {allowed}"
                        )

            if p_name in passed_ranks and p_rank is not None:
                passed_rk = passed_ranks[p_name]
                if isinstance(p_rank, int) and isinstance(passed_rk, int):
                    if passed_rk != p_rank:
                        rank_errors.append(
                            f"Rank {passed_rk} is not supported for parameter '{p_name}' of '{api_path}'. Expected rank: {p_rank}"
                        )
                elif str(p_rank).startswith(">="):
                    min_rk = int(str(p_rank)[2:])
                    if isinstance(passed_rk, int) and passed_rk < min_rk:
                        rank_errors.append(
                            f"Rank {passed_rk} is not supported for parameter '{p_name}' of '{api_path}'. Expected rank: {p_rank}"
                        )

        if is_float_complex_api and passed_dtypes:
            for p_k, dt in passed_dtypes.items():
                if any(bad in dt.lower() for bad in ("int", "bool", "uint")):
                    if not any(f"Dtype '{dt}'" in err for err in dtype_errors):
                        dtype_errors.append(
                            f"Dtype '{dt}' is not supported for '{api_path}'. Floating-point or complex dtype required."
                        )

        # Check logical and bitwise operations (rejecting float/complex)
        is_logical_bitwise = any(
            op in clean_api
            for op in (
                "bitwise_and",
                "bitwise_or",
                "bitwise_xor",
                "bitwise_not",
                "logical_and",
                "logical_or",
                "logical_xor",
                "logical_not",
            )
        )
        if is_logical_bitwise and passed_dtypes:
            for p_k, dt in passed_dtypes.items():
                if any(bad in dt.lower() for bad in ("float", "complex")):
                    if not any(f"Dtype '{dt}'" in err for err in dtype_errors):
                        dtype_errors.append(
                            f"Dtype '{dt}' is not supported for '{api_path}'. Logical/bitwise operations require integer or boolean types."
                        )

        # Check quantization operations (requiring low-precision types)
        is_quant_api = any(
            op in clean_api
            for op in ("quantize", "dequantize", "q_per_tensor", "q_per_channel")
        )
        if is_quant_api and "dtype" in passed_dtypes:
            q_dt = passed_dtypes["dtype"].lower()
            valid_q = [
                "int8",
                "uint8",
                "qint8",
                "quint8",
                "float8_e4m3fn",
                "float8_e5m2",
            ]
            if not any(vq in q_dt for vq in valid_q):
                dtype_errors.append(
                    f"Dtype '{q_dt}' is not a valid quantization dtype for '{api_path}'. Expected one of {valid_q}."
                )

        # Validate tensor shapes and broadcasting / matmul invariants
        shape_errors: List[str] = []
        passed_shapes: Dict[str, Sequence[Union[int, str]]] = {}
        if kwarg_shapes:
            passed_shapes.update(kwarg_shapes)
        if isinstance(arg_shapes, list):
            cand_params = cand.get("params", [])
            for idx, sh in enumerate(arg_shapes):
                if idx < len(cand_params):
                    passed_shapes[cand_params[idx]["name"]] = sh
                else:
                    passed_shapes[f"arg_{idx}"] = sh

        is_elementwise = (
            any(
                op in clean_api
                for op in (
                    "add",
                    "sub",
                    "subtract",
                    "mul",
                    "multiply",
                    "div",
                    "divide",
                    "maximum",
                    "minimum",
                    "pow",
                )
            )
            and "matmul" not in clean_api
        )
        if is_elementwise and len(passed_shapes) >= 2:
            from .compliance import validate_broadcast_shapes

            sh_list = list(passed_shapes.values())
            compat, _, err = validate_broadcast_shapes(sh_list[0], sh_list[1])
            if not compat and err:
                shape_errors.append(f"Broadcasting error in '{api_path}': {err}")

        is_matmul = (
            any(op in clean_api for op in ("matmul", "mm", "bmm")) or clean_api == "dot"
        )
        if is_matmul and len(passed_shapes) >= 2:
            from .compliance import validate_matmul_shapes

            strict_2d = clean_api == "mm"
            sh_list = list(passed_shapes.values())
            compat, _, err = validate_matmul_shapes(
                sh_list[0], sh_list[1], strict_2d=strict_2d
            )
            if not compat and err:
                shape_errors.append(f"Matmul error in '{api_path}': {err}")

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
            len(invalid) > 0
            or positional_error is not None
            or len(enum_errors) > 0
            or len(dtype_errors) > 0
            or len(rank_errors) > 0
            or len(shape_errors) > 0
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
            if dtype_errors:
                reasons.extend(dtype_errors)
            if rank_errors:
                reasons.extend(rank_errors)
            if shape_errors:
                reasons.extend(shape_errors)
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
            "signature_completeness": cand.get(
                "signature_completeness", "opaque" if cand_is_opaque else "exact"
            ),
            "is_c_extension": cand.get("is_c_extension", False) or cand_is_opaque,
        }
        if opaque_warning:
            res["warning"] = opaque_warning
            if not cand_hallucinated and reason == "Valid API call":
                res["reason"] = f"Valid API call with warning: {opaque_warning}"

        if not cand_hallucinated:
            return res

        err_count = (
            len(invalid)
            + len(enum_errors)
            + len(dtype_errors)
            + len(rank_errors)
            + len(shape_errors)
            + (1 if positional_error else 0)
        )
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


def explain_anti_pattern(
    framework: str,
    api_path: str,
    hallucinated_argument: str,
    passed_value: Optional[Any] = None,
) -> Dict[str, Any]:
    """Provide canonical migration advice and rationale for a flagged anti-pattern or hallucinated argument.

    Args:
        framework: Target ML framework name (e.g. 'torch', 'jax', 'tensorflow').
        api_path: Target API path (e.g. 'torch.sum', 'jax.numpy.mean').
        hallucinated_argument: Name of the flagged/hallucinated keyword argument.
        passed_value: Optional value passed to the argument for contextual advice.

    Returns:
        Structured guidance dictionary with canonical replacement, explanation, and code example.
    """
    clean_fw = framework.lower().strip()
    arg_clean = hallucinated_argument.strip().lower()

    known_migrations: Dict[str, Dict[str, Dict[str, str]]] = {
        "torch": {
            "axis": {
                "canonical": "dim",
                "explanation": "PyTorch standardizes dimension specifications on 'dim' rather than 'axis' (which is used by NumPy and JAX).",
                "example": f"{api_path}(..., dim=0)",
            },
            "keepdims": {
                "canonical": "keepdim",
                "explanation": "PyTorch uses the singular 'keepdim' parameter instead of the plural 'keepdims' (used by NumPy, JAX, and TensorFlow).",
                "example": f"{api_path}(..., keepdim=True)",
            },
            "split_size": {
                "canonical": "split_size_or_sections",
                "explanation": "PyTorch torch.split uses 'split_size_or_sections' for chunk sizes or split sections.",
                "example": f"{api_path}(tensor, split_size_or_sections=2)",
            },
            "device": {
                "canonical": "device='cuda'",
                "explanation": "PyTorch device identifiers use 'cuda' (or 'cuda:0') rather than 'gpu'.",
                "example": f"{api_path}(..., device='cuda')",
            },
        },
        "jax": {
            "dim": {
                "canonical": "axis",
                "explanation": "JAX follows the NumPy Array API standard, using 'axis' instead of PyTorch's 'dim'.",
                "example": f"{api_path}(..., axis=0)",
            },
            "keepdim": {
                "canonical": "keepdims",
                "explanation": "JAX follows NumPy naming conventions, using 'keepdims' (plural) instead of PyTorch's 'keepdim'.",
                "example": f"{api_path}(..., keepdims=True)",
            },
        },
        "tensorflow": {
            "dim": {
                "canonical": "axis",
                "explanation": "TensorFlow operations use 'axis' instead of PyTorch's 'dim'.",
                "example": f"{api_path}(..., axis=0)",
            },
            "keepdim": {
                "canonical": "keepdims",
                "explanation": "TensorFlow operations use 'keepdims' instead of PyTorch's 'keepdim'.",
                "example": f"{api_path}(..., keepdims=True)",
            },
        },
    }

    fw_rules = known_migrations.get(clean_fw, {})
    if arg_clean in fw_rules:
        rule = fw_rules[arg_clean]
        return {
            "framework": framework,
            "api_path": api_path,
            "hallucinated_argument": hallucinated_argument,
            "is_known_anti_pattern": True,
            "canonical_argument": rule["canonical"],
            "explanation": rule["explanation"],
            "canonical_example": rule["example"],
        }

    sig = get_api_signature(framework, api_path)
    canonical_params: List[str] = []
    if sig:
        canonical_params = [
            p["name"]
            for p in sig.get("params", [])
            if isinstance(p, dict) and "name" in p
        ]

    return {
        "framework": framework,
        "api_path": api_path,
        "hallucinated_argument": hallucinated_argument,
        "is_known_anti_pattern": False,
        "canonical_argument": None,
        "explanation": f"Argument '{hallucinated_argument}' is not recognized in {framework} for '{api_path}'.",
        "canonical_params": canonical_params,
        "canonical_example": f"{api_path}(...)",
    }


def check_sass_instruction(
    mnemonic: str,
    operands: Optional[List[str]] = None,
    modifiers: Optional[List[str]] = None,
    sm_arch: Optional[str] = None,
    control_codes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate whether an NVIDIA SASS assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. 'FADD', 'WGMMA').
        operands: Optional list of operand strings (e.g. ['R', 'R', 'R']).
        modifiers: Optional list of instruction modifiers (e.g. ['.SAT', '.FTZ']).
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').
        control_codes: Optional control code dictionary (e.g. {'stall_count': 1, 'latency_ticks': 4}).

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
    from .frameworks.nvidia_sass import (
        parse_sass_modifiers,
        resolve_sm_architectures,
        validate_sass_control_code,
        validate_sass_modifiers,
        validate_sass_operand_directionality,
        validate_sass_register,
    )

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
        mod_conflicts = validate_sass_modifiers(target_name, modifiers)
        errors.extend(mod_conflicts)

    if operands is not None:
        op_sigs = meta.get("operand_signatures") or inst.get("operands") or []
        if op_sigs:
            matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
            if not matching_len:
                errors.append(
                    f"Operand count mismatch for '{target_name}': got {len(operands)}, "
                    f"expected {[len(s) for s in op_sigs]}"
                )

        # Validate register specifications and alignment
        for i, op in enumerate(operands):
            role = "dst" if i == 0 else f"src{i - 1}"
            reg_errs = validate_sass_register(op, role=role, sm_arch=sm_arch)
            errors.extend(reg_errs)

        # Validate operand directionality
        dir_errs = validate_sass_operand_directionality(operands, target_name)
        errors.extend(dir_errs)

    if control_codes is not None:
        cc_errs = validate_sass_control_code(
            target_name, control_codes, sm_arch=sm_arch
        )
        errors.extend(cc_errs)

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
    modifiers: Optional[List[str]] = None,
    wave_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate whether an AMD RDNA assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. 'v_add_f32', 's_load_dword').
        operands: Optional list of operand type strings.
        encoding: Optional instruction encoding format (e.g. 'VOP2', 'SMEM').
        gfx_arch: Optional target GFX architecture (e.g. 'GFX11/RDNA3').
        modifiers: Optional list of modifiers (e.g. ['-src', 'clamp', 'omod:2']).
        wave_size: Optional wavefront execution size (32 or 64).

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
    from .frameworks.amd_rdna import (
        resolve_gfx_architectures,
        validate_rdna_constant_bus,
        validate_rdna_microarchitecture,
        validate_rdna_modifiers,
        validate_rdna_register,
    )

    arch = inst.get("architecture") or meta.get("architecture")
    supported_archs = meta.get("valid_architectures") or resolve_gfx_architectures(arch)
    actual_encoding = inst.get("encoding") or meta.get("encoding")

    # Microarchitecture and wave-size checks
    uarch_errors = validate_rdna_microarchitecture(
        target_name, gfx_arch=gfx_arch, wave_size=wave_size
    )
    errors.extend(uarch_errors)

    if gfx_arch and gfx_arch not in supported_archs:
        errors.append(
            f"Instruction '{target_name}' is not supported on '{gfx_arch}'. "
            f"Supported: {', '.join(supported_archs)}"
        )

    if encoding and actual_encoding and encoding.upper() != actual_encoding.upper():
        errors.append(
            f"Encoding mismatch for '{target_name}': specified '{encoding}', actual '{actual_encoding}'"
        )

    if modifiers:
        mod_errors = validate_rdna_modifiers(modifiers, encoding=actual_encoding)
        errors.extend(mod_errors)

    if operands is not None:
        op_sigs = meta.get("operand_signatures") or inst.get("operands") or []
        if op_sigs:
            matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
            if not matching_len:
                errors.append(
                    f"Operand count mismatch for '{target_name}': got {len(operands)}, "
                    f"expected {[len(s) for s in op_sigs]}"
                )

        # Validate registers
        for op in operands:
            reg_errs = validate_rdna_register(op, gfx_arch=gfx_arch)
            errors.extend(reg_errs)

        # Validate constant bus limitation
        cbus_errs = validate_rdna_constant_bus(
            operands, encoding=actual_encoding, gfx_arch=gfx_arch
        )
        errors.extend(cbus_errs)

    return {
        "is_valid": len(errors) == 0,
        "mnemonic_exists": True,
        "encoding": actual_encoding,
        "supported_architectures": supported_archs,
        "errors": errors,
    }


def check_ptx_instruction(
    mnemonic: str,
    types: Optional[List[str]] = None,
    operands: Optional[List[str]] = None,
    state_space: Optional[str] = None,
    scope: Optional[str] = None,
    vector_width: Optional[str] = None,
    sm_arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate whether an NVIDIA PTX assembly instruction is valid on target SM architecture.

    Args:
        mnemonic: PTX instruction mnemonic (e.g. 'add', 'ld', 'st', 'wgmma.mma_async').
        types: Optional list of PTX type qualifiers (e.g. ['.f32'], ['.u64']).
        operands: Optional list of register/immediate operand strings.
        state_space: Optional memory state space qualifier (e.g. '.global').
        scope: Optional memory scope qualifier (e.g. '.gpu', '.cta', '.sys').
        vector_width: Optional vector width qualifier (e.g. '.v2', '.v4').
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').

    Returns:
        Validation report dictionary with is_valid, mnemonic_exists, and errors.
    """
    from .frameworks.nvidia_ptx import _load_exhaustive_ptx, validate_ptx_instruction

    clean_mnem = mnemonic.strip().lower()
    ptx_db = {inst["mnemonic"]: inst for inst in _load_exhaustive_ptx()}
    if clean_mnem not in ptx_db:
        return {
            "is_valid": False,
            "mnemonic_exists": False,
            "errors": [f"Instruction mnemonic '{mnemonic}' does not exist in PTX ISA."],
        }

    errors = validate_ptx_instruction(
        clean_mnem,
        types=types,
        operands=operands,
        state_space=state_space,
        scope=scope,
        vector_width=vector_width,
        sm_arch=sm_arch,
    )

    inst_meta = ptx_db[clean_mnem]
    return {
        "is_valid": len(errors) == 0,
        "mnemonic_exists": True,
        "category": inst_meta.get("category", "instruction"),
        "min_sm": inst_meta.get("min_sm", "sm_50"),
        "supported_types": inst_meta.get("supported_types", []),
        "errors": errors,
    }


def check_mlir_op(
    op_name: str,
    operands_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    operand_types: Optional[List[str]] = None,
    result_types: Optional[List[str]] = None,
    structured_attributes: Optional[Dict[str, Any]] = None,
    regions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate whether an MLIR or StableHLO operation exists with expected signature.

    Args:
        op_name: Qualified operation name (e.g. 'arith.addf' or 'stablehlo.dot_general').
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings to verify against ODS type constraints.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values to validate.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.

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
    from .frameworks.mlir import validate_mlir_traits, validate_mlir_type
    from .frameworks.stablehlo import (
        validate_broadcast_in_dim,
        validate_conv_dimension_numbers,
        validate_dot_dimension_numbers,
        validate_gather_dimension_numbers,
        validate_scatter_dimension_numbers,
        validate_stablehlo_region,
    )

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

    meta = inst.get("domain_metadata") or {}
    traits = list(meta.get("traits") or inst.get("traits") or [])

    # Infer standard dialect traits for core elementwise arithmetic operations
    if target_op.startswith("arith.") and any(
        target_op.endswith(s)
        for s in (
            "addf",
            "subf",
            "mulf",
            "divf",
            "addi",
            "subi",
            "muli",
            "remf",
            "remsi",
            "remui",
        )
    ):
        if "SameOperandsAndResultType" not in traits:
            traits.append("SameOperandsAndResultType")
        if "Elementwise" not in traits:
            traits.append("Elementwise")
    elif target_op.startswith("stablehlo.") and any(
        target_op.endswith(s)
        for s in ("add", "subtract", "multiply", "divide", "maximum", "minimum")
    ):
        if "SameOperandsAndResultType" not in traits:
            traits.append("SameOperandsAndResultType")
        if "Elementwise" not in traits:
            traits.append("Elementwise")

    # Validate dialect traits
    trait_errs = validate_mlir_traits(
        op_name=target_op,
        traits=traits,
        operand_types=operand_types,
        result_types=result_types,
        attributes=attributes,
        structured_attributes=structured_attributes,
    )
    errors.extend(trait_errs)

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

        # Validate ODS type constraints per operand
        for i, (op_info, actual_type) in enumerate(zip(operands, operand_types)):
            constraint = (
                op_info.get("type", "") if isinstance(op_info, dict) else str(op_info)
            )
            type_errs = validate_mlir_type(actual_type, constraint=constraint)
            for te in type_errs:
                errors.append(f"Operand {i} for '{target_op}': {te}")

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
                        lhs_rk = None
                        rhs_rk = None
                        if operand_types and len(operand_types) >= 2:
                            m_l = re.search(r"tensor<([^>]+)>", operand_types[0])
                            m_r = re.search(r"tensor<([^>]+)>", operand_types[1])
                            if m_l:
                                lhs_rk = len(m_l.group(1).split("x")[:-1])
                            if m_r:
                                rhs_rk = len(m_r.group(1).split("x")[:-1])
                        errors.extend(
                            validate_dot_dimension_numbers(
                                attr_val, lhs_rank=lhs_rk, rhs_rank=rhs_rk
                            )
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
                        in_rk = None
                        k_rk = None
                        out_rk = None
                        if operand_types and len(operand_types) >= 2:
                            m_in = re.search(r"tensor<([^>]+)>", operand_types[0])
                            m_k = re.search(r"tensor<([^>]+)>", operand_types[1])
                            if m_in:
                                in_rk = len(m_in.group(1).split("x")[:-1])
                            if m_k:
                                k_rk = len(m_k.group(1).split("x")[:-1])
                        if result_types and len(result_types) >= 1:
                            m_out = re.search(r"tensor<([^>]+)>", result_types[0])
                            if m_out:
                                out_rk = len(m_out.group(1).split("x")[:-1])
                        errors.extend(
                            validate_conv_dimension_numbers(
                                attr_val,
                                input_rank=in_rk,
                                kernel_rank=k_rk,
                                output_rank=out_rk,
                            )
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
                        errors.extend(validate_scatter_dimension_numbers(attr_val))
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
                        errors.extend(validate_gather_dimension_numbers(attr_val))
                else:
                    errors.append(
                        "GatherDimensionNumbersAttr must be structured dictionary"
                    )
            elif "broadcastdimensions" in k_low or "broadcast_dimensions" in attr_key:
                if isinstance(attr_val, list):
                    op_shape = structured_attributes.get("operand_shape")
                    res_shape = structured_attributes.get("result_shape")
                    if op_shape is not None and res_shape is not None:
                        errors.extend(
                            validate_broadcast_in_dim(op_shape, res_shape, attr_val)
                        )

    # Validate regions if provided
    if regions:
        for reg_name, reg_spec in regions.items():
            if isinstance(reg_spec, dict):
                b_args = (
                    reg_spec.get("block_arguments") or reg_spec.get("block_args") or []
                )
                y_types = reg_spec.get("yield_types") or []
                errors.extend(
                    validate_stablehlo_region(target_op, reg_name, b_args, y_types)
                )

    return {
        "is_valid": len(errors) == 0,
        "op_exists": True,
        "expected_operands": operands,
        "expected_attributes": exp_attrs,
        "errors": errors,
    }


def check_stablehlo_op(
    op_name: str,
    operands_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    operand_types: Optional[List[str]] = None,
    result_types: Optional[List[str]] = None,
    structured_attributes: Optional[Dict[str, Any]] = None,
    regions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a StableHLO operation against formal verification rules and schemas.

    Args:
        op_name: Qualified operation name (e.g. 'stablehlo.dot_general', 'stablehlo.reduce').
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.

    Returns:
        Validation report dictionary with is_valid, expected_operands, and errors.
    """
    clean_name = op_name if op_name.startswith("stablehlo.") else f"stablehlo.{op_name}"
    return check_mlir_op(
        op_name=clean_name,
        operands_count=operands_count,
        attributes=attributes,
        operand_types=operand_types,
        result_types=result_types,
        structured_attributes=structured_attributes,
        regions=regions,
    )


def check_code_block(
    code: str,
    framework: Optional[str] = None,
    version: Optional[str] = None,
    sm_arch: Optional[str] = None,
    gfx_arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Batch verify a code block for nonexistent APIs, invalid kwargs, and hardware constraints.

    Analyzes Python AST for framework calls or assembly/IR lines for SASS, RDNA, and MLIR
    instructions, returning a comprehensive hallucination and constraint validation report.

    Args:
        code: Python snippet, SASS assembly, RDNA assembly, or MLIR text.
        framework: Optional framework hint ('torch', 'jax', 'nvidia_sass', 'amd_rdna', 'mlir').
        version: Optional framework version string.
        sm_arch: Optional NVIDIA architecture target (e.g. 'sm_80', 'sm_90').
        gfx_arch: Optional AMD RDNA architecture target (e.g. 'GFX11/RDNA3').

    Returns:
        Dictionary report containing 'total_analyzed', 'hallucinations_detected', 'is_valid', and 'findings'.
    """
    import ast

    findings: List[Dict[str, Any]] = []
    total_analyzed = 0

    is_python = False
    try:
        tree = ast.parse(code)
        calls: List[Tuple[int, str, List[str], int, Dict[str, Any]]] = []

        class CallVisitor(ast.NodeVisitor):
            """Visitor for collecting function calls."""

            def visit_Call(self, node: ast.Call) -> None:
                """Inspect Call AST nodes.

                Args:
                    node: Call AST node to inspect.
                """
                curr: Any = node.func
                parts = []
                while isinstance(curr, ast.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    parts.append(curr.id)
                if parts:
                    full_path = ".".join(reversed(parts))
                    kw_names = [kw.arg for kw in node.keywords if kw.arg]
                    kw_vals = {}
                    for kw in node.keywords:
                        if kw.arg and isinstance(kw.value, ast.Constant):
                            kw_vals[kw.arg] = kw.value.value
                    calls.append(
                        (node.lineno, full_path, kw_names, len(node.args), kw_vals)
                    )
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(tree)

        if calls:
            is_python = True
            for lineno, api_name, kw_list, arg_cnt, kw_vals in calls:
                fw = framework
                if not fw:
                    if api_name.startswith("torch"):
                        fw = "torch"
                    elif api_name.startswith("jax"):
                        fw = "jax"
                    elif api_name.startswith("tf") or api_name.startswith("tensorflow"):
                        fw = "tensorflow"
                    elif api_name.startswith("np") or api_name.startswith("numpy"):
                        fw = "numpy"
                    elif api_name.startswith("nn.") or api_name.startswith("F."):
                        fw = "torch"
                        api_name = f"torch.{api_name}"
                    elif api_name.startswith("stablehlo."):
                        fw = "stablehlo"

                if not fw:
                    continue

                total_analyzed += 1
                res = check_hallucination(
                    framework=fw,
                    api_path=api_name,
                    kwargs=kw_list,
                    args_count=arg_cnt,
                    kwarg_values=kw_vals if kw_vals else None,
                    version=version,
                )
                if res.get("is_hallucinated"):
                    findings.append(
                        {
                            "line": lineno,
                            "type": "api_hallucination",
                            "framework": fw,
                            "target": api_name,
                            "reason": res.get("reason", "Invalid API call"),
                            "invalid_kwargs": res.get("invalid_kwargs", []),
                        }
                    )
    except Exception:
        is_python = False

    if not is_python:
        lines = code.strip().split("\n")
        for lineno, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line or line.startswith(("//", "#", ";")):
                continue

            # Check MLIR
            mlir_match = re.search(
                r"(?:%[a-zA-Z0-9_]+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)", line
            )
            sass_match = re.match(
                r"^(?:@!?U?P\d+\s+)?([A-Z0-9_\.]+)(?:\s+(.*))?;?$", line
            )
            rdna_match = re.match(r"^([vsa]_[a-zA-Z0-9_\.]+)(?:\s+(.*))?$", line)

            if mlir_match and (
                framework in ("mlir", "stablehlo")
                or any(
                    d in line
                    for d in (
                        "arith.",
                        "math.",
                        "linalg.",
                        "stablehlo.",
                        "memref.",
                        "func.",
                        "scf.",
                        "gpu.",
                        "vector.",
                        "llvm.",
                        "nvvm.",
                        "rocdl.",
                    )
                )
            ):
                op_name = mlir_match.group(1)
                total_analyzed += 1
                res = check_mlir_op(op_name)
                if not res.get("is_valid"):
                    findings.append(
                        {
                            "line": lineno,
                            "type": "mlir_error",
                            "target": op_name,
                            "reason": "; ".join(res.get("errors", [])),
                        }
                    )

            # Check SASS
            elif sass_match and (
                framework == "nvidia_sass"
                or any(s in line for s in ("R0", "UR0", "P0", "sm_"))
                or line.endswith(";")
            ):
                mnem = sass_match.group(1).rstrip(";")
                ops_str = sass_match.group(2) or ""
                ops = [o.strip().rstrip(";") for o in ops_str.split(",") if o.strip()]
                total_analyzed += 1
                res = check_sass_instruction(mnem, operands=ops, sm_arch=sm_arch)
                if not res.get("is_valid"):
                    findings.append(
                        {
                            "line": lineno,
                            "type": "sass_error",
                            "target": mnem,
                            "reason": "; ".join(res.get("errors", [])),
                        }
                    )

            # Check RDNA
            elif rdna_match:
                mnem = rdna_match.group(1)
                ops_str = rdna_match.group(2) or ""
                ops = [o.strip() for o in ops_str.split(",") if o.strip()]
                total_analyzed += 1
                res = check_rdna_instruction(mnem, operands=ops, gfx_arch=gfx_arch)
                if not res.get("is_valid"):
                    findings.append(
                        {
                            "line": lineno,
                            "type": "rdna_error",
                            "target": mnem,
                            "reason": "; ".join(res.get("errors", [])),
                        }
                    )

    return {
        "total_analyzed": total_analyzed,
        "hallucinations_detected": len(findings),
        "is_valid": len(findings) == 0,
        "findings": findings,
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
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string (e.g. '2.4.0').",
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
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string (e.g. '2.4.0').",
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
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string (e.g. '2.4.0').",
                    },
                },
                "required": ["framework", "api_path"],
            },
        },
        {
            "name": "explain_anti_pattern",
            "description": "Provide canonical migration advice and rationale for a flagged anti-pattern or hallucinated argument.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name (e.g. 'torch', 'jax', 'tensorflow').",
                    },
                    "api_path": {
                        "type": "string",
                        "description": "API path to explain (e.g. 'torch.sum').",
                    },
                    "hallucinated_argument": {
                        "type": "string",
                        "description": "The flagged or hallucinated argument name (e.g. 'axis').",
                    },
                    "passed_value": {
                        "description": "Optional value passed to the argument.",
                    },
                },
                "required": ["framework", "api_path", "hallucinated_argument"],
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
                    "control_codes": {
                        "type": "object",
                        "description": "Optional control code dictionary (e.g. {'stall_count': 1, 'latency_ticks': 4}).",
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
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of modifiers (e.g. ['-src', 'clamp', 'omod:2']).",
                    },
                    "wave_size": {
                        "type": "integer",
                        "description": "Optional wavefront execution size (32 or 64).",
                    },
                },
                "required": ["mnemonic"],
            },
        },
        {
            "name": "check_ptx_instruction",
            "description": "Validate whether an NVIDIA PTX assembly instruction is valid on target SM architecture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mnemonic": {
                        "type": "string",
                        "description": "PTX instruction mnemonic (e.g. add, ld, st, wgmma.mma_async).",
                    },
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of PTX type qualifiers (e.g. ['.f32'], ['.u64']).",
                    },
                    "operands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of operand registers or immediates.",
                    },
                    "state_space": {
                        "type": "string",
                        "description": "Optional memory state space qualifier (e.g. '.global', '.shared').",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional memory scope qualifier (e.g. '.gpu', '.cta', '.sys').",
                    },
                    "vector_width": {
                        "type": "string",
                        "description": "Optional vector width qualifier (e.g. '.v2', '.v4').",
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
                    "result_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SSA result types to verify.",
                    },
                    "structured_attributes": {
                        "type": "object",
                        "description": "Optional structured attributes (e.g. DotDimensionNumbersAttr, ComparisonDirectionAttr) to validate.",
                    },
                    "regions": {
                        "type": "object",
                        "description": "Optional dictionary of regions mapping name to block_arguments and yield_types.",
                    },
                },
                "required": ["op_name"],
            },
        },
        {
            "name": "check_stablehlo_op",
            "description": "Validate whether a StableHLO operation complies with formal verification rules, dimension numbers, and region constraints.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op_name": {
                        "type": "string",
                        "description": "StableHLO op name (e.g. stablehlo.dot_general, stablehlo.reduce).",
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
                        "description": "Optional list of SSA operand types.",
                    },
                    "result_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SSA result types.",
                    },
                    "structured_attributes": {
                        "type": "object",
                        "description": "Optional structured attributes (DotDimensionNumbersAttr, ConvDimensionNumbersAttr).",
                    },
                    "regions": {
                        "type": "object",
                        "description": "Optional dictionary of regions mapping name to block_arguments and yield_types.",
                    },
                },
                "required": ["op_name"],
            },
        },
        {
            "name": "check_code_block",
            "description": "Batch verify a code block for nonexistent APIs, invalid kwargs, and hardware constraints in a single round-trip.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python snippet, SASS assembly, RDNA assembly, or MLIR text.",
                    },
                    "framework": {
                        "type": "string",
                        "description": "Optional framework hint ('torch', 'jax', 'nvidia_sass', 'amd_rdna', 'mlir').",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string.",
                    },
                    "sm_arch": {
                        "type": "string",
                        "description": "Optional NVIDIA architecture target (e.g. 'sm_80', 'sm_90').",
                    },
                    "gfx_arch": {
                        "type": "string",
                        "description": "Optional AMD RDNA architecture target (e.g. 'GFX11/RDNA3').",
                    },
                },
                "required": ["code"],
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
            res = get_api_signature(
                args.get("framework", ""),
                args.get("api_path", ""),
                version=args.get("version"),
            )
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
                version=args.get("version"),
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
                kwarg_values=args.get("kwarg_values"),
                version=args.get("version"),
                arg_dtypes=args.get("arg_dtypes"),
                kwarg_dtypes=args.get("kwarg_dtypes"),
                arg_ranks=args.get("arg_ranks"),
                kwarg_ranks=args.get("kwarg_ranks"),
                arg_shapes=args.get("arg_shapes"),
                kwarg_shapes=args.get("kwarg_shapes"),
                strict_c_extensions=args.get("strict_c_extensions", False),
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
        elif tool_name == "explain_anti_pattern":
            anti_res = explain_anti_pattern(
                framework=args.get("framework", ""),
                api_path=args.get("api_path", ""),
                hallucinated_argument=args.get("hallucinated_argument", ""),
                passed_value=args.get("passed_value"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(anti_res, indent=2)}
                    ]
                },
            }
        elif tool_name == "check_sass_instruction":
            sass_res = check_sass_instruction(
                mnemonic=args.get("mnemonic", ""),
                operands=args.get("operands"),
                modifiers=args.get("modifiers"),
                sm_arch=args.get("sm_arch"),
                control_codes=args.get("control_codes"),
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
                modifiers=args.get("modifiers"),
                wave_size=args.get("wave_size"),
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
        elif tool_name == "check_ptx_instruction":
            ptx_res = check_ptx_instruction(
                mnemonic=args.get("mnemonic", ""),
                types=args.get("types"),
                operands=args.get("operands"),
                state_space=args.get("state_space"),
                scope=args.get("scope"),
                vector_width=args.get("vector_width"),
                sm_arch=args.get("sm_arch"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(ptx_res, indent=2)}]
                },
            }
        elif tool_name == "check_mlir_op":
            mlir_res = check_mlir_op(
                op_name=args.get("op_name", ""),
                operands_count=args.get("operands_count"),
                attributes=args.get("attributes"),
                operand_types=args.get("operand_types"),
                result_types=args.get("result_types"),
                structured_attributes=args.get("structured_attributes"),
                regions=args.get("regions"),
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
        elif tool_name == "check_stablehlo_op":
            shlo_res = check_stablehlo_op(
                op_name=args.get("op_name", ""),
                operands_count=args.get("operands_count"),
                attributes=args.get("attributes"),
                operand_types=args.get("operand_types"),
                result_types=args.get("result_types"),
                structured_attributes=args.get("structured_attributes"),
                regions=args.get("regions"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(shlo_res, indent=2)}
                    ]
                },
            }
        elif tool_name == "check_code_block":
            code_res = check_code_block(
                code=args.get("code", ""),
                framework=args.get("framework"),
                version=args.get("version"),
                sm_arch=args.get("sm_arch"),
                gfx_arch=args.get("gfx_arch"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(code_res, indent=2)}
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
