"""PyTorch API Snapshot Extractor.

Provides functions to dynamically introspect the PyTorch library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

import inspect
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from ml_framework_snapshots.utils import CExtensionSignature, get_all_members
from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier

import typing

try:
    import torch.nn as _nn
    import torch.optim as _optim  # pragma: no cover
    import torch.utils.data as _data  # pragma: no cover

    nn: typing.Any = _nn  # pragma: no cover
    optim: typing.Any = _optim  # pragma: no cover
    data: typing.Any = _data  # pragma: no cover
except ImportError:  # pragma: no cover
    nn = None
    optim = None
    data = None


def _scan_losses(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn` for loss classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found losses.

    """
    if not nn:
        return []
    found = []
    for name, obj in get_all_members(nn):
        if inspect.isclass(obj) and name.endswith("Loss") and name != "_Loss":
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _scan_optimizers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.optim` for optimizer classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found optimizers.

    """
    if not optim:
        return []
    found = []
    for name, obj in get_all_members(optim):
        if inspect.isclass(obj) and name != "Optimizer":
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, optim.Optimizer):
                    found.append(GhostInspector.inspect(obj, f"torch.optim.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _get_activation_names() -> Set[str]:
    """Retrieve all activation module names from torch.nn.modules.activation with fallback.

    Returns:
        Set of activation module class names.
    """
    names: Set[str] = set()
    try:
        import torch.nn.modules.activation as act_mod

        for name, obj in get_all_members(act_mod):
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                names.add(name)
    except Exception:
        pass
    if not names:
        names = {
            "ReLU",
            "Sigmoid",
            "Tanh",
            "GELU",
            "SiLU",
            "Softmax",
            "LeakyReLU",
        }
    return names


def _scan_activations(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn` for activation functions.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found activations.

    """
    if not nn:
        return []
    found = []
    target_activations = _get_activation_names()

    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module) and name in target_activations:
                    found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _scan_layers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn` for layer classes (excluding losses and activations).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found layers.

    """
    if not nn:
        return []
    found = []
    target_activations = _get_activation_names()

    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    if not name.endswith("Loss") and name not in target_activations:
                        found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:  # pragma: no cover
                pass
    return found


def _scan_schedulers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.optim.lr_scheduler` for schedulers.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found schedulers.

    """
    if not optim or not hasattr(optim, "lr_scheduler"):
        return []
    found = []
    for name, obj in get_all_members(optim.lr_scheduler):
        if inspect.isclass(obj) and name != "LRScheduler":
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(
                GhostInspector.inspect(obj, f"torch.optim.lr_scheduler.{name}")
            )
    return found


def _scan_initializers(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.nn.init` for initializers.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found initializers.

    """
    if not nn or not hasattr(nn, "init"):
        return []
    found = []
    for name, obj in get_all_members(nn.init):
        if inspect.isfunction(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(GhostInspector.inspect(obj, f"torch.nn.init.{name}"))
    return found


def _scan_metrics(include_nonpublic: bool) -> List[GhostRef]:
    """Scan for metrics (delegates to torchmetrics if installed, otherwise returns empty).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found metrics.

    """
    found = []
    try:
        import torchmetrics

        for name, obj in get_all_members(torchmetrics):
            if inspect.isclass(obj) and (include_nonpublic or not name.startswith("_")):
                try:
                    found.append(GhostInspector.inspect(obj, f"torchmetrics.{name}"))
                except Exception:  # pragma: no cover
                    pass
    except ImportError:
        pass
    return found


def _scan_dataloaders(include_nonpublic: bool) -> List[GhostRef]:
    """Scan `torch.utils.data` for dataloader-related classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found dataloaders.

    """
    if not data:
        return []
    found = []
    for name, obj in get_all_members(data):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(GhostInspector.inspect(obj, f"torch.utils.data.{name}"))
    return found


def _scan_array_api(include_nonpublic: bool) -> List[GhostRef]:
    """Scan top-level PyTorch module for array API functions and submodules.

    Exhaustively discovers:
        - All public operations in `torch.*`
        - `torch.linalg.*`
        - `torch.special.*`
        - `torch.fft.*`
        - `torch.nn.functional.*` (convolutions, pooling, linear, activations, attention, loss)

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found array API functions.
    """
    found = []
    try:
        import torch

        # 1. Top-level array API functions
        for name, obj in get_all_members(torch):
            if not include_nonpublic and name.startswith("_"):
                continue
            if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj):
                try:
                    found.append(GhostInspector.inspect(obj, f"torch.{name}"))
                except Exception:  # pragma: no cover
                    pass

        # 2. linalg module
        if hasattr(torch, "linalg"):
            for name, obj in get_all_members(torch.linalg):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and callable(obj)
                    and not inspect.isclass(obj)
                ):
                    try:
                        found.append(
                            GhostInspector.inspect(obj, f"torch.linalg.{name}")
                        )
                    except Exception:  # pragma: no cover
                        pass

        # 3. special module
        if hasattr(torch, "special"):
            for name, obj in get_all_members(torch.special):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and callable(obj)
                    and not inspect.isclass(obj)
                ):
                    try:
                        found.append(
                            GhostInspector.inspect(obj, f"torch.special.{name}")
                        )
                    except Exception:  # pragma: no cover
                        pass

        # 4. fft module
        if hasattr(torch, "fft"):
            for name, obj in get_all_members(torch.fft):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(obj, f"torch.fft.{name}")
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 5. Exhaustive nn.functional module
        if hasattr(torch, "nn") and hasattr(torch.nn, "functional"):
            for name, obj in get_all_members(torch.nn.functional):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(
                                    obj, f"torch.nn.functional.{name}"
                                )
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 6. autograd functions
        if hasattr(torch, "autograd"):
            for name, obj in get_all_members(torch.autograd):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(obj, f"torch.autograd.{name}")
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 7. distributed functions
        if hasattr(torch, "distributed"):
            for name, obj in get_all_members(torch.distributed):
                if not name.startswith("_") or include_nonpublic:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(obj, f"torch.distributed.{name}")
                            )
                        except Exception:  # pragma: no cover
                            pass

        # 8. torch.Tensor instance methods
        if hasattr(torch, "Tensor"):
            for name, obj in inspect.getmembers(torch.Tensor):
                if not include_nonpublic and name.startswith("_"):
                    continue
                if callable(obj) and not inspect.isclass(obj):
                    try:
                        found.append(
                            GhostInspector.inspect(
                                obj, f"torch.Tensor.{name}", kind="method"
                            )
                        )
                    except Exception:  # pragma: no cover
                        pass

    except ImportError:
        pass

    return found


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the PyTorch API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    if category == SemanticTier.LOSS:
        return _scan_losses(include_nonpublic)
    elif category == SemanticTier.OPTIMIZER:
        return _scan_optimizers(include_nonpublic)
    elif category == SemanticTier.ACTIVATION:
        return _scan_activations(include_nonpublic)
    elif category == SemanticTier.LAYER:
        return _scan_layers(include_nonpublic)
    elif category == SemanticTier.SCHEDULER:
        return _scan_schedulers(include_nonpublic)
    elif category == SemanticTier.INITIALIZER:
        return _scan_initializers(include_nonpublic)
    elif category == SemanticTier.METRIC:
        return _scan_metrics(include_nonpublic)
    elif category == SemanticTier.DATALOADER:
        return _scan_dataloaders(include_nonpublic)
    elif category == SemanticTier.ARRAY_API:
        return _scan_array_api(include_nonpublic)
    return []


FLOAT_COMPLEX_OPS: Set[str] = {
    "cholesky",
    "inverse",
    "linalg_inv",
    "linalg_cholesky",
    "linalg_det",
    "linalg_slogdet",
    "linalg_eig",
    "linalg_eigh",
    "linalg_eigvals",
    "linalg_eigvalsh",
    "linalg_svd",
    "linalg_svdvals",
    "linalg_solve",
    "linalg_qr",
    "linalg_lu",
    "linalg_matrix_power",
    "linalg_pinv",
}

FLOAT_OPS: Set[str] = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "rsqrt",
    "erf",
    "erfc",
    "sigmoid",
    "softmax",
    "log_softmax",
    "gelu",
    "silu",
}

RANK_RULES: Dict[str, Dict[str, Union[int, str]]] = {
    "mm": {"input": 2, "mat2": 2},
    "bmm": {"input": 3, "mat2": 3},
    "mv": {"input": 2, "mat": 2, "vec": 1},
    "dot": {"input": 1, "other": 1},
    "vdot": {"input": 1, "other": 1},
    "linalg_inv": {"input": ">=2"},
    "linalg_cholesky": {"input": ">=2"},
    "cholesky": {"input": ">=2"},
}


def infer_torch_dtype_and_rank(
    op_name: str, param_name: str, type_str: str = ""
) -> Tuple[Optional[List[str]], Optional[Union[int, str]]]:
    """Infer allowed PyTorch tensor dtypes and rank constraints for an operator parameter.

    Args:
        op_name: Name of the operator (e.g. 'linalg_inv', 'mm', 'add').
        param_name: Parameter name (e.g. 'A', 'input', 'mat2').
        type_str: Optional type annotation string.

    Returns:
        A tuple of (allowed_dtypes_list_or_None, rank_or_None).
    """
    clean_op = op_name.split(".")[-1].lower()
    clean_param = param_name.lower()

    dtypes: Optional[List[str]] = None
    rank: Optional[Union[int, str]] = None

    normalized_op = clean_op.replace("linalg.", "linalg_")
    if clean_op in FLOAT_COMPLEX_OPS or normalized_op in FLOAT_COMPLEX_OPS:
        dtypes = ["float32", "float64", "complex64", "complex128"]
    elif clean_op in FLOAT_OPS or normalized_op in FLOAT_OPS:
        dtypes = ["float16", "bfloat16", "float32", "float64"]
    elif "float" in type_str.lower():
        dtypes = ["float16", "bfloat16", "float32", "float64"]
    elif "int" in type_str.lower() and "tensor" not in type_str.lower():
        dtypes = ["int8", "int16", "int32", "int64"]

    rank_dict = RANK_RULES.get(clean_op) or RANK_RULES.get(normalized_op)
    if rank_dict:
        if clean_param in rank_dict:
            rank = rank_dict[clean_param]
        elif "input" in rank_dict and clean_param in ("self", "a"):
            rank = rank_dict["input"]

    return dtypes, rank


def parse_native_functions_yaml(
    yaml_content_or_path: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse PyTorch native_functions.yaml into structured operator definitions and overloads.

    Args:
        yaml_content_or_path: File path to native_functions.yaml or raw YAML string.

    Returns:
        Dictionary mapping base operator names to a list of overload dictionaries.
    """
    import importlib

    yaml = importlib.import_module("yaml")

    content = yaml_content_or_path
    if os.path.exists(yaml_content_or_path):
        with open(yaml_content_or_path, "r", encoding="utf-8") as f:
            content = f.read()

    data = yaml.safe_load(content)
    if not isinstance(data, list):
        return {}

    ops: Dict[str, List[Dict[str, Any]]] = {}
    sig_pattern = re.compile(
        r"^([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?\((.*)\)\s*->\s*(.*)$"
    )

    for item in data:
        if not isinstance(item, dict) or "func" not in item:
            continue
        func_str = str(item["func"]).strip()
        m = sig_pattern.match(func_str)
        if not m:
            continue

        base_op, overload_name, args_str, returns_type = m.groups()
        overload_name = overload_name or "default"
        returns_type = returns_type.strip()

        params = []
        is_kwarg_only = False
        raw_args = [a.strip() for a in args_str.split(",") if a.strip()]

        for raw_arg in raw_args:
            if raw_arg == "*":
                is_kwarg_only = True
                continue

            parts = raw_arg.split("=")
            default_val = parts[1].strip() if len(parts) > 1 else None
            decl = parts[0].strip()

            decl_parts = decl.split()
            if len(decl_parts) >= 2:
                arg_type = decl_parts[0]
                arg_name = decl_parts[1]
            else:
                arg_type = "Any"
                arg_name = decl_parts[0]

            is_out = arg_name == "out" or "(a!)" in arg_type
            clean_type = re.sub(r"\(.*?\)", "", arg_type).strip()

            dtypes, rank = infer_torch_dtype_and_rank(base_op, arg_name, clean_type)

            kind = "KEYWORD_ONLY" if is_kwarg_only else "POSITIONAL_OR_KEYWORD"
            norm_name = "input" if arg_name == "self" else arg_name

            params.append(
                {
                    "name": norm_name,
                    "kind": kind,
                    "default": default_val,
                    "annotation": clean_type,
                    "is_out": is_out,
                    "dtypes": dtypes,
                    "rank": rank,
                }
            )

        overload_info = {
            "func": func_str,
            "name": base_op,
            "overload_name": overload_name,
            "params": params,
            "returns_type": returns_type,
            "is_inplace": base_op.endswith("_"),
            "is_out": any(p.get("is_out") for p in params),
        }

        if base_op not in ops:
            ops[base_op] = []
        ops[base_op].append(overload_info)

    return ops


def get_aten_op_schema(op_name: str) -> Optional[List[Dict[str, Any]]]:
    """Retrieve structured schema and overloads for an ATen operator from torch.ops.aten.

    Args:
        op_name: Operator name (e.g. 'add', 'relu', 'matmul', 'add_').

    Returns:
        List of overload dictionaries if found, otherwise None.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    clean_op = op_name.split(".")[-1]
    aten_op = getattr(torch.ops.aten, clean_op, None)
    if not aten_op or not hasattr(aten_op, "overloads"):
        return None

    results: List[Dict[str, Any]] = []
    for ov_name in aten_op.overloads():
        ov = getattr(aten_op, ov_name)
        schema = getattr(ov, "_schema", None)
        if not schema:
            continue

        params: List[Dict[str, Any]] = []
        for arg in schema.arguments:
            p_kind = "KEYWORD_ONLY" if arg.kwarg_only else "POSITIONAL_OR_KEYWORD"
            p_name = "input" if arg.name == "self" else arg.name
            p_default = str(arg.default_value) if arg.has_default_value() else None
            p_anno = str(arg.type)
            is_out = getattr(arg, "is_out", False)

            dtypes, rank = infer_torch_dtype_and_rank(clean_op, p_name, p_anno)

            params.append(
                {
                    "name": p_name,
                    "kind": p_kind,
                    "default": p_default,
                    "annotation": p_anno,
                    "is_out": is_out,
                    "dtypes": dtypes,
                    "rank": rank,
                }
            )

        ret_type = str(schema.returns[0].type) if schema.returns else "Tensor"
        results.append(
            {
                "overload_name": ov_name,
                "params": params,
                "returns_type": ret_type,
                "is_inplace": clean_op.endswith("_"),
                "is_out": any(p.get("is_out") for p in params),
            }
        )

    return results if results else None


def extract_aten_c_extension_signature(
    target: Any, name: Optional[str] = None, is_method: bool = False
) -> Optional[CExtensionSignature]:
    """Extract a CExtensionSignature with exhaustive overloads from torch.ops.aten.

    Args:
        target: Target callable or operator.
        name: Optional qualified name or attribute path.
        is_method: Whether the target is an instance method (stripping receiver 'self').

    Returns:
        CExtensionSignature with primary parameters and overloads if found, else None.
    """
    op_name = str(name or getattr(target, "__name__", "") or "")
    schemas = get_aten_op_schema(op_name)
    if not schemas:
        return None

    is_method_call = is_method or (name is not None and ".Tensor." in name)

    primary_idx = 0
    for idx, ov in enumerate(schemas):
        if ov.get("overload_name") in ("Tensor", "default"):
            primary_idx = idx
            break

    primary = schemas[primary_idx]
    secondary = [ov for idx, ov in enumerate(schemas) if idx != primary_idx]

    def to_param_tuples(
        p_list: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
        """Convert parameter dictionary list to CExtensionSignature tuple format.

        Args:
            p_list: List of parameter dictionaries.

        Returns:
            List of (name, kind, default, annotation) tuples.
        """
        filtered = p_list
        if is_method_call and filtered and filtered[0]["name"] in ("input", "self"):
            filtered = filtered[1:]
        return [(p["name"], p["kind"], p["default"], p["annotation"]) for p in filtered]

    overload_objs = [
        CExtensionSignature(
            params=to_param_tuples(ov["params"]),
            returns_type=ov.get("returns_type"),
            overloads=[],
        )
        for ov in secondary
    ]

    return CExtensionSignature(
        params=to_param_tuples(primary["params"]),
        returns_type=primary.get("returns_type"),
        overloads=overload_objs,
    )
