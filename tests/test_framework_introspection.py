"""Tests for multi-framework collectors: IR, ONNX Runtime, NumPy, MLX, HuggingFace, TensorFlow, and JAX."""

import builtins
import importlib
import inspect
import sys
import types
import typing
from typing import Any
from unittest.mock import patch

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ml_framework_snapshots.frameworks import (
    huggingface as hf_mod,
    ir as ir_mod,
    jax as jax_fw,
    mlx as mlx_mod,
    numpy as numpy_mod,
    onnxruntime as ort_mod,
    tensorflow as tf_fw,
)
from ml_framework_snapshots.models import GhostInspector


def test_ir_collect_api_all_categories() -> None:
    """Test ir_mod.collect_api across supported and unsupported categories."""
    # Supported categories
    refs_array = ir_mod.collect_api(SemanticTier.ARRAY_API)
    assert len(refs_array) == 3
    names = {r.name for r in refs_array}
    assert "LogicalGraph" in names
    assert "LogicalNode" in names
    assert "topological_sort" in names

    refs_neural = ir_mod.collect_api(SemanticTier.NEURAL, include_nonpublic=True)
    assert len(refs_neural) == 3

    refs_util = ir_mod.collect_api(SemanticTier.UTIL)
    assert len(refs_util) == 3

    # Unsupported category triggers empty return
    refs_loss = ir_mod.collect_api(SemanticTier.LOSS)
    assert refs_loss == []


def test_onnxruntime_collect_api_edge_branches(mocker: Any) -> None:
    """Test onnxruntime.collect_api for full line and branch coverage.

    Args:
        mocker: Pytest mocker fixture.
    """
    # 1. Unsupported category
    assert ort_mod.collect_api(SemanticTier.LOSS) == []

    # 2. ImportError
    real_import = importlib.import_module

    def mock_import_err(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising on onnxruntime.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Raises:
            ImportError: Simulated import error.

        Returns:
            Imported module.
        """
        if name == "onnxruntime":
            raise ImportError("simulated onnxruntime error")
        return real_import(name, *args, **kwargs)

    with patch("importlib.import_module", side_effect=mock_import_err):
        assert ort_mod.collect_api(SemanticTier.MODEL) == []

    # 3. Comprehensive mock module covering all branches
    fake_mod = types.ModuleType("onnxruntime")

    class InferenceSession:
        """Mock InferenceSession class."""

        pass

    class OtherSession:
        """Mock session class."""

        pass

    def dummy_util() -> None:
        """Mock utility function."""
        pass

    def _private_func() -> None:
        """Private function."""
        pass

    setattr(fake_mod, "InferenceSession", InferenceSession)
    setattr(fake_mod, "OtherSession", OtherSession)
    setattr(fake_mod, "dummy_util", dummy_util)
    setattr(fake_mod, "_private_func", _private_func)
    setattr(fake_mod, "none_obj", None)
    setattr(fake_mod, "raise_exc", lambda: None)
    setattr(fake_mod, "return_none_ref", lambda: None)

    def mock_import_fake(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import returning fake onnxruntime.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Imported module.
        """
        if name == "onnxruntime":
            return fake_mod
        return real_import(name, *args, **kwargs)

    providers_present = [False]

    def mock_inspect(obj: Any, path: str) -> Any:
        """Mock inspect to trigger various inspection outcomes.

        Args:
            obj: Target object.
            path: Target path string.

        Returns:
            GhostRef or None, or raises exception.
        """
        if "raise_exc" in path:
            raise RuntimeError("simulated inspect error")
        if "return_none_ref" in path:
            return None
        if "InferenceSession" in path and providers_present[0]:
            return GhostRef(
                name="InferenceSession",
                api_path=path,
                kind="class",
                params=[GhostParam(name="providers", kind=ParameterKind.KEYWORD_ONLY)],
            )
        return GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="class" if inspect.isclass(obj) else "function",
            params=[],
        )

    mocker.patch.object(GhostInspector, "inspect", side_effect=mock_inspect)

    with patch("importlib.import_module", side_effect=mock_import_fake):
        # Pass 1: InferenceSession without providers parameter (branch not has_providers is True)
        providers_present[0] = False
        res_model = ort_mod.collect_api(SemanticTier.MODEL, include_nonpublic=False)
        names_model = {r.name for r in res_model}
        assert "InferenceSession" in names_model
        assert "OtherSession" in names_model

        # Verify InferenceSession has providers injected
        inf_ref = next(r for r in res_model if r.name == "InferenceSession")
        assert any(p.name == "providers" for p in inf_ref.params)

        # Pass 2: InferenceSession WITH existing providers parameter (branch 55 -> 66: not has_providers is False)
        providers_present[0] = True
        res_model2 = ort_mod.collect_api(SemanticTier.MODEL, include_nonpublic=False)
        inf_ref2 = next(r for r in res_model2 if r.name == "InferenceSession")
        assert len([p for p in inf_ref2.params if p.name == "providers"]) == 1

        # UTIL category with include_nonpublic=True:
        # - includes _private_func
        # - skips Sessions (obj_cat == MODEL, category == UTIL)
        # - inspects dummy_util
        # - handles return_none_ref returning None
        # - handles raise_exc raising Exception
        res_util = ort_mod.collect_api(SemanticTier.UTIL, include_nonpublic=True)
        names_util = {r.name for r in res_util}
        assert "dummy_util" in names_util
        assert "_private_func" in names_util


def test_numpy_collect_api_all_branches(mocker: Any) -> None:
    """Test numpy.collect_api for full line and branch coverage.

    Args:
        mocker: Pytest mocker fixture.
    """
    # 1. np is None branch
    mocker.patch.object(numpy_mod, "np", None)
    assert numpy_mod.collect_api(SemanticTier.ACTIVATION) == []
    assert numpy_mod.collect_api(SemanticTier.ARRAY_API) == []

    # 2. Unsupported category
    fake_np = types.ModuleType("numpy")
    mocker.patch.object(numpy_mod, "np", fake_np)
    assert numpy_mod.collect_api(SemanticTier.MODEL) == []

    # 3. ACTIVATION category:
    # - has callable ops that succeed
    # - has non-callable op (branch 37 -> 36: not callable(obj))
    # - has missing op (hasattr is False)
    # - has op where inspect raises Exception
    def tanh() -> None:
        """Fake tanh."""
        pass

    def exp() -> None:
        """Fake exp."""
        pass

    setattr(fake_np, "tanh", tanh)
    setattr(fake_np, "exp", exp)
    setattr(fake_np, "maximum", 42)
    # minimum is omitted so hasattr is False

    original_inspect = GhostInspector.inspect

    def mock_inspect_act(obj: Any, name: str, is_public: bool = True) -> Any:
        """Mock inspect for activation.

        Args:
            obj: Object to inspect.
            name: Path.
            is_public: Public flag.

        Returns:
            GhostRef or raises Exception.
        """
        if "exp" in name:
            raise RuntimeError("simulated inspect error")
        return original_inspect(obj, name, is_public=is_public)

    mocker.patch.object(GhostInspector, "inspect", side_effect=mock_inspect_act)
    res_act = numpy_mod.collect_api(SemanticTier.ACTIVATION)
    assert len(res_act) == 1
    assert res_act[0].name == "tanh"

    # 4. ARRAY_API category:
    # - test array_ops with callable, non-callable, missing, and exception
    fake_np_array = types.ModuleType("numpy")

    def abs() -> None:
        """Fake abs."""
        pass

    def add() -> None:
        """Fake add."""
        pass

    setattr(fake_np_array, "abs", abs)
    setattr(fake_np_array, "add", add)
    setattr(fake_np_array, "all", "not_callable")
    # remaining array_ops not defined -> hasattr False

    # linalg submodule:
    # - private member (startswith '_')
    # - class member (inspect.isclass is True -> skipped)
    # - non-callable member (callable is False -> skipped)
    # - valid callable member that succeeds
    # - valid callable member where inspect raises Exception
    class LinAlgClass:
        """Class in linalg."""

        pass

    def det() -> None:
        """Fake det."""
        pass

    def inv() -> None:
        """Fake inv."""
        pass

    fake_linalg = types.ModuleType("numpy.linalg")
    setattr(fake_linalg, "_private", lambda: None)
    setattr(fake_linalg, "LinAlgClass", LinAlgClass)
    setattr(fake_linalg, "non_callable", 123)
    setattr(fake_linalg, "det", det)
    setattr(fake_linalg, "inv", inv)
    setattr(fake_np_array, "linalg", fake_linalg)

    # fft submodule:
    # - private member
    # - class member
    # - non-callable member
    # - valid callable member that succeeds
    # - valid callable member where inspect raises Exception
    class FFTClass:
        """Class in fft."""

        pass

    def fft() -> None:
        """Fake fft."""
        pass

    def ifft() -> None:
        """Fake ifft."""
        pass

    fake_fft_mod = types.ModuleType("numpy.fft")
    setattr(fake_fft_mod, "_private", lambda: None)
    setattr(fake_fft_mod, "FFTClass", FFTClass)
    setattr(fake_fft_mod, "non_callable", 456)
    setattr(fake_fft_mod, "fft", fft)
    setattr(fake_fft_mod, "ifft", ifft)
    setattr(fake_np_array, "fft", fake_fft_mod)

    def mock_inspect_arr(obj: Any, name: str, is_public: bool = True) -> Any:
        """Mock inspect for array api.

        Args:
            obj: Object to inspect.
            name: Path.
            is_public: Public flag.

        Returns:
            GhostRef or raises Exception.
        """
        if "add" in name or "inv" in name or "ifft" in name:
            raise RuntimeError("simulated inspect error")
        return original_inspect(obj, name, is_public=is_public)

    mocker.patch.object(numpy_mod, "np", fake_np_array)
    mocker.patch.object(GhostInspector, "inspect", side_effect=mock_inspect_arr)

    res_arr = numpy_mod.collect_api(SemanticTier.ARRAY_API)
    names_arr = {r.name for r in res_arr}
    assert "abs" in names_arr
    assert "det" in names_arr
    assert "fft" in names_arr

    # Also test when np does not have linalg or fft attributes
    fake_np_bare = types.ModuleType("numpy")
    setattr(fake_np_bare, "abs", abs)
    mocker.patch.object(numpy_mod, "np", fake_np_bare)
    res_bare = numpy_mod.collect_api(SemanticTier.ARRAY_API)
    assert len(res_bare) == 1


def test_numpy_import_error_handling() -> None:
    """Test numpy module import failure fallback branch."""
    saved_np = numpy_mod.np
    saved_sys_np = sys.modules.get("numpy")

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on numpy.

        Args:
            name: Module name.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            ImportError: Simulated import error.

        Returns:
            Imported module.
        """
        if name == "numpy":
            raise ImportError("simulated numpy import error")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(numpy_mod)
        assert numpy_mod.np is None

    # Restore clean numpy import
    if saved_sys_np is not None:
        sys.modules["numpy"] = saved_sys_np
    numpy_mod.np = saved_np
    assert numpy_mod.np is not None


def test_mlx_collect_api_all_branches(mocker: Any) -> None:
    """Test mlx.collect_api for all categories, lines, and branches.

    Args:
        mocker: Pytest mocker fixture.
    """
    # 1. mlx is None
    mocker.patch.object(mlx_mod, "mlx", None)
    assert mlx_mod.collect_api(SemanticTier.LAYER) == []

    # 2. Mock mlx structure
    fake_mlx = types.ModuleType("mlx")
    fake_nn = types.ModuleType("mlx.nn")
    fake_losses = types.ModuleType("mlx.nn.losses")
    fake_optimizers = types.ModuleType("mlx.optimizers")
    fake_core = types.ModuleType("mlx.core")
    fake_fft = types.ModuleType("mlx.core.fft")
    fake_linalg = types.ModuleType("mlx.core.linalg")

    setattr(fake_mlx, "nn", fake_nn)
    setattr(fake_mlx, "optimizers", fake_optimizers)
    setattr(fake_mlx, "core", fake_core)

    # LAYER category
    class UpperLayer:
        """Valid layer class with uppercase name."""

        pass

    class lower_layer:
        """Class with lowercase name."""

        pass

    class _PrivateLayer:
        """Private layer class."""

        pass

    def not_a_class() -> None:
        """Function not a class."""
        pass

    setattr(fake_nn, "UpperLayer", UpperLayer)
    setattr(fake_nn, "lower_layer", lower_layer)
    setattr(fake_nn, "_PrivateLayer", _PrivateLayer)
    setattr(fake_nn, "not_a_class", not_a_class)

    # ACTIVATION category
    setattr(fake_nn, "relu", lambda x: x)
    setattr(fake_nn, "_relu", lambda x: x)
    setattr(fake_nn, "unrelated_act", lambda x: x)

    # LOSS category
    def cross_entropy_loss() -> None:
        """Function containing loss."""
        pass

    class CustomLoss:
        """Class containing loss."""

        pass

    def _private_loss() -> None:
        """Private loss function."""
        pass

    setattr(fake_losses, "cross_entropy_loss", cross_entropy_loss)
    setattr(fake_losses, "CustomLoss", CustomLoss)
    setattr(fake_losses, "_private_loss", _private_loss)
    setattr(fake_losses, "something_else", lambda: None)
    setattr(fake_losses, "number_loss", 999)
    setattr(fake_nn, "losses", fake_losses)

    # OPTIMIZER category
    class Adam:
        """Valid optimizer class."""

        pass

    class adam_lower:
        """Lowercase optimizer class."""

        pass

    class _PrivateOptimizer:
        """Private optimizer class."""

        pass

    setattr(fake_optimizers, "Adam", Adam)
    setattr(fake_optimizers, "adam_lower", adam_lower)
    setattr(fake_optimizers, "_PrivateOptimizer", _PrivateOptimizer)
    setattr(fake_optimizers, "optim_func", lambda: None)

    # ARRAY_API category
    def fake_abs() -> None:
        """Fake abs function."""
        pass

    def fake_core_exc() -> None:
        """Core function that triggers inspect exception."""
        pass

    class CoreClass:
        """Class in core."""

        pass

    setattr(fake_core, "abs", fake_abs)
    setattr(fake_core, "fake_core_exc", fake_core_exc)
    setattr(fake_core, "CoreClass", CoreClass)
    setattr(fake_core, "_private_core", lambda: None)

    def fake_fft_fn() -> None:
        """Fake fft function."""
        pass

    def fake_fft_exc() -> None:
        """FFT function that triggers inspect exception."""
        pass

    setattr(fake_fft, "fft", fake_fft_fn)
    setattr(fake_fft, "fake_fft_exc", fake_fft_exc)
    setattr(fake_fft, "ClassFFT", CoreClass)
    setattr(fake_fft, "_private_fft", lambda: None)
    setattr(fake_core, "fft", fake_fft)

    def fake_linalg_fn() -> None:
        """Fake linalg function."""
        pass

    def fake_linalg_exc() -> None:
        """Linalg function that triggers inspect exception."""
        pass

    setattr(fake_linalg, "norm", fake_linalg_fn)
    setattr(fake_linalg, "fake_linalg_exc", fake_linalg_exc)
    setattr(fake_linalg, "ClassLinalg", CoreClass)
    setattr(fake_linalg, "_private_linalg", lambda: None)
    setattr(fake_core, "linalg", fake_linalg)

    mocker.patch.object(mlx_mod, "mlx", fake_mlx)

    def mock_inspect(obj: Any, path: str) -> Any:
        """Mock inspect to simulate exceptions on targeted functions.

        Args:
            obj: Target object.
            path: Target path string.

        Returns:
            GhostRef or raises Exception.
        """
        if (
            "fake_core_exc" in path
            or "fake_fft_exc" in path
            or "fake_linalg_exc" in path
        ):
            raise RuntimeError("simulated inspect error")
        return GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="class" if inspect.isclass(obj) else "function",
            params=[],
        )

    mocker.patch.object(GhostInspector, "inspect", side_effect=mock_inspect)

    # Unsupported category triggers falling through elif chain to line 150 (branch 100 -> 150)
    assert mlx_mod.collect_api(SemanticTier.MODEL) == []

    # LAYER: public vs nonpublic
    layers_pub = mlx_mod.collect_api(SemanticTier.LAYER, include_nonpublic=False)
    assert {r.name for r in layers_pub} == {"UpperLayer"}
    layers_all = mlx_mod.collect_api(SemanticTier.LAYER, include_nonpublic=True)
    assert {r.name for r in layers_all} == {"UpperLayer"}

    # ACTIVATION: public vs nonpublic
    act_pub = mlx_mod.collect_api(SemanticTier.ACTIVATION, include_nonpublic=False)
    assert {r.name for r in act_pub} == {"relu"}
    act_all = mlx_mod.collect_api(SemanticTier.ACTIVATION, include_nonpublic=True)
    assert {r.name for r in act_all} == {"relu"}

    # LOSS: public vs nonpublic, and test when losses attr is missing
    losses_pub = mlx_mod.collect_api(SemanticTier.LOSS, include_nonpublic=False)
    assert {r.name for r in losses_pub} == {"cross_entropy_loss", "CustomLoss"}
    losses_all = mlx_mod.collect_api(SemanticTier.LOSS, include_nonpublic=True)
    assert "_private_loss" in {r.name for r in losses_all}

    del fake_nn.losses
    assert mlx_mod.collect_api(SemanticTier.LOSS) == []
    setattr(fake_nn, "losses", fake_losses)

    # OPTIMIZER: public vs nonpublic
    opt_pub = mlx_mod.collect_api(SemanticTier.OPTIMIZER, include_nonpublic=False)
    assert {r.name for r in opt_pub} == {"Adam"}
    opt_all = mlx_mod.collect_api(SemanticTier.OPTIMIZER, include_nonpublic=True)
    assert {r.name for r in opt_all} == {"Adam"}

    # ARRAY_API: includes core, fft, linalg, handles exceptions, public vs nonpublic
    arr_pub = mlx_mod.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    names_arr = {r.name for r in arr_pub}
    assert "abs" in names_arr
    assert "fft" in names_arr
    assert "norm" in names_arr
    assert "fake_core_exc" not in names_arr

    arr_all = mlx_mod.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    names_arr_all = {r.name for r in arr_all}
    assert "_private_core" in names_arr_all
    assert "_private_fft" in names_arr_all
    assert "_private_linalg" in names_arr_all

    # Test ARRAY_API when core is missing
    fake_mlx_empty = types.ModuleType("mlx")
    mocker.patch.object(mlx_mod, "mlx", fake_mlx_empty)
    assert mlx_mod.collect_api(SemanticTier.ARRAY_API) == []

    # Test ARRAY_API when fft and linalg submodules are missing from core (branches 117->131, 132->150)
    fake_mlx_no_sub = types.ModuleType("mlx")
    fake_core_no_sub = types.ModuleType("mlx.core")
    setattr(fake_mlx_no_sub, "core", fake_core_no_sub)
    mocker.patch.object(mlx_mod, "mlx", fake_mlx_no_sub)
    assert mlx_mod.collect_api(SemanticTier.ARRAY_API) == []

    # Test outer exception handling in _collect_live (lines 147-148)
    mocker.patch.object(mlx_mod, "mlx", fake_mlx)
    mocker.patch(
        "ml_framework_snapshots.frameworks.mlx.get_all_members",
        side_effect=RuntimeError("simulated get_all_members crash"),
    )
    assert mlx_mod.collect_api(SemanticTier.LAYER) == []


def test_mlx_import_error_handling() -> None:
    """Test mlx module import success and failure fallback branches."""
    # 1. Test simulated import success of mlx
    mock_mlx_root = types.ModuleType("mlx")
    mock_mlx_core = types.ModuleType("mlx.core")
    mock_mlx_nn = types.ModuleType("mlx.nn")
    mock_mlx_opt = types.ModuleType("mlx.optimizers")

    with patch.dict(
        sys.modules,
        {
            "mlx": mock_mlx_root,
            "mlx.core": mock_mlx_core,
            "mlx.nn": mock_mlx_nn,
            "mlx.optimizers": mock_mlx_opt,
        },
    ):
        importlib.reload(mlx_mod)
        assert mlx_mod.mlx is mock_mlx_root

    # 2. Test simulated import failure of mlx
    real_import = builtins.__import__

    def mock_import_fail(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising on mlx.

        Args:
            name: Module name.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            ImportError: Simulated import error.

        Returns:
            Imported module.
        """
        if name.startswith("mlx"):
            raise ImportError("simulated mlx import error")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import_fail):
        importlib.reload(mlx_mod)
        assert mlx_mod.mlx is None

    # Restore clean reload
    importlib.reload(mlx_mod)


def test_huggingface_collect_api_all_branches(mocker: Any) -> None:
    """Test huggingface.collect_huggingface for full line and branch coverage.

    Args:
        mocker: Pytest mocker fixture.
    """
    # 1. Category not in category_mapping
    mapping = {SemanticTier.MODEL: ["Model"]}
    assert (
        hf_mod.collect_huggingface("transformers", mapping, SemanticTier.OPTIMIZER)
        == []
    )

    # 2. ImportError
    real_import = importlib.import_module

    def mock_import_err(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising on transformers.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Raises:
            ImportError: Simulated import error.

        Returns:
            Imported module.
        """
        if name == "transformers":
            raise ImportError("simulated transformers error")
        return real_import(name, *args, **kwargs)

    with patch("importlib.import_module", side_effect=mock_import_err):
        assert (
            hf_mod.collect_huggingface("transformers", mapping, SemanticTier.MODEL)
            == []
        )

    # 3. _extract_generation_kwargs branches:
    # - inspect.signature raises Exception (lines 46-47)
    class DummyGenWithException:
        """Class whose generate method cannot be inspected."""

        def generate(self) -> None:
            """Generate method."""
            pass

    ref = GhostRef(name="DummyGen", api_path="DummyGen", kind="class", params=[])
    mocker.patch(
        "ml_framework_snapshots.frameworks.huggingface.inspect.signature",
        side_effect=TypeError("cannot inspect signature"),
    )
    hf_mod._extract_generation_kwargs(DummyGenWithException(), ref)
    assert len(ref.params) == 0

    # - successful signature extraction covering self, varargs, varkw, existing param, default empty/nonempty, anno empty/nonempty
    class DummyGenFull:
        """Class with comprehensive generate signature."""

        @typing.no_type_check
        def generate(
            self,
            existing_param: str,
            default_val: int = 10,
            no_default: float = 0.0,
            no_anno=None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            """Generate method.

            Args:
                existing_param: Existing param.
                default_val: Default value param.
                no_default: No default param.
                no_anno: No annotation param.
                *args: Varargs.
                **kwargs: Varkwargs.
            """
            pass

    ref_gen = GhostRef(
        name="DummyGenFull",
        api_path="DummyGenFull",
        kind="class",
        params=[
            GhostParam(name="existing_param", kind=ParameterKind.POSITIONAL_OR_KEYWORD)
        ],
    )
    mocker.stopall()
    hf_mod._extract_generation_kwargs(DummyGenFull(), ref_gen)
    gen_param_names = [p.name for p in ref_gen.params]
    assert "existing_param" in gen_param_names
    assert "default_val" in gen_param_names
    assert "no_default" in gen_param_names
    assert "no_anno" in gen_param_names
    assert len([p for p in ref_gen.params if p.name == "existing_param"]) == 1

    # 4. _parse_pretrained_config:
    # - non-annotated class (lines 57-58: if not hasattr(obj, "__annotations__"): return)
    # - already-existing param (branch 64 -> 60)
    # - string annotation vs non-string annotation
    class NoAnnotations:
        """Class without __annotations__."""

        pass

    ref2 = GhostRef(name="NoAnno", api_path="NoAnno", kind="class", params=[])
    hf_mod._parse_pretrained_config(NoAnnotations(), ref2)
    assert len(ref2.params) == 0

    class AnnotatedConfig:
        """Class with annotations."""

        __annotations__ = {
            "existing": int,
            "str_anno": "str",
            "type_anno": float,
        }

    ref3 = GhostRef(
        name="AnnotatedConfig",
        api_path="AnnotatedConfig",
        kind="class",
        params=[GhostParam(name="existing", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    hf_mod._parse_pretrained_config(AnnotatedConfig, ref3)
    param_names = [p.name for p in ref3.params]
    assert "existing" in param_names
    assert "str_anno" in param_names
    assert "type_anno" in param_names
    assert len([p for p in ref3.params if p.name == "existing"]) == 1

    # 5. _handle_automodel_factory:
    # - already has config param (branch 85 -> exit)
    ref4 = GhostRef(
        name="AutoModel",
        api_path="AutoModel",
        kind="class",
        params=[GhostParam(name="config", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    hf_mod._handle_automodel_factory(None, "AutoModel", ref4)
    assert len([p for p in ref4.params if p.name == "config"]) == 1

    # 6. collect_huggingface with large transformers module (>100 attrs):
    # - name does NOT start/end with accepted token -> line 143 continue
    # - getattr raises Exception -> line 146 continue
    # - obj is None -> line 148 continue
    # - inspect raises Exception -> lines 178-179
    class SampleGenModel:
        """Sample model with generate method."""

        def generate(self, input_ids: str = "") -> None:
            """Generate method.

            Args:
                input_ids: Input IDs.
            """
            pass

    class LargeMod:
        """Large mock module simulating transformers."""

        def __dir__(self) -> Any:
            """Return >100 attribute names.

            Returns:
                List of string attribute names.
            """
            names = [f"ignored_symbol_{i}" for i in range(105)]
            names.extend(
                [
                    "_private_skip",
                    "AutoConfig",
                    "AutoModelWithGen",
                    "PreTrainedModelValid",
                    "CustomPipeline",
                    "CustomTokenizer",
                    "CustomTokenizerFast",
                    "TFAutoModelValid",
                    "FlaxAutoModelValid",
                    "AutoScheduler",
                    "pipeline_fn",
                    "AutoRaiseGetattr",
                    "AutoNone",
                    "AutoReturnNoneRef",
                    "AutoInspectExc",
                ]
            )
            return names

        def __getattr__(self, name: str) -> Any:
            """Simulate getattr results.

            Args:
                name: Attribute name.

            Raises:
                RuntimeError: When simulated getattr error is requested.

            Returns:
                Attribute value or None.
            """
            if name == "AutoRaiseGetattr":
                raise RuntimeError("simulated getattr failure")
            if name == "AutoNone":
                return None
            if name == "AutoReturnNoneRef":
                return lambda: None
            if name == "AutoInspectExc":
                return lambda: None
            if name == "AutoConfig":
                return type("AutoConfig", (), {"__annotations__": {"vocab_size": int}})
            if name == "AutoModelWithGen":
                return SampleGenModel()
            return lambda: None

    def mock_import_large(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import returning LargeMod.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Imported module.
        """
        if name == "transformers":
            return LargeMod()
        return real_import(name, *args, **kwargs)

    def mock_inspect_large(obj: Any, path: str) -> Any:
        """Mock inspect for large transformers module.

        Args:
            obj: Object to inspect.
            path: Target path string.

        Returns:
            GhostRef or raises Exception.
        """
        if "AutoInspectExc" in path:
            raise RuntimeError("simulated inspect error")
        if "AutoReturnNoneRef" in path:
            return None
        return GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
        )

    mocker.patch.object(GhostInspector, "inspect", side_effect=mock_inspect_large)

    mapping_transformers = {
        SemanticTier.MODEL: [
            "Config",
            "Model",
            "Pipeline",
            "Tokenizer",
            "TFAuto",
            "FlaxAuto",
            "pipeline",
            "PreTrained",
        ],
        SemanticTier.OPTIMIZER: ["Scheduler"],
        SemanticTier.UTIL: ["other"],
    }
    with patch("importlib.import_module", side_effect=mock_import_large):
        # MODEL category
        res_large_model = hf_mod.collect_huggingface(
            "transformers",
            mapping_transformers,
            SemanticTier.MODEL,
            include_nonpublic=False,
        )
        names_large_model = {r.name for r in res_large_model}
        assert "AutoConfig" in names_large_model
        assert "AutoModelWithGen" in names_large_model
        assert "PreTrainedModelValid" in names_large_model
        assert "CustomPipeline" in names_large_model
        assert "CustomTokenizer" in names_large_model
        assert "CustomTokenizerFast" in names_large_model
        assert "TFAutoModelValid" in names_large_model
        assert "FlaxAutoModelValid" in names_large_model
        assert "ignored_symbol_0" not in names_large_model

        # OPTIMIZER category
        res_large_opt = hf_mod.collect_huggingface(
            "transformers", mapping_transformers, SemanticTier.OPTIMIZER
        )
        names_large_opt = {r.name for r in res_large_opt}
        assert "AutoScheduler" in names_large_opt

        # UTIL category
        res_large_util = hf_mod.collect_huggingface(
            "transformers",
            mapping_transformers,
            SemanticTier.UTIL,
            include_nonpublic=True,
        )
        names_large_util = {r.name for r in res_large_util}
        assert "pipeline_fn" in names_large_util

        # Direct calls to collect_transformers, collect_diffusers, collect_tokenizers
        res_tr = hf_mod.collect_transformers(SemanticTier.MODEL)
        assert len(res_tr) > 0

    fake_diff_lib = types.ModuleType("diffusers")
    setattr(fake_diff_lib, "DiffusionPipeline", lambda: None)

    fake_tok_lib = types.ModuleType("tokenizers")
    setattr(fake_tok_lib, "TokenizerModel", lambda: None)

    def mock_import_helpers(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import for helper functions.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Fake module.
        """
        if name == "diffusers":
            return fake_diff_lib
        if name == "tokenizers":
            return fake_tok_lib
        return real_import(name, *args, **kwargs)

    with patch("importlib.import_module", side_effect=mock_import_helpers):
        res_diff = hf_mod.collect_diffusers(SemanticTier.MODEL)
        assert len(res_diff) >= 1
        res_tok = hf_mod.collect_tokenizers(SemanticTier.MODEL)
        assert len(res_tok) >= 1


def test_tensorflow_collect_api_inspection_failures(mocker: Any) -> None:
    """Test tensorflow collect_api when GhostInspector.inspect raises exceptions across all branches.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_math = types.ModuleType("tf.math")
    setattr(fake_math, "abs", lambda: None)

    fake_linalg = types.ModuleType("tf.linalg")
    setattr(fake_linalg, "matmul", lambda: None)

    fake_raw_ops = types.ModuleType("tf.raw_ops")
    setattr(fake_raw_ops, "raw_add", lambda: None)

    class FakeTensor:
        """Fake Tensor class."""

        def get_shape(self) -> None:
            """Method docstring."""
            pass

    class FakeVariable:
        """Fake Variable class."""

        def assign(self) -> None:
            """Method docstring."""
            pass

    fake_tf = types.ModuleType("tf")
    setattr(fake_tf, "math", fake_math)
    setattr(fake_tf, "linalg", fake_linalg)
    setattr(fake_tf, "raw_ops", fake_raw_ops)
    setattr(fake_tf, "concat", lambda: None)
    setattr(fake_tf, "custom_tensor_op", lambda: None)
    setattr(fake_tf, "Tensor", FakeTensor)
    setattr(fake_tf, "Variable", FakeVariable)

    mocker.patch.object(tf_fw, "tf", fake_tf)
    mocker.patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=Exception("inspection failure"),
    )

    res = tf_fw.collect_api(SemanticTier.ARRAY_API)
    assert res == []


def test_jax_import_reload_branches() -> None:
    """Test jax module reload when jax is successfully imported and when it fails.

    Returns:
        None.
    """
    fake_jax_mod = types.ModuleType("jax")
    with patch.dict("sys.modules", {"jax": fake_jax_mod}):
        importlib.reload(jax_fw)
        assert jax_fw.jax is fake_jax_mod

    with patch.dict("sys.modules", {"jax": None}):
        importlib.reload(jax_fw)
        assert jax_fw.jax is None


def test_jax_sharding_and_pallas_collection(mocker: Any) -> None:
    """Test collection of jax.sharding and jax.experimental.pallas symbols.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """

    class Mesh:
        """Mock Mesh class."""

        pass

    def shard_map() -> None:
        """Mock shard_map function."""
        pass

    def _priv_sharding() -> None:
        """Private sharding function."""
        pass

    fake_sharding = types.ModuleType("jax.sharding")
    setattr(fake_sharding, "Mesh", Mesh)
    setattr(fake_sharding, "shard_map", shard_map)
    setattr(fake_sharding, "_priv_sharding", _priv_sharding)
    setattr(fake_sharding, "not_callable", 42)

    def pallas_call() -> None:
        """Mock pallas_call kernel."""
        pass

    def _priv_pallas() -> None:
        """Private pallas kernel."""
        pass

    fake_pallas = types.ModuleType("jax.experimental.pallas")
    setattr(fake_pallas, "pallas_call", pallas_call)
    setattr(fake_pallas, "_priv_pallas", _priv_pallas)
    setattr(fake_pallas, "not_callable", 99)

    fake_experimental = types.ModuleType("jax.experimental")
    setattr(fake_experimental, "pallas", fake_pallas)

    fake_jax = types.ModuleType("jax")
    setattr(fake_jax, "sharding", fake_sharding)
    setattr(fake_jax, "experimental", fake_experimental)
    setattr(fake_jax, "lax", types.ModuleType("jax.lax"))
    setattr(fake_jax, "random", types.ModuleType("jax.random"))
    setattr(fake_jax, "Array", type("Array", (), {}))

    fake_jnp = types.ModuleType("jax.numpy")
    fake_numpy = types.ModuleType("numpy")
    setattr(fake_numpy, "float32", float)

    mocker.patch.dict(
        "sys.modules",
        {
            "jax": fake_jax,
            "jax.sharding": fake_sharding,
            "jax.experimental": fake_experimental,
            "jax.experimental.pallas": fake_pallas,
            "jax.lax": fake_jax.lax,
            "jax.random": fake_jax.random,
            "jax.numpy": fake_jnp,
            "numpy": fake_numpy,
        },
    )

    # 1. Public symbols only
    refs = jax_fw._scan_array_api(include_nonpublic=False)
    paths = {r.api_path for r in refs}
    assert "jax.sharding.shard_map" in paths
    assert "jax.experimental.pallas.pallas_call" in paths
    assert "jax.sharding._priv_sharding" not in paths
    assert "jax.experimental.pallas._priv_pallas" not in paths

    # 2. Including non-public symbols
    refs_all = jax_fw._scan_array_api(include_nonpublic=True)
    paths_all = {r.api_path for r in refs_all}
    assert "jax.sharding._priv_sharding" in paths_all
    assert "jax.experimental.pallas._priv_pallas" in paths_all


def test_jax_array_ops_inspect_exceptions(mocker: Any) -> None:
    """Test jax array ops collection when GhostInspector.inspect raises exceptions across all branches.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_jnp = types.ModuleType("jax.numpy")
    setattr(fake_jnp, "abs", lambda: None)

    fake_lax = types.ModuleType("jax.lax")
    setattr(fake_lax, "add", lambda: None)

    fake_random = types.ModuleType("jax.random")
    setattr(fake_random, "uniform", lambda: None)

    class FakeArray:
        """Fake Array class."""

        def reshape(self) -> None:
            """Method docstring."""
            pass

    fake_sharding = types.ModuleType("jax.sharding")
    setattr(fake_sharding, "shard_map", lambda: None)

    fake_pallas = types.ModuleType("jax.experimental.pallas")
    setattr(fake_pallas, "pallas_call", lambda: None)

    fake_experimental = types.ModuleType("jax.experimental")
    setattr(fake_experimental, "pallas", fake_pallas)

    fake_jax = types.ModuleType("jax")
    setattr(fake_jax, "lax", fake_lax)
    setattr(fake_jax, "random", fake_random)
    setattr(fake_jax, "sharding", fake_sharding)
    setattr(fake_jax, "experimental", fake_experimental)
    setattr(fake_jax, "jit", lambda fn: fn)
    setattr(fake_jax, "grad", lambda fn: fn)
    setattr(fake_jax, "vmap", lambda fn: fn)
    setattr(fake_jax, "pmap", lambda fn: fn)
    setattr(fake_jax, "checkpoint", lambda fn: fn)
    setattr(fake_jax, "Array", FakeArray)

    fake_numpy = types.ModuleType("numpy")
    setattr(fake_numpy, "float32", float)

    mocker.patch.dict(
        "sys.modules",
        {
            "jax": fake_jax,
            "jax.numpy": fake_jnp,
            "jax.lax": fake_lax,
            "jax.random": fake_random,
            "jax.sharding": fake_sharding,
            "jax.experimental": fake_experimental,
            "jax.experimental.pallas": fake_pallas,
            "numpy": fake_numpy,
        },
    )

    def mock_inspect(obj: Any, name: str, **kwargs: Any) -> Any:
        """Mock inspect to simulate failure except for float32.

        Args:
            obj: Target object.
            name: Target path string.
            **kwargs: Extra arguments.

        Returns:
            GhostRef or raises Exception.
        """
        if name == "jax.numpy.float32":
            return GhostRef(name="float32", api_path="jax.numpy.float32", kind="class")
        raise Exception("inspection failure")

    mocker.patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    refs = jax_fw._scan_array_api(include_nonpublic=False)
    assert len(refs) == 1
    assert refs[0].api_path == "jax.numpy.float32"


def test_jax_pallas_import_failure(mocker: Any) -> None:
    """Test jax array ops collection when importing jax.experimental.pallas raises an exception.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_jax = types.ModuleType("jax")
    setattr(fake_jax, "lax", types.ModuleType("jax.lax"))
    setattr(fake_jax, "random", types.ModuleType("jax.random"))
    setattr(fake_jax, "Array", type("Array", (), {}))
    fake_jnp = types.ModuleType("jax.numpy")
    fake_numpy = types.ModuleType("numpy")
    setattr(fake_numpy, "float32", float)

    mocker.patch.dict(
        "sys.modules",
        {
            "jax": fake_jax,
            "jax.numpy": fake_jnp,
            "jax.lax": fake_jax.lax,
            "jax.random": fake_jax.random,
            "numpy": fake_numpy,
        },
    )

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising on pallas.

        Args:
            name: Module name.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Imported module or raises RuntimeError.
        """
        if "pallas" in name:
            raise RuntimeError("pallas not supported")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)
    refs = jax_fw.collect_api(SemanticTier.ARRAY_API)
    assert isinstance(refs, list)


def test_jax_array_ops_import_error(mocker: Any) -> None:
    """Test jax array ops collection when import jax raises ImportError.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising on jax.

        Args:
            name: Module name.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Imported module or raises ImportError.
        """
        if name == "jax":
            raise ImportError("no jax")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)
    assert jax_fw.collect_api(SemanticTier.ARRAY_API) == []
