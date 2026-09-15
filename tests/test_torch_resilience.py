"""Tests for fault tolerance, fallback handling, and inspection resilience in the PyTorch framework collector."""

import importlib
import types
from typing import Any
from unittest.mock import patch

from ml_framework_snapshots.frameworks.torch import (
    _scan_array_api,
    _scan_metrics,
    get_aten_op_schema,
    get_jit_schemas_for_op,
)
from ml_framework_snapshots.models import GhostInspector, GhostPythonRef


def test_torch_module_import_reload() -> None:
    """Test torch module reload handling when importing torch.nn/optim/data fails.

    Returns:
        None.
    """
    from ml_framework_snapshots.frameworks import torch as torch_fw

    real_import = __import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on torch submodules.

        Args:
            name: Module name.
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Raises:
            ImportError: Simulated import error for torch submodules.

        Returns:
            Imported module.
        """
        if name in ("torch.nn", "torch.optim", "torch.utils.data"):
            raise ImportError(f"simulated {name} import failure")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(torch_fw)
        assert torch_fw.nn is None
        assert torch_fw.optim is None
        assert torch_fw.data is None

    # Restore clean reload
    importlib.reload(torch_fw)
    if getattr(torch_fw, "torch", None) is not None:
        assert torch_fw.nn is not None
    else:
        assert torch_fw.nn is None


def test_torch_scan_metrics_inspect_exception(mocker: Any) -> None:
    """Test _scan_metrics catches exceptions during GhostInspector.inspect.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """

    class DummyMetric:
        """Dummy metric class."""

        pass

    fake_metrics = types.ModuleType("torchmetrics")
    setattr(fake_metrics, "DummyMetric", DummyMetric)

    with patch.dict("sys.modules", {"torchmetrics": fake_metrics}):
        mocker.patch.object(
            GhostInspector,
            "inspect",
            side_effect=RuntimeError("simulated inspect failure"),
        )
        refs = _scan_metrics(include_nonpublic=False)
        assert refs == []


def test_torch_scan_array_api_inspect_exceptions(mocker: Any) -> None:
    """Test _scan_array_api catches exceptions across all submodules and ops.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_torch = types.ModuleType("torch")

    # 1. Top-level callable
    setattr(fake_torch, "dummy_op", lambda x: x)

    # 2. linalg callable
    linalg_mod = types.ModuleType("torch.linalg")
    setattr(linalg_mod, "norm", lambda x: x)
    setattr(fake_torch, "linalg", linalg_mod)

    # 3. special callable
    special_mod = types.ModuleType("torch.special")
    setattr(special_mod, "erf", lambda x: x)
    setattr(fake_torch, "special", special_mod)

    # 4. fft callable
    fft_mod = types.ModuleType("torch.fft")
    setattr(fft_mod, "fft", lambda x: x)
    setattr(fake_torch, "fft", fft_mod)

    # 5. nn.functional callable
    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")
    setattr(functional_mod, "relu", lambda x: x)
    setattr(nn_mod, "functional", functional_mod)
    setattr(fake_torch, "nn", nn_mod)

    # 6. autograd callable
    autograd_mod = types.ModuleType("torch.autograd")
    setattr(autograd_mod, "grad", lambda x: x)
    setattr(fake_torch, "autograd", autograd_mod)

    # 7. distributed callable
    dist_mod = types.ModuleType("torch.distributed")
    setattr(dist_mod, "all_reduce", lambda x: x)
    setattr(fake_torch, "distributed", dist_mod)

    # 8. Tensor instance method
    class MockTensor:
        """Mock Tensor class."""

        def add_(self, other: Any) -> Any:
            """In-place add.

            Args:
                other: Other tensor.

            Returns:
                Self.
            """
            return self

    setattr(fake_torch, "Tensor", MockTensor)

    # 9. torch.ops.aten callable
    class MockOps:
        """Mock ops container."""

        class MockAten:
            """Mock aten namespace."""

            add = staticmethod(lambda x, y: x)

        aten = MockAten()

    setattr(fake_torch, "ops", MockOps())

    with patch.dict("sys.modules", {"torch": fake_torch}):
        mocker.patch.object(
            GhostInspector,
            "inspect",
            side_effect=ValueError("simulated inspection failure"),
        )
        refs = _scan_array_api(include_nonpublic=True)
        assert refs == []


def test_torch_scan_array_api_tensor_inplace_tags_none(mocker: Any) -> None:
    """Test _scan_array_api populates environment_tags when initial tags are None.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_torch = types.ModuleType("torch")

    class MockTensor:
        """Mock Tensor class."""

        def add_(self, other: Any) -> Any:
            """In-place add.

            Args:
                other: Other tensor.

            Returns:
                Self.
            """
            return self

    setattr(fake_torch, "Tensor", MockTensor)

    mock_ref = GhostPythonRef(
        name="add_",
        api_path="torch.Tensor.add_",
        kind="method",
        domain_metadata=None,
        environment_tags=None,
    )

    with patch.dict("sys.modules", {"torch": fake_torch}):
        mocker.patch.object(
            GhostInspector,
            "inspect",
            return_value=mock_ref,
        )
        refs = _scan_array_api(include_nonpublic=False)
        assert len(refs) == 1
        assert isinstance(refs[0], GhostPythonRef)
        assert refs[0].domain_metadata == {"is_in_place": True}
        assert refs[0].environment_tags == ["in_place_mutation"]


def test_get_jit_schemas_for_op_import_exception() -> None:
    """Test get_jit_schemas_for_op handles exception during torch import.

    Returns:
        None.
    """
    with patch.dict("sys.modules", {"torch": None}):
        res = get_jit_schemas_for_op("add")
        assert res == []


def test_get_aten_op_schema_import_error() -> None:
    """Test get_aten_op_schema handles ImportError on torch import.

    Returns:
        None.
    """
    with patch.dict("sys.modules", {"torch": None}):
        res = get_aten_op_schema("add")
        assert res is None
