"""Tests for cupy and dask framework collectors."""

from typing import Any
import types

from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_framework_snapshots.models import GhostInspector
import ml_framework_snapshots.frameworks.cupy as cupy_fw
import ml_framework_snapshots.frameworks.dask as dask_fw


def create_module(name: str, attrs: dict[str, Any]) -> types.ModuleType:
    """Create a mock module with given attributes.

    Args:
        name: Parameter.
        attrs: Parameter.
    """
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_cupy_collect(mocker: Any) -> None:
    """Test cupy_fw.collect_api function.

    Args:
        mocker: Parameter.
    """
    # Test when cp is None
    mocker.patch("ml_framework_snapshots.frameworks.cupy.cp", None)
    assert cupy_fw.collect_api(SemanticTier.ACTIVATION) == []

    # Mock cp
    def fake_tanh() -> None:
        """Fake tanh function."""
        pass  # pragma: no cover

    def fake_exp() -> None:
        """Fake exp function."""
        pass  # pragma: no cover

    def fake_private() -> None:
        """Fake private function."""
        pass  # pragma: no cover

    fake_cp = create_module(
        "cupy",
        {
            "tanh": fake_tanh,
            "exp": fake_exp,
            "maximum": 42,
            "_private": fake_private,
            "not_callable": 42,
            "error_func": lambda: (
                None
            ),  # We'll mock GhostInspector.inspect to raise exception for this
        },
    )
    mocker.patch("ml_framework_snapshots.frameworks.cupy.cp", fake_cp)

    # Test ACTIVATION
    orig_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, fqn: str, is_public: bool) -> Any:
        """Mock inspect to simulate an error.

        Args:
            obj: Parameter.
            fqn: Parameter.
            is_public: Parameter.
        """
        if fqn == "cupy.exp":
            raise ValueError("mock error")
        return orig_inspect(obj, fqn, is_public=is_public)

    mocker.patch(
        "ml_framework_snapshots.frameworks.cupy.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = cupy_fw.collect_api(SemanticTier.ACTIVATION)
    assert len(res) == 1  # tanh only

    # Test ARRAY_API
    res = cupy_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert any(r.api_path == "cupy.tanh" for r in res)
    assert not any(r.api_path == "cupy._private" for r in res)

    res = cupy_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert any(r.api_path == "cupy._private" for r in res)

    # Test other tier
    assert cupy_fw.collect_api(SemanticTier.OPTIMIZER) == []


def test_dask_collect(mocker: Any) -> None:
    """Test dask_fw.collect_api function.

    Args:
        mocker: Parameter.
    """
    # Test when da is None
    mocker.patch("ml_framework_snapshots.frameworks.dask.da", None)
    assert dask_fw.collect_api(SemanticTier.ARRAY_API) == []

    def fake_func() -> None:
        """Fake function."""
        pass  # pragma: no cover

    def fake_private() -> None:
        """Fake private function."""
        pass  # pragma: no cover

    fake_da = create_module(
        "dask.array",
        {
            "func": fake_func,
            "_private": fake_private,
            "not_callable": 42,
        },
    )
    mocker.patch("ml_framework_snapshots.frameworks.dask.da", fake_da)

    orig_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, fqn: str, is_public: bool) -> Any:
        """Mock inspect to simulate an error.

        Args:
            obj: Parameter.
            fqn: Parameter.
            is_public: Parameter.
        """
        if fqn == "dask.array._private":
            raise ValueError("mock error")
        return orig_inspect(obj, fqn, is_public=is_public)

    mocker.patch(
        "ml_framework_snapshots.frameworks.dask.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = dask_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert any(r.api_path == "dask.array.func" for r in res)

    res = dask_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    # _private raises error, so it won't be collected
    assert not any(r.api_path == "dask.array._private" for r in res)

    # Test other tier
    assert dask_fw.collect_api(SemanticTier.OPTIMIZER) == []


def test_cupy_import_success(mocker: Any) -> None:
    """Test cupy import logic when module is available.

    Args:
        mocker: Parameter.
    """
    import sys

    # Mock cupy as a valid module
    fake_cupy = types.ModuleType("cupy")
    mocker.patch.dict(sys.modules, {"cupy": fake_cupy})

    # Reload framework to trigger the try block successfully
    if "ml_framework_snapshots.frameworks.cupy" in sys.modules:  # pragma: no branch
        del sys.modules["ml_framework_snapshots.frameworks.cupy"]

    import ml_framework_snapshots.frameworks.cupy as c_fw

    assert c_fw.cp is fake_cupy


def test_cupy_import_error(mocker: Any) -> None:
    """Test cupy import logic when module is not available.

    Args:
        mocker: Parameter.
    """
    import sys

    mocker.patch.dict(sys.modules, {"cupy": None})
    if "ml_framework_snapshots.frameworks.cupy" in sys.modules:  # pragma: no branch
        del sys.modules["ml_framework_snapshots.frameworks.cupy"]

    import ml_framework_snapshots.frameworks.cupy as c_fw

    assert c_fw.cp is None


def test_dask_import_error(mocker: Any) -> None:
    """Test dask import logic when module is not available.

    Args:
        mocker: Parameter.
    """
    import sys

    mocker.patch.dict(sys.modules, {"dask.array": None, "dask": None})
    if "ml_framework_snapshots.frameworks.dask" in sys.modules:  # pragma: no branch
        del sys.modules["ml_framework_snapshots.frameworks.dask"]

    import ml_framework_snapshots.frameworks.dask as d_fw

    assert d_fw.da is None
