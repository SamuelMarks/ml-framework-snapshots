"""Tests for missing framework dependencies."""

from typing import Any
from unittest.mock import patch, MagicMock
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_deepspeed_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.deepspeed import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class DeepspeedMock:
        """Class docstring."""

        def __dir__(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return ["_hidden", "initialize", "nothing"]

        def initialize(self) -> Any:
            """Function docstring."""
            pass

        def nothing(self) -> Any:
            """Function docstring."""
            pass

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=lambda obj, name: (
            None
            if "nothing" in name
            else MagicMock(params=[MagicMock(**{"name": "config_params"})])
        ),
    ):
        with patch("importlib.import_module", return_value=DeepspeedMock()):
            collect_api(SemanticTier.MODEL)
            collect_api(SemanticTier.MODEL, include_nonpublic=True)

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        return_value=MagicMock(params=[]),
    ):
        with patch("importlib.import_module", return_value=DeepspeedMock()):
            collect_api(SemanticTier.MODEL)


def test_onnxruntime_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.onnxruntime import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class OnnxMock:
        """Class docstring."""

        def __dir__(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return ["_hidden", "InferenceSession", "nothing"]

        def InferenceSession(self) -> Any:
            """Function docstring."""
            pass

        def nothing(self) -> Any:
            """Function docstring."""
            pass

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=lambda obj, name: (
            None
            if "nothing" in name
            else MagicMock(params=[MagicMock(**{"name": "path_or_bytes"})])
        ),
    ):
        with patch("importlib.import_module", return_value=OnnxMock()):
            collect_api(SemanticTier.MODEL)
            collect_api(SemanticTier.MODEL, include_nonpublic=True)

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        return_value=MagicMock(params=[]),
    ):
        with patch("importlib.import_module", return_value=OnnxMock()):
            collect_api(SemanticTier.MODEL)


def test_triton_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.triton import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.UTIL) == []
    assert collect_api(SemanticTier.LOSS) == []

    mock_mod = MagicMock()
    with patch.dict("sys.modules", {"triton": mock_mod}):
        collect_api(SemanticTier.UTIL)

    mock_mod.__dir__ = lambda self: ["_hidden", "fn", "not_ref"]  # type: ignore

    def fake_fn(a: Any, b: Any, c: Any) -> Any:
        """Function docstring.

        Args:
            a: description
            b: description
            c: description
        """
        pass

    fake_fn.__annotations__ = {"a": "int constexpr", "b": "float"}

    class Wrap:
        """Class docstring."""

        fn = fake_fn

    mock_mod.fn = Wrap
    mock_mod.not_ref = (
        lambda: None
    )  # mock inspector will return None for this because it's not a valid ref or we can mock inspector

    param_a = MagicMock()
    param_a.name = "a"
    param_b = MagicMock()
    param_b.name = "b"

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=lambda obj, name: (
            None if "not_ref" in name else MagicMock(params=[param_a, param_b])
        ),
    ):
        with patch(
            "importlib.import_module",
            side_effect=lambda x: (
                mock_mod
                if x == "triton.language" or x == "triton"
                else ImportError("No")
            ),
        ):
            collect_api(SemanticTier.UTIL)

    from ml_framework_snapshots.frameworks.triton import _extract_triton_kernel
    from ml_framework_snapshots.models import GhostInspector

    class NoAnno:
        """Kernel callable without annotations."""

        def __call__(self, x: Any) -> Any:
            """Call method."""
            return x

    no_anno = NoAnno()
    _extract_triton_kernel(no_anno, "no_anno", "mod", GhostInspector())


def test_sklearn_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.sklearn import collect_api

    with patch(
        "ml_framework_snapshots.frameworks.sklearn.get_all_members",
        side_effect=Exception,
    ):
        res = collect_api(SemanticTier.LAYER)
        assert res == []


def test_huggingface_missing() -> None:
    """Function docstring."""
    from ml_framework_snapshots.frameworks.huggingface import (
        _extract_generation_kwargs,
        _parse_pretrained_config,
        collect_transformers,
    )
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    class M:
        """Class docstring."""

        def generate(self, input: Any, *args: Any, **kwargs: Any) -> Any:
            """Function docstring.

            Args:
                input: description
                args: description
                kwargs: description
            """
            pass

    r = GhostRef(name="A", api_path="A", kind="class", params=[])
    _extract_generation_kwargs(M(), r)

    class M2:
        """Class docstring."""

        __annotations__ = {"a": int, "b": "str"}

    r2 = GhostRef(name="A", api_path="A", kind="class", params=[])
    _parse_pretrained_config(M2(), r2)

    with patch("importlib.import_module", side_effect=ImportError):
        collect_transformers(SemanticTier.MODEL)

    collect_transformers(SemanticTier.LOSS)

    class BadProp:
        """Class docstring."""

        @property
        def bad(self) -> Any:
            """Function docstring.

            Raises:
                ValueError: Exception.
            """
            raise ValueError()

    mock_mod = MagicMock()
    mock_mod.__dir__ = lambda self: [  # type: ignore
        "_hidden",
        "DummyConfig",
        "AutoModel",
        "Other",
        "bad",
    ]
    mock_mod.DummyConfig = type("DummyConfig", (), {})()  # No __annotations__
    mock_mod.AutoModel = MagicMock()

    # Mock getattr correctly so it raises for 'bad'
    # we need to simulate the module behavior
    class FakeMod:
        """Class docstring."""

        def __dir__(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return [
                "_hidden",
                "DummyConfig",
                "AutoModelA",
                "AutoModelB",
                "Other",
                "bad",
                "ThrowConfig",
                "ReturnNone",
                "GenModel",
                "RefNone",
                "EmptyConfig",
                "EmptyConfigB",
            ]

        @property
        def bad(self) -> Any:
            """Function docstring.

            Raises:
                ValueError: Exception.
            """
            raise ValueError()

        @property
        def DummyConfig(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return type("DummyConfig", (), {"__annotations__": {}})()

        @property
        def EmptyConfig(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return type("EmptyConfig", (), {"__annotations__": {"a": int}})()

        @property
        def EmptyConfigB(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return type("EmptyConfigB", (), {"__annotations__": {"b": int}})()

        @property
        def AutoModelA(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def AutoModelB(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def Other(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def ThrowConfig(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

        @property
        def ReturnNone(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return None

        @property
        def GenModel(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """

            class M:
                """Class docstring."""

                def generate(self, input: Any, *args: Any, **kwargs: Any) -> Any:
                    """Function docstring.

                    Args:
                        input: description
                        args: description
                        kwargs: description
                    """
                    pass

            return M()

        @property
        def RefNone(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return MagicMock()

    def mock_inspect(obj: Any, name: Any) -> Any:
        """Function docstring.

        Args:
            obj: description
            name: description


        Raises:
            Exception: Exception.

        Returns:
            Return value.
        """
        if "ThrowConfig" in name:
            raise Exception("test")
        if "RefNone" in name:
            return None
        if "AutoModelA" in name:
            return MagicMock(params=[])
        if "AutoModelB" in name:
            return MagicMock(params=[MagicMock(**{"name": "config"})])
        if "GenModel" in name:
            return GhostRef(
                name="GenModel",
                api_path="GenModel",
                kind="class",
                params=[
                    GhostParam(
                        name="input",
                        kind="POSITIONAL_OR_KEYWORD",
                        default_value=None,
                        annotation="Any",
                    )
                ],
            )
        if "EmptyConfig" in name:
            return MagicMock(params=[MagicMock(**{"name": "a"})])
        return MagicMock(params=[MagicMock(**{"name": "config"})])

    with patch(
        "ml_framework_snapshots.models.GhostInspector.inspect",
        side_effect=mock_inspect,
    ):
        with patch("importlib.import_module", return_value=FakeMod()):
            collect_transformers(SemanticTier.MODEL)
            collect_transformers(SemanticTier.UTIL)


def test_hardware_exhaustive_fallbacks() -> None:
    """Test offline canonical fallbacks for all hardware/compiler targets when exhaustive files are missing."""
    import os
    from unittest.mock import patch
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    from ml_framework_snapshots.frameworks import (
        amd_rdna,
        mlir,
        nvidia_ptx,
        nvidia_sass,
        stablehlo,
    )

    orig_exists = os.path.exists

    def fake_exists(p: str) -> bool:
        """Simulate missing exhaustive JSON dataset files.

        Args:
            p: Path to check.

        Returns:
            False if target is an exhaustive JSON file, otherwise delegates to real os.path.exists.
        """
        if any(p.endswith(s) for s in ["_exhaustive.json", "_exhaustive.json.gz"]):
            return False
        return orig_exists(p)

    with patch("os.path.exists", side_effect=fake_exists):
        rdna_refs = amd_rdna.collect_api(SemanticTier.UTIL)
        assert len(rdna_refs) > 0

        mlir_refs = mlir.collect_api(SemanticTier.UTIL)
        assert len(mlir_refs) > 0

        ptx_refs = nvidia_ptx.collect_api(SemanticTier.UTIL)
        assert len(ptx_refs) > 0

        shlo_refs = stablehlo.collect_api(SemanticTier.UTIL)
        assert len(shlo_refs) > 0

        sass_refs = nvidia_sass.collect_api(SemanticTier.UTIL)
        assert len(sass_refs) > 0


def test_orbax_checkpoint_inspect_error(mocker: Any) -> None:
    """Test orbax checkpoint exception handling during member inspection and member retrieval."""
    import ml_framework_snapshots.frameworks.orbax_checkpoint as orbax_mod
    from ml_framework_snapshots.models import GhostInspector
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    def bad_fn() -> None:
        """Dummy function for raising inspection errors."""
        pass

    mock_ocp = mocker.MagicMock()
    mock_ocp.bad_fn = bad_fn
    mocker.patch.object(orbax_mod, "ocp", mock_ocp)

    # 1. Inner exception handler (lines 39-40)
    mocker.patch(
        "ml_framework_snapshots.frameworks.orbax_checkpoint.get_all_members",
        return_value=[("bad_fn", bad_fn)],
    )
    mocker.patch.object(
        GhostInspector,
        "inspect",
        side_effect=Exception("inspection failed"),
    )
    res = orbax_mod.collect_api(SemanticTier.ARRAY_API)
    assert res == []

    # 2. Outer exception handler (lines 41-42)
    mocker.patch(
        "ml_framework_snapshots.frameworks.orbax_checkpoint.get_all_members",
        side_effect=RuntimeError("member enumeration error"),
    )
    res_outer = orbax_mod.collect_api(SemanticTier.ARRAY_API)
    assert res_outer == []


def test_pax_import_coverage(monkeypatch: Any) -> None:
    """Test pax import logic when praxis is available and unavailable."""
    import importlib
    import sys
    import types
    import ml_framework_snapshots.frameworks.pax as pax_mod
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    # Available
    fake_paxml: Any = types.ModuleType("paxml")
    fake_praxis: Any = types.ModuleType("praxis")
    fake_layers: Any = types.ModuleType("praxis.layers")
    fake_praxis.layers = fake_layers
    monkeypatch.setitem(sys.modules, "paxml", fake_paxml)
    monkeypatch.setitem(sys.modules, "praxis", fake_praxis)
    monkeypatch.setitem(sys.modules, "praxis.layers", fake_layers)
    importlib.reload(pax_mod)
    assert getattr(pax_mod, "praxis") is fake_praxis

    # Unavailable
    monkeypatch.setitem(sys.modules, "paxml", None)
    importlib.reload(pax_mod)
    assert getattr(pax_mod, "praxis") is None
    assert pax_mod.collect_api(SemanticTier.MODEL) == []

    # Restore
    monkeypatch.undo()
    importlib.reload(pax_mod)


def test_sklearn_import_coverage(monkeypatch: Any) -> None:
    """Test sklearn import logic when sklearn is unavailable."""
    import importlib
    import sys
    import ml_framework_snapshots.frameworks.sklearn as sklearn_mod
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    monkeypatch.setitem(sys.modules, "sklearn", None)
    importlib.reload(sklearn_mod)
    assert getattr(sklearn_mod, "sklearn") is None
    assert sklearn_mod.collect_api(SemanticTier.MODEL) == []

    # Restore
    monkeypatch.undo()
    importlib.reload(sklearn_mod)


def test_flax_nnx_coverage(monkeypatch: Any, mocker: Any) -> None:
    """Test flax.nnx import fallback and inspection exception handling."""
    import importlib
    import sys
    import types
    import ml_framework_snapshots.frameworks.flax_nnx as nnx_mod
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    # 1. Unavailable import
    monkeypatch.setitem(sys.modules, "flax", None)
    monkeypatch.setitem(sys.modules, "flax.nnx", None)
    importlib.reload(nnx_mod)
    assert nnx_mod.nnx is None
    assert nnx_mod.collect_api(SemanticTier.LAYER) == []

    # 2. Available with issubclass TypeError
    fake_nnx: Any = types.ModuleType("flax.nnx")

    class BadClass:
        """Dummy uninheritable class to trigger TypeError in issubclass."""

        pass

    fake_nnx.BadClass = BadClass
    fake_nnx.Module = (
        123  # Non-class causes issubclass(BadClass, 123) to raise TypeError
    )
    monkeypatch.setitem(sys.modules, "flax", types.ModuleType("flax"))
    monkeypatch.setitem(sys.modules, "flax.nnx", fake_nnx)
    importlib.reload(nnx_mod)
    mocker.patch(
        "ml_framework_snapshots.frameworks.flax_nnx.get_all_members",
        return_value=[("BadClass", BadClass)],
    )
    res = nnx_mod.collect_api(SemanticTier.LAYER)
    assert res == []

    # Restore
    monkeypatch.undo()
    importlib.reload(nnx_mod)


def test_deepspeed_coverage_edges(mocker: Any) -> None:
    """Test deepspeed inspect exception handling and existing config_params."""
    import types
    import ml_framework_snapshots.frameworks.deepspeed as ds_mod
    from ml_framework_snapshots.models import GhostInspector
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, SemanticTier

    fake_ds: Any = types.ModuleType("deepspeed")
    fake_ds.initialize = lambda: None
    fake_ds.none_ref = lambda: None
    fake_ds.error_op = lambda: None
    fake_ds.z_extra_util = lambda: None
    mocker.patch(
        "ml_framework_snapshots.frameworks.deepspeed.importlib.import_module",
        return_value=fake_ds,
    )

    def mock_inspect(obj: Any, name: str) -> Any:
        """Mock inspect to test various DeepSpeed discovery branches.

        Args:
            obj: Object to inspect.
            name: Fully qualified API name.

        Returns:
            GhostRef object or raises simulated exception.
        """
        if "error_op" in name:
            raise Exception("simulated error")
        if "none_ref" in name:
            return None
        if "initialize" in name:
            return GhostRef(
                name="initialize",
                api_path="deepspeed.initialize",
                kind="function",
                params=[GhostParam(name="config_params", kind="KEYWORD_ONLY")],
            )
        return GhostRef(
            name="z_extra_util",
            api_path="deepspeed.z_extra_util",
            kind="function",
            params=[],
        )

    mocker.patch.object(
        GhostInspector,
        "inspect",
        side_effect=mock_inspect,
    )
    # MODEL category checks initialize with config_params and non-matching UTIL attrs
    res_model = ds_mod.collect_api(SemanticTier.MODEL)
    assert len(res_model) == 1

    # UTIL category triggers none_ref (branch 53 -> 36), inspect exception on error_op, and appends z_extra_util
    res_util = ds_mod.collect_api(SemanticTier.UTIL)
    assert len(res_util) == 1
    assert res_util[0].name == "z_extra_util"


def test_optax_shim_import_reload() -> None:
    """Test optax_shim module reload handling when optax is missing and present.

    Returns:
        None.
    """
    import importlib
    from ml_framework_snapshots.frameworks import optax_shim

    with patch.dict("sys.modules", {"optax": None}):
        importlib.reload(optax_shim)
        assert getattr(optax_shim, "optax") is None

    # Restore clean reload
    importlib.reload(optax_shim)
    import ml_framework_snapshots.frameworks.jax as jax_fw

    importlib.reload(jax_fw)


def test_orbax_checkpoint_import_reload() -> None:
    """Test orbax_checkpoint module reload handling when orbax.checkpoint is missing and present.

    Returns:
        None.
    """
    import importlib
    from ml_framework_snapshots.frameworks import orbax_checkpoint

    with patch.dict("sys.modules", {"orbax.checkpoint": None, "orbax": None}):
        importlib.reload(orbax_checkpoint)
        assert getattr(orbax_checkpoint, "ocp") is None

    # Restore clean reload
    importlib.reload(orbax_checkpoint)


def test_tensorflow_import_reload() -> None:
    """Test tensorflow module reload handling when tensorflow is missing and present.

    Returns:
        None.
    """
    import importlib
    from ml_framework_snapshots.frameworks import tensorflow as tf_fw

    with patch.dict("sys.modules", {"tensorflow": None}):
        importlib.reload(tf_fw)
        assert getattr(tf_fw, "tf") is None

    # Restore clean reload
    importlib.reload(tf_fw)
