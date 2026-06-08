"""Module docstring."""

from unittest.mock import patch, MagicMock
from ml_switcheroo_ir.schema.ghost import SemanticTier


# --- DIFF ---
def test_diff_branches():
    """Function docstring."""
    from ml_framework_snapshots.diff import diff_snapshots

    # Hit line 81->87 (_compare_params empty)
    s1 = {
        "categories": {
            "MODEL": [{"name": "A", "api_path": "A", "kind": "function", "params": []}]
        }
    }
    s2 = {
        "categories": {
            "MODEL": [{"name": "A", "api_path": "A", "kind": "function", "params": []}]
        }
    }
    diff_snapshots(s1, s2)

    # Hit branch where kind changed but it's not breaking (e.g. POSITIONAL_ONLY -> POSITIONAL_OR_KEYWORD)
    s3 = {
        "categories": {
            "MODEL": [
                {
                    "name": "B",
                    "api_path": "B",
                    "kind": "function",
                    "params": [{"name": "p", "kind": "POSITIONAL_ONLY"}],
                }
            ]
        }
    }
    s4 = {
        "categories": {
            "MODEL": [
                {
                    "name": "B",
                    "api_path": "B",
                    "kind": "function",
                    "params": [{"name": "p", "kind": "POSITIONAL_OR_KEYWORD"}],
                }
            ]
        }
    }
    diff_snapshots(s3, s4)


# --- EXPORT ---
def test_export_branches():
    """Function docstring."""
    from ml_framework_snapshots.export import (
        _py_type_to_proto,
        to_pydantic,
        to_json_schema,
        _ghost_to_cdd_ir,
    )

    assert _py_type_to_proto(None) == "string"
    assert _py_type_to_proto("") == "string"

    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    r = GhostRef(
        name="X",
        api_path="X",
        kind="function",
        params=[
            GhostParam(
                name="p1", kind="KEYWORD_ONLY", default="None", annotation="int"
            ),
            GhostParam(
                name="p2", kind="VAR_POSITIONAL", default="None", annotation="int"
            ),
        ],
    )
    to_pydantic(r)
    to_json_schema(r)

    r_empty_anno = GhostRef(
        name="Y",
        api_path="Y",
        kind="function",
        params=[GhostParam(name="p_empty", kind="KEYWORD_ONLY")],
        returns_type="str",
    )  # No returns_description
    _ghost_to_cdd_ir(r_empty_anno)

    r_ret_desc = GhostRef(
        name="Z", api_path="Z", kind="function", params=[], returns_description="desc"
    )
    _ghost_to_cdd_ir(r_ret_desc)


# --- FRAMEWORKS ---


def test_deepspeed_missing():
    """Function docstring."""
    from ml_framework_snapshots.frameworks.deepspeed import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class DeepspeedMock:
        """Class docstring."""

        def __dir__(self):
            """Function docstring."""
            return ["_hidden", "initialize", "nothing"]

        def initialize(self):
            """Function docstring."""
            pass

        def nothing(self):
            """Function docstring."""
            pass

    with patch("importlib.import_module", return_value=DeepspeedMock()):
        with patch(
            "ml_framework_snapshots.models.GhostInspector.inspect",
            side_effect=lambda obj, name: (
                None
                if "nothing" in name
                else MagicMock(params=[MagicMock(**{"name": "config_params"})])
            ),
        ):
            collect_api(SemanticTier.MODEL)
            collect_api(SemanticTier.MODEL, include_nonpublic=True)

    with patch("importlib.import_module", return_value=DeepspeedMock()):
        with patch(
            "ml_framework_snapshots.models.GhostInspector.inspect",
            return_value=MagicMock(params=[]),
        ):
            collect_api(SemanticTier.MODEL)


def test_onnxruntime_missing():
    """Function docstring."""
    from ml_framework_snapshots.frameworks.onnxruntime import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class OnnxMock:
        """Class docstring."""

        def __dir__(self):
            """Function docstring."""
            return ["_hidden", "InferenceSession", "nothing"]

        def InferenceSession(self):
            """Function docstring."""
            pass

        def nothing(self):
            """Function docstring."""
            pass

    with patch("importlib.import_module", return_value=OnnxMock()):
        with patch(
            "ml_framework_snapshots.models.GhostInspector.inspect",
            side_effect=lambda obj, name: (
                None
                if "nothing" in name
                else MagicMock(params=[MagicMock(**{"name": "path_or_bytes"})])
            ),
        ):
            collect_api(SemanticTier.MODEL)
            collect_api(SemanticTier.MODEL, include_nonpublic=True)

    with patch("importlib.import_module", return_value=OnnxMock()):
        with patch(
            "ml_framework_snapshots.models.GhostInspector.inspect",
            return_value=MagicMock(params=[]),
        ):
            collect_api(SemanticTier.MODEL)


def test_triton_missing():
    """Function docstring."""
    from ml_framework_snapshots.frameworks.triton import collect_api

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_api(SemanticTier.UTIL) == []
    assert collect_api(SemanticTier.LOSS) == []

    mock_mod = MagicMock()
    with patch.dict("sys.modules", {"triton": mock_mod}):
        collect_api(SemanticTier.UTIL)

    mock_mod.__dir__ = lambda self: ["_hidden", "fn", "not_ref"]

    def fake_fn(a, b):
        """Function docstring.

        Args:
            a: description
            b: description
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
        "importlib.import_module",
        side_effect=lambda x: (
            mock_mod if x == "triton.language" or x == "triton" else ImportError("No")
        ),
    ):
        with patch(
            "ml_framework_snapshots.models.GhostInspector.inspect",
            side_effect=lambda obj, name: (
                None if "not_ref" in name else MagicMock(params=[param_a, param_b])
            ),
        ):
            collect_api(SemanticTier.UTIL)


def test_sklearn_missing():
    """Function docstring."""
    from ml_framework_snapshots.frameworks.sklearn import collect_api

    with patch(
        "ml_framework_snapshots.frameworks.sklearn.get_all_members",
        side_effect=Exception,
    ):
        res = collect_api(SemanticTier.LAYER)
        assert res == []


def test_huggingface_missing():
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

        def generate(self, input, *args, **kwargs):
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
        def bad(self):
            """Function docstring."""
            raise ValueError()

    mock_mod = MagicMock()
    mock_mod.__dir__ = lambda self: [
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

        def __dir__(self):
            """Function docstring."""
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
        def bad(self):
            """Function docstring."""
            raise ValueError()

        @property
        def DummyConfig(self):
            """Function docstring."""
            return type("DummyConfig", (), {"__annotations__": {}})()

        @property
        def EmptyConfig(self):
            """Function docstring."""
            return type("EmptyConfig", (), {"__annotations__": {"a": int}})()

        @property
        def EmptyConfigB(self):
            """Function docstring."""
            return type("EmptyConfigB", (), {"__annotations__": {"b": int}})()

        @property
        def AutoModelA(self):
            """Function docstring."""
            return MagicMock()

        @property
        def AutoModelB(self):
            """Function docstring."""
            return MagicMock()

        @property
        def Other(self):
            """Function docstring."""
            return MagicMock()

        @property
        def ThrowConfig(self):
            """Function docstring."""
            return MagicMock()

        @property
        def ReturnNone(self):
            """Function docstring."""
            return None

        @property
        def GenModel(self):
            """Function docstring."""

            class M:
                """Class docstring."""

                def generate(self, input, *args, **kwargs):
                    """Function docstring.

                    Args:
                        input: description
                        args: description
                        kwargs: description
                    """
                    pass

            return M()

        @property
        def RefNone(self):
            """Function docstring."""
            return MagicMock()

    def mock_inspect(obj, name):
        """Function docstring.

        Args:
            obj: description
            name: description
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

    with patch("importlib.import_module", return_value=FakeMod()):
        with patch(
            "ml_framework_snapshots.models.GhostInspector.inspect",
            side_effect=mock_inspect,
        ):
            collect_transformers(SemanticTier.MODEL)
            collect_transformers(SemanticTier.UTIL)
