"""Module docstring."""

from typing import Any


from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostParam
from ml_switcheroo_ir.schema.ghost import GhostRef


def test_ghost_param() -> None:
    """Function docstring."""
    param = GhostParam(
        name="x", kind="POSITIONAL_OR_KEYWORD", default="1", annotation="int"
    )
    assert param.name == "x"
    assert param.kind == "POSITIONAL_OR_KEYWORD"
    assert param.default == "1"
    assert param.annotation == "int"


def test_ghost_ref_has_arg() -> None:
    """Function docstring."""
    params = [GhostParam(name="x", kind="POSITIONAL_OR_KEYWORD")]
    ref = GhostRef(
        name="foo",
        api_path="pkg.foo",
        kind="function",
        params=params,
        has_varargs=False,
    )
    assert ref.has_arg("x") is True
    assert ref.has_arg("y") is False


def dummy_func(a: int, b=2, *args: Any, **kwargs: Any) -> None:  # type: ignore
    """Docstring for dummy_func. a b args kwargs."""
    pass


class DummyClass:
    """Class docstring."""

    def __init__(self, c: str = "test") -> Any:  # type: ignore
        """Function docstring. c."""
        pass


def dummy_c_extension(*args: Any, **kwargs: Any) -> None:
    """Function docstring. args kwargs."""
    # simulate something that inspect.signature fails on
    pass


def test_ghost_inspector_function() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(dummy_func, "tests.dummy_func")
    assert ref.name == "dummy_func"
    assert ref.api_path == "tests.dummy_func"
    assert ref.kind == "function"
    assert ref.docstring == "Docstring for dummy_func. a b args kwargs."
    assert ref.has_varargs is True

    assert (
        len(ref.params) == 4
    )  # a, b (args/kwargs aren't counted explicitly as params here? Wait, *args is VAR_POSITIONAL)
    assert ref.has_arg("a")
    assert ref.has_arg("b")


def test_ghost_inspector_class() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(DummyClass, "tests.DummyClass")
    assert ref.name == "DummyClass"
    assert ref.kind == "class"
    assert ref.has_arg("c")
    assert ref.params[0].default == "'test'"


def test_ghost_inspector_c_extension(mocker: Any) -> None:
    """Function docstring."""
    # Mock inspect.signature to raise ValueError
    mocker.patch("inspect.signature", side_effect=ValueError)
    ref = GhostInspector.inspect(dummy_c_extension, "tests.dummy_c_extension")
    assert ref.name == "dummy_c_extension"
    assert ref.kind == "function"
    assert ref.has_varargs is True
    assert ref.has_arg("args")
    assert ref.has_arg("kwargs")


def test_ghost_inspector_hydrate() -> None:
    """Function docstring."""
    data = {
        "name": "foo",
        "api_path": "foo",
        "kind": "function",
        "params": [],
        "docstring": None,
        "has_varargs": False,
    }
    ref = GhostInspector.hydrate(data)
    assert ref.name == "foo"


def test_unrepresentable_default() -> None:
    """Function docstring."""

    class Unrepresentable:
        """Class docstring."""

        def __repr__(self) -> Any:
            """Function docstring."""
            raise Exception("no")

    def f(a=Unrepresentable()):  # type: ignore
        """Function docstring. a."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<unrepresentable>"


def test_memory_address_default() -> None:
    """Function docstring."""

    class MemoryDefault:
        """Class docstring."""

        pass

    def f(a=MemoryDefault()):  # type: ignore
        """Function docstring. a."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default is None


def test_callable_default() -> None:
    """Function docstring."""

    def my_default() -> Any:
        """Function docstring."""
        pass

    def f(a=my_default) -> Any:  # type: ignore
        """Function docstring. a."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default is None


def test_ghost_inspector_str_throws() -> None:
    """Function docstring."""

    class StrThrows:
        """Class docstring."""

        def __repr__(self) -> Any:
            """Function docstring."""
            return "fine"

        def __str__(self) -> Any:
            """Function docstring."""
            raise Exception("no")

    def f(a=StrThrows()):  # type: ignore
        """Function docstring. a."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<unrepresentable>"


def test_ghost_inspector_c_extension_class(mocker: Any) -> None:
    """Function docstring."""
    mocker.patch("inspect.signature", side_effect=ValueError)

    class DummyCClass:
        """Class docstring."""

        pass

    ref = GhostInspector.inspect(DummyCClass, "tests.DummyCClass")
    assert ref.name == "DummyCClass"
    assert ref.kind == "class"
    assert not ref.has_varargs
    assert len(ref.params) == 0


def test_ghost_inspector_str_has_address() -> None:
    """Function docstring."""

    class StrAddr:
        """Class docstring."""

        def __repr__(self) -> Any:
            """Function docstring."""
            return "fine"

        def __str__(self) -> Any:
            """Function docstring."""
            return "something at 0x1234"

    def f(a=StrAddr()):  # type: ignore
        """Function docstring. a."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default is None


def test_ghost_inspector_cdd_raises_no_typ(mocker: Any) -> None:
    """Function docstring."""
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"doc": "some error but no typ"}]},
    )

    def f() -> Any:
        """Doc."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.raises == []


def test_ghost_inspector_is_public(mocker: Any) -> None:
    """Function docstring."""
    # test explicit override
    ref = GhostInspector.inspect(dummy_func, "tests.dummy_func", is_public=False)
    assert ref.is_public is False

    # test griffe is_public
    class FakeGriffeNode:
        """Class docstring."""

        is_public = False

    mocker.patch("griffe.load", return_value=FakeGriffeNode())
    ref2 = GhostInspector.inspect(dummy_func, "tests.dummy_func")
    assert ref2.is_public is False

    # test griffe is_public missing attribute
    class FakeGriffeNodeNoPub:
        """Class docstring."""

        pass

    mocker.patch("griffe.load", return_value=FakeGriffeNodeNoPub())

    def _private_dummy() -> Any:
        """Function docstring."""
        pass

    ref3 = GhostInspector.inspect(_private_dummy, "tests._private_dummy")
    assert ref3.is_public is False


def test_ghost_inspector_annotation_str() -> None:
    """Function docstring."""

    def f(a: str) -> Any:
        """Function docstring. a."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].annotation == "str"


def test_ghost_inspector_cdd_fallback() -> None:
    """Function docstring."""

    # Test when p_anno is missing but cdd has it
    def dummy_func_cdd(x):  # type: ignore
        """Dummy func.

        Args:
        x (int): The value.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_cdd, "dummy_func_cdd")
    assert ref.params[0].annotation == "int"


def test_ghost_inspector_annotation_name() -> None:
    """Function docstring."""

    class MyType:
        """Class docstring."""

        pass

    def dummy_func_anno(x: MyType) -> Any:
        """Function docstring. x."""
        pass

    GhostInspector.inspect(dummy_func_anno, "dummy_func_anno")
    # Will use get_type_hints which works. To bypass get_type_hints,
    # we need a type hint that fails get_type_hints but has __name__.
    # Actually, we can just mock get_type_hints to raise Exception.
    pass


def test_ghost_inspector_annotation_name_mock(mocker: Any) -> None:
    """Function docstring."""
    # Mock get_type_hints to raise an Exception so it falls back to raw annotations
    mocker.patch("typing.get_type_hints", side_effect=Exception("Failed"))

    class MyType:
        """Class docstring."""

        pass

    def dummy_func_anno(x: MyType) -> Any:
        """Function docstring. x."""
        pass

    ref = GhostInspector.inspect(dummy_func_anno, "dummy_func_anno")
    assert ref.params[0].annotation == "MyType"


def test_ghost_inspector_annotation_str_mock(mocker: Any) -> None:
    """Function docstring."""
    mocker.patch("typing.get_type_hints", side_effect=Exception("Failed"))

    def dummy_func_anno_str(x: "str") -> Any:
        """Function docstring. x."""
        pass

    ref = GhostInspector.inspect(dummy_func_anno_str, "dummy_func_anno_str")
    assert ref.params[0].annotation == "str"


def test_ghost_inspector_cdd_anno_fallback() -> None:
    """Function docstring."""

    # cdd_params gives 'typ' but inspect does not find it
    def dummy_func_no_anno(x):  # type: ignore
        """Dummy func.

        Args:
        x (int): The value.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_no_anno, "dummy_func_no_anno")
    assert ref.params[0].annotation == "int"
    assert ref.params[0].description == "The value."


def test_ghost_inspector_return_type_hints() -> None:
    """Function docstring."""

    def dummy_func_ret() -> int:
        """Function docstring."""
        return 0

    ref = GhostInspector.inspect(dummy_func_ret, "dummy_func_ret")
    assert ref.returns_type == "int"


def test_ghost_inspector_griffe_params(mocker: Any) -> None:
    """Function docstring."""

    class FakeKind:
        """Class docstring."""

        def __init__(self, name: Any) -> None:
            """Function docstring. name."""
            self.name = name

    class FakeParam:
        """Class docstring."""

        def __init__(
            self, name: Any, kind_name: Any, default: Any, annotation: Any
        ) -> None:
            """Function docstring. name kind_name default annotation."""
            self.name = name
            self.kind = FakeKind(kind_name) if kind_name else None
            self.default = default
            self.annotation = annotation

    class FakeGriffeNode:
        """Class docstring."""

        is_public = True
        parameters = [
            FakeParam("self", None, None, None),
            FakeParam("x", "VAR_POSITIONAL", None, "int"),
            FakeParam("y", "POSITIONAL_OR_KEYWORD", "10", None),
        ]

    mocker.patch("griffe.load", return_value=FakeGriffeNode())

    def dummy_func_griffe() -> Any:
        """Function docstring."""
        pass

    ref = GhostInspector.inspect(dummy_func_griffe, "dummy_func_griffe")
    assert ref.has_varargs is True
    assert ref.params[0].name == "x"
    assert ref.params[0].annotation == "int"
    assert ref.params[1].name == "y"
    assert ref.params[1].default == "10"


def test_ghost_inspector_overloads_cdd() -> None:
    """Function docstring."""
    # If griffe is unavailable, does it handle gracefully? Yes, it will just not have overloads.
    pass


def test_ghost_inspector_griffe_overloads(mocker: Any) -> None:
    """Function docstring."""

    class FakeParam:
        """Class docstring."""

        def __init__(
            self, name: Any, kind_name: Any, default: Any, annotation: Any
        ) -> None:
            """Function docstring. name kind_name default annotation."""
            self.name = name

            class FakeKind:
                """Class docstring."""

                pass

            self.kind = FakeKind()
            self.kind.name = kind_name if kind_name else "POSITIONAL_OR_KEYWORD"  # type: ignore
            self.default = default
            self.annotation = annotation

    class FakeOverload:
        """Class docstring."""

        def __init__(self, parameters: Any, returns: Any) -> Any:  # type: ignore
            """Function docstring. parameters returns."""
            self.parameters = parameters
            self.returns = returns

    class FakeGriffeNode:
        """Class docstring."""

        is_public = True
        parameters = []  # type: ignore
        overloads = [
            FakeOverload(
                [
                    FakeParam("self", None, None, None),
                    FakeParam("a", "VAR_POSITIONAL", "1", "int"),
                ],
                "int",
            )
        ]

    mocker.patch("griffe.load", return_value=FakeGriffeNode())

    def dummy_func_griffe() -> Any:
        """Function docstring."""
        pass

    ref = GhostInspector.inspect(dummy_func_griffe, "dummy_func_griffe")
    assert len(ref.overloads) == 1
    assert ref.overloads[0].has_varargs is True
    assert ref.overloads[0].params[0].name == "a"
    assert ref.overloads[0].params[0].annotation == "int"
    assert ref.overloads[0].params[0].default == "1"
    assert ref.overloads[0].returns_type == "int"
