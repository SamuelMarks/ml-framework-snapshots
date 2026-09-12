"""Module docstring."""

from typing import Any


from ml_framework_snapshots.models import GhostInspector


def some_function(a: int = 1, b: str = "test", *args: Any, **kwargs: Any) -> None:
    """A test function.

    Args:
        a: The first parameter.
        b: The second parameter.
        args: description
        kwargs: description
    """
    pass


class SomeClass:
    """A test class."""

    def __init__(self, x: float, y: list = []) -> Any:  # type: ignore
        """:param x: The x coordinate.

        :type x: float
        :param y: The y list.
        :type y: list

        Args:
            x: Parameter.
            y: Parameter.
        """
        pass


def test_ghost_inspector_cdd_function() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(some_function, "some_function")
    assert ref.name == "some_function"
    assert ref.kind == "function"
    assert len(ref.params) == 4  # a, b, args, kwargs

    a_param = next(p for p in ref.params if p.name == "a")
    assert a_param.annotation == "int"
    assert a_param.default == "1"
    assert a_param.description == "The first parameter."

    b_param = next(p for p in ref.params if p.name == "b")
    assert b_param.annotation == "str"
    assert b_param.default == "'test'"
    assert b_param.description == "The second parameter."


def test_ghost_inspector_cdd_class() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(SomeClass, "SomeClass")
    assert ref.name == "SomeClass"
    assert ref.kind == "class"

    x_param = next(p for p in ref.params if p.name == "x")
    assert x_param.annotation == "float"
    assert x_param.description == "The x coordinate."

    y_param = next(p for p in ref.params if p.name == "y")
    assert y_param.annotation == "list"
    assert y_param.default == "[]"
    assert y_param.description == "The y list."
