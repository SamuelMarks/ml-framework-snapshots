"""Module docstring."""

from ml_framework_snapshots.models import GhostInspector


def dummy_decorator(func) -> None:
    """Function docstring.

    Args:
        func: description
    """

    def wrapper(*args, **kwargs):
        """Function docstring.

        Args:
            args: description
            kwargs: description
        """
        return func(*args, **kwargs)

    wrapper.__wrapped__ = func
    return wrapper


def tf_function_mock(func) -> None:
    """Function docstring.

    Args:
        func: description
    """

    class TFFunction:
        """Class docstring."""

        def __init__(self, f):
            """Function docstring.

            Args:
                f: description
            """
            self._python_function = f

    return TFFunction(func)


def test_unwrap_standard_decorator() -> None:
    """Function docstring."""

    @dummy_decorator
    def my_func(a: int, b: str = "test"):
        """My doc.

        Args:
            a: description
            b: description
        """
        pass

    ref = GhostInspector.inspect(my_func, "my_func")
    assert ref.has_arg("a")
    assert ref.has_arg("b")
    assert ref.params[0].annotation == "int"


def test_unwrap_tf_decorator() -> None:
    """Function docstring."""

    @tf_function_mock
    def my_tf_func(tensor, training=False):
        """Function docstring.

        Args:
            tensor: description
            training: description
        """
        pass

    ref = GhostInspector.inspect(my_tf_func, "my_tf_func")
    assert ref.has_arg("tensor")
    assert ref.has_arg("training")


def test_unwrap_generic_decorator() -> None:
    """Function docstring."""

    def generic_dec(func):
        """Function docstring.

        Args:
            func: description
        """

        class Wrapper:
            """Class docstring."""

            pass

        w = Wrapper()
        setattr(w, "__original_fn", func)
        return w

    @generic_dec
    def fn1(a):
        """Function docstring.

        Args:
            a: description
        """
        pass

    ref = GhostInspector.inspect(fn1, "fn1")
    assert ref.has_arg("a")


def test_unwrap_variant_decorator() -> None:
    """Function docstring."""

    def variant_dec(func):
        """Function docstring.

        Args:
            func: description
        """

        class Wrapper:
            """Class docstring."""

            pass

        w = Wrapper()
        w._original_fn = func
        return w

    @variant_dec
    def fn2(b):
        """Function docstring.

        Args:
            b: description
        """
        pass

    ref = GhostInspector.inspect(fn2, "fn2")
    assert ref.has_arg("b")
