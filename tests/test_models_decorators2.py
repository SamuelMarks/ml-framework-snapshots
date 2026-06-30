"""Module docstring."""

from typing import Any


from ml_framework_snapshots.models import GhostInspector


def torch_compile_mock(func: Any) -> None:
    """Function docstring.

    Args:
        func: description


    Returns:
        Return value.
    """

    class Compiled:
        """Class docstring."""

        def __init__(self, f: Any) -> Any:  # type: ignore
            """Function docstring.

            Args:
                f: description
            """
            self._orig_mod = f

    return Compiled(func)  # type: ignore


def test_unwrap_torch_compile() -> None:
    """Function docstring."""

    @torch_compile_mock
    def my_compiled(x: Any, y=10) -> Any:  # type: ignore
        """Function docstring.

        Args:
            x: description
            y: description
        """
        pass

    ref = GhostInspector.inspect(my_compiled, "my_compiled")
    assert ref.has_arg("x")
    assert ref.has_arg("y")
