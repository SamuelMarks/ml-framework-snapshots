"""Module docstring."""

from ml_framework_snapshots.models import GhostInspector


def test_ghost_inspector_overloads() -> None:
    """Function docstring."""
    # We write it out to a file so Griffe can load it properly
    import pathlib

    p = pathlib.Path("test_overload_mock_unique.py")
    p.write_text("""
from typing import overload, Any

@overload
def my_overloaded_func(a: int) -> int: ...

@overload
def my_overloaded_func(a: str, b: str) -> str: ...

def my_overloaded_func(a: Any, b: Any = None) -> Any:
    pass
""")

    import sys
    from ml_framework_snapshots.models import _GRIFFE_CACHE

    _GRIFFE_CACHE.clear()

    sys.path.insert(0, ".")
    if "test_overload_mock_unique" in sys.modules:
        del sys.modules["test_overload_mock_unique"]
    import test_overload_mock_unique

    ref = GhostInspector.inspect(
        test_overload_mock_unique.my_overloaded_func,
        "test_overload_mock_unique.my_overloaded_func",
    )

    assert hasattr(ref, "overloads")
    assert len(ref.overloads) == 2
    assert ref.overloads[0].params[0].name == "a"
    assert ref.overloads[0].params[0].annotation == "int"
    assert ref.overloads[0].returns_type == "int"

    assert ref.overloads[1].params[1].name == "b"
    assert ref.overloads[1].params[1].annotation == "str"
    assert ref.overloads[1].returns_type == "str"

    p.unlink()
