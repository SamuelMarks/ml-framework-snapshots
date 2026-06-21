from typing import Any
"""Module docstring."""

from ml_framework_snapshots.models import sanitize_type_str


def test_sanitize_type_str() -> None:
    """Function docstring."""
    assert sanitize_type_str(None) is None
    assert sanitize_type_str("<class 'list'>") == "list"
    assert sanitize_type_str("typing.List[int]") == "list[int]"
    assert (
        sanitize_type_str("typing.Dict[str, typing.Tuple[int, int]]")
        == "dict[str, tuple[int, int]]"
    )
    assert sanitize_type_str("Union[list, dict]") == "list | dict"
    assert sanitize_type_str("typing.Union[str, int]") == "str | int"
    assert sanitize_type_str("Optional[int]") == "int | None"
    assert sanitize_type_str("typing.Optional[float]") == "float | None"
    assert sanitize_type_str("typing.Set[int]") == "set[int]"
    assert sanitize_type_str("Type[int]") == "type[int]"
    assert sanitize_type_str("int") == "int"
    assert sanitize_type_str("some_invalid_ast!") == "some_invalid_ast!"
    assert (
        sanitize_type_str("Union[int]") == "Union[int]"
    )  # no transformation if len < 2
    assert sanitize_type_str("typing.Any") == "Any"
