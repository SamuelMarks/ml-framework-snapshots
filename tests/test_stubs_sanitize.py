"""Module docstring."""

from ml_framework_snapshots.stubs import _sanitize_default


def test_sanitize_default():
    """Function docstring."""
    assert _sanitize_default("<unrepresentable>") == "..."
    assert _sanitize_default("'linear'") == "'linear'"
    assert _sanitize_default("10") == "10"
    assert _sanitize_default("[1, 2, 3]") == "[1, 2, 3]"
    assert _sanitize_default("Enum.A") == "Enum.A"
    assert _sanitize_default("my_var") == "my_var"
    assert _sanitize_default("-1.0") == "-1.0"
    assert _sanitize_default("1 + 1") == "1 + 1"
    assert _sanitize_default("<class 'list'>") == "..."
    assert _sanitize_default("f()") == "..."
