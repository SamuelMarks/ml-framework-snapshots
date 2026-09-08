"""Module docstring."""

from typing import Any


from ml_framework_snapshots.utils import get_all_members, extract_c_extension_signature


class FaultyModule:
    """Class docstring."""

    __all__ = ["good", "bad"]
    good = 1

    @property
    def bad(self) -> Any:
        """Function docstring.

        Raises:
            ValueError: Exception.
        """
        raise ValueError("Cannot access")


class NoAllModule:
    """Class docstring."""

    a = 1


def test_get_all_members_exception() -> None:
    """Function docstring."""
    members = get_all_members(FaultyModule())
    assert dict(members)["good"] == 1
    assert "bad" not in dict(members)

    members2 = get_all_members(NoAllModule())
    assert dict(members2)["a"] == 1


def test_extract_c_ext_coverage() -> None:
    """Function docstring."""

    def empty_doc() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    empty_doc.__doc__ = ""
    assert extract_c_extension_signature(empty_doc, "empty_doc") is None

    def bad_syntax() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    bad_syntax.__doc__ = "func(a b c) -> int"
    assert extract_c_extension_signature(bad_syntax, "bad_syntax") is None

    def no_match() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    no_match.__doc__ = "This is just a description without signature."
    assert extract_c_extension_signature(no_match, "no_match") is None

    def string_def() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    string_def.__doc__ = "func(a='hello', b=None, *args: int, c=1, **kwargs: float) -> str\n\nDescription text."
    sig = extract_c_extension_signature(string_def, "func")
    assert sig is not None
    names = [s[0] for s in sig]
    assert "a" in names
    assert "args" in names
    assert "c" in names
    assert "kwargs" in names
    assert sig[0][2] == "'hello'"  # a='hello'
    assert sig[1][2] == "None"  # b=None

    def func_with_self() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    func_with_self.__doc__ = "func(self, x=torch.float32)"
    sig2 = extract_c_extension_signature(func_with_self, "func")
    assert sig2 is not None
    assert sig2[0][0] == "x"
    assert sig2[0][2] == "torch.float32"  # value error fallback

    def leading_empty() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    leading_empty.__doc__ = (
        "\n\nOverloaded function.\n\n1. func(x: int) -> int\n2. func(y: str) -> str"
    )
    sig_leading = extract_c_extension_signature(leading_empty, "func")
    assert sig_leading is not None
    assert len(sig_leading.overloads) == 1

    def numbered_with_empty() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    numbered_with_empty.__doc__ = (
        "func(x: int) -> int\n\n1. func(x: int) -> int\n2. func(y: str) -> str"
    )
    sig_num = extract_c_extension_signature(numbered_with_empty, "func")
    assert sig_num is not None
    assert len(sig_num.overloads) == 2

    def sig_then_overload_header() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    sig_then_overload_header.__doc__ = "func(x: int) -> int\nOverloaded function.\n1. func(x: int) -> int\n2. func(y: str) -> str"
    sig_so = extract_c_extension_signature(sig_then_overload_header, "func")
    assert sig_so is not None
    assert len(sig_so.overloads) == 2

    def sig_then_doc_text() -> Any:
        """Function docstring."""
        pass  # pragma: no cover

    sig_then_doc_text.__doc__ = (
        "func(x: int) -> int\nSome description without overload header."
    )
    sig_dt = extract_c_extension_signature(sig_then_doc_text, "func")
    assert sig_dt is not None
    assert len(sig_dt.overloads) == 0


def test_utils_branches() -> None:
    """Function docstring."""
    from ml_framework_snapshots.utils import (
        extract_c_extension_signature,
    )

    class Dummy:
        """Class docstring."""

        pass

    Dummy.__doc__ = "\n"
    assert extract_c_extension_signature(Dummy, "X") is None


def test_normalize_c_sig_args_and_unparse_branches() -> None:
    """Test branches in _normalize_c_sig_args, _parse_c_extension_sig_str, and docstring usage scanning."""
    from ml_framework_snapshots.utils import (
        _normalize_c_sig_args,
        _parse_c_extension_sig_str,
    )

    # Test with trailing comma, keyword-only marker '*', slash '/', varargs, varkwargs, colon, C++ type with default, and plain arg
    raw = "Tensor a, *, /, *args, **kwargs, c: int, const Tensor& d = None, plain_arg, "
    normalized = _normalize_c_sig_args(raw)
    assert 'a: "Tensor"' in normalized
    assert "*" in normalized
    assert "/" in normalized
    assert "*args" in normalized
    assert "**kwargs" in normalized
    assert "c: int" in normalized
    assert 'd: "Tensor" = None' in normalized
    assert "plain_arg" in normalized

    # Test unparse_anno when node is None (unannotated parameter)
    parsed = _parse_c_extension_sig_str("func(unannotated_arg)")
    assert parsed is not None
    assert parsed[1][0][3] is None  # annotation is None

    # Test C++ style signature where first ast.parse raises SyntaxError and normalized parse succeeds (lines 162-163)
    parsed_cpp = _parse_c_extension_sig_str(
        "func(Tensor self, bool inplace=False) -> Tensor"
    )
    assert parsed_cpp is not None
    assert parsed_cpp[1][0][0] == "inplace"

    # Test invalid syntax after normalization to hit inner SyntaxError handler
    assert _parse_c_extension_sig_str("func(a b c)") is None

    # Test _normalize_c_sig_args with empty string and bracket ending
    assert _normalize_c_sig_args("") == ""
    assert _normalize_c_sig_args("std::vector<int>") == "std::vector<int>"

    # Test fallback scanning lines where candidate parse fails first, then succeeds
    def usage_func() -> Any:
        pass

    usage_func.__doc__ = (
        "Some description\n"
        ">>> usage_func(invalid syntax arg)\n"
        ">>> usage_func(valid_arg: int) -> int\n"
    )
    sig_usage = extract_c_extension_signature(usage_func, "usage_func")
    assert sig_usage is not None
    assert sig_usage[0][0] == "valid_arg"
