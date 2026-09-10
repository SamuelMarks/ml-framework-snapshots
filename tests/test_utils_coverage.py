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
        """Function with usage docstring demonstrating signature parsing fallback."""
        pass

    usage_func.__doc__ = (
        "Some description\n"
        ">>> usage_func(invalid syntax arg)\n"
        ">>> usage_func(valid_arg: int) -> int\n"
    )
    sig_usage = extract_c_extension_signature(usage_func, "usage_func")
    assert sig_usage is not None
    assert sig_usage[0][0] == "valid_arg"


def test_tablegen_utils_coverage() -> None:
    """Test full branch coverage for TableGen list splitting and trait extraction."""
    from ml_framework_snapshots.utils import (
        extract_tablegen_traits,
        split_tablegen_list,
    )

    # Empty and simple cases
    assert split_tablegen_list("") == []
    assert split_tablegen_list("  ") == []
    assert split_tablegen_list("Pure, Commutative") == ["Pure", "Commutative"]

    # Nested brackets, angles, parens, strings with commas and escaped characters
    complex_raw = (
        'Pure, AllElementTypesMatch<["operand", "result"]>, '
        'DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>, '
        'SomeAttr<"foo, bar", (ins I32:$x)>, // line comment\n'
        'Escaped<"hello \\"world\\", test">'
    )
    res = split_tablegen_list(complex_raw)
    assert "Pure" in res
    assert 'AllElementTypesMatch<["operand", "result"]>' in res
    assert 'DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>' in res
    assert 'SomeAttr<"foo, bar", (ins I32:$x)>' in res
    assert 'Escaped<"hello \\"world\\", test">' in res

    # extract_tablegen_traits without brackets
    assert extract_tablegen_traits("No brackets here") == []

    # extract_tablegen_traits with malformed or unclosed bracket
    assert extract_tablegen_traits("[UnclosedTrait") == []

    # extract_tablegen_traits with escaped quote inside brackets and string with brackets
    escaped_bracket_text = (
        'let x = "[not_a_trait]"; let traits = [Trait<"nested \\"quote\\" [inside]">];'
    )
    assert extract_tablegen_traits(escaped_bracket_text) == [
        'Trait<"nested \\"quote\\" [inside]">'
    ]

    # extract_tablegen_traits with specialized interfaces
    sample_text = (
        'Op<"custom", [Pure, "getAsmResultNames", ["getAsmResultNames"], '
        '"inferReturnTypeComponents", ["inferReturnTypeComponents"], '
        '"inferResultRanges", ["inferResultRanges"], // ignore comment\n'
        'AllTypesMatch<["a", "b"]>]>'
    )
    traits = extract_tablegen_traits(sample_text)
    assert "Pure" in traits
    assert "OpAsmOpInterface::getAsmResultNames" in traits
    assert "InferTypeOpInterface::inferReturnTypeComponents" in traits
    assert "InferIntRangeInterface::inferResultRanges" in traits
    assert 'AllTypesMatch<["a", "b"]>' in traits

    # Branch 676 -> 678: empty elements between commas
    assert split_tablegen_list("Trait1, , Trait2, ") == ["Trait1", "Trait2"]

    # Escaped backslash and quotes before outer bracket (lines 703-704, 706-707)
    text_with_escaped_pre_bracket = (
        r'let x = "escaped \"quote\" before [not_a_trait]"; let traits = [Trait1];'
    )
    assert extract_tablegen_traits(text_with_escaped_pre_bracket) == ["Trait1"]

    # Fallback and interface loops (line 748 and branch 758 -> 745)
    from unittest.mock import patch

    with patch(
        "ml_framework_snapshots.utils.split_tablegen_list",
        return_value=["// comment", "   ", "TraitA"],
    ):
        res_mocked = extract_tablegen_traits("[dummy]")
        assert res_mocked == ["TraitA"]


def test_get_custom_snapshots_paths_deduplication(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Test deduplication and existence check in get_custom_snapshots_paths.

    Args:
        tmp_path: Pytest tmp_path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    import os
    from ml_framework_snapshots.utils import get_custom_snapshots_paths

    p1 = str(tmp_path / "snap1")
    os.makedirs(p1, exist_ok=True)
    monkeypatch.setenv(
        "ML_SNAPSHOTS_PATH",
        f"{p1}{os.pathsep}{p1}{os.pathsep}/non/existent/path",
    )
    paths = get_custom_snapshots_paths()
    assert paths == [p1]
