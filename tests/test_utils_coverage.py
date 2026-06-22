"""Module docstring."""

from typing import Any


from ml_framework_snapshots.utils import get_all_members, extract_c_extension_signature


class FaultyModule:
    """Class docstring."""

    __all__ = ["good", "bad"]
    good = 1

    @property
    def bad(self) -> Any:
        """Function docstring."""
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
        pass

    empty_doc.__doc__ = ""
    assert extract_c_extension_signature(empty_doc, "empty_doc") is None

    def bad_syntax() -> Any:
        """Function docstring."""
        pass

    bad_syntax.__doc__ = "func(a b c) -> int"
    assert extract_c_extension_signature(bad_syntax, "bad_syntax") is None

    def no_match() -> Any:
        """Function docstring."""
        pass

    no_match.__doc__ = "This is just a description without signature."
    assert extract_c_extension_signature(no_match, "no_match") is None

    def string_def() -> Any:
        """Function docstring."""
        pass

    string_def.__doc__ = (
        "func(a='hello', b=None, *args: int, c=1, **kwargs: float) -> str"
    )
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
        pass

    func_with_self.__doc__ = "func(self, x=torch.float32)"
    sig2 = extract_c_extension_signature(func_with_self, "func")
    assert sig2 is not None
    assert sig2[0][0] == "x"
    assert sig2[0][2] == "torch.float32"  # value error fallback
