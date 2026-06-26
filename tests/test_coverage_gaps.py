"""Module docstring."""

from typing import Any


from ml_framework_snapshots.models import GhostInspector


def dummy_func_with_docstring(x: int) -> None:
    """Dummy function.

    Args:
        x (int): The x value.

    Returns:
        str: A string.

    Raises:
        ValueError: If x is bad.
    """
    pass


def dummy_func_sphinx_docstring(y: Any) -> None:
    """Sphinx doc.

    :param y: The y value.
    :type y: int
    :raises TypeError: If wrong.
    """
    pass


def test_cdd_returns_and_params() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert ref.returns_description is not None
    assert ref.params[0].name == "x"
    assert ref.params[0].description == "The x value."


def test_cdd_sphinx_raises() -> None:
    """Function docstring."""
    ref = GhostInspector.inspect(
        dummy_func_sphinx_docstring, "tests.dummy_func_sphinx_docstring"
    )
    assert "TypeError" in ref.raises


def test_griffe_parsing() -> None:
    """Function docstring."""
    import urllib.parse

    ref = GhostInspector.inspect(urllib.parse.urljoin, "urllib.parse.urljoin")
    assert ref.name == "urljoin"
    assert ref.has_arg("base")
    assert ref.has_arg("url")


def test_cdd_exception_handling(mocker: Any) -> None:
    """Function docstring."""
    mocker.patch("cdd.docstring.parse.docstring", side_effect=ValueError)
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert ref.name == "dummy_func_with_docstring"


def test_cdd_direct_raises(mocker: Any) -> None:
    """Function docstring."""
    # Mock cdd to return direct raises list
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"typ": "KeyError"}]},
    )
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert "KeyError" in ref.raises


def test_inspect_annotation_fallback() -> None:
    """Function docstring."""

    class ForwardRefStr:
        """Class docstring."""

        # A class that lacks __name__ but is passed as annotation
        pass

    def func_annotated(a: ForwardRefStr()):  # type: ignore
        """Function docstring.

        Args:
            a: description
        """
        pass

    ref = GhostInspector.inspect(func_annotated, "tests.func_annotated")
    assert "ForwardRefStr" in ref.params[0].annotation


def test_griffe_self_skip() -> None:
    """Function docstring."""

    # griffe self skipping
    class A:
        """Class docstring."""

        def m(self, x: Any) -> Any:
            """Function docstring.

            Args:
                x: description
            """
            pass

    # we need a real class importable by griffe to test griffe skipping 'self'
    import email.message

    ref = GhostInspector.inspect(
        email.message.Message.set_payload, "email.message.Message.set_payload"
    )
    assert not ref.has_arg("self")
    assert ref.has_arg("payload")


def test_griffe_varargs() -> None:
    """Function docstring."""
    import subprocess

    ref = GhostInspector.inspect(subprocess.run, "subprocess.run")
    assert ref.has_varargs


def test_models_raises_no_typ(mocker: Any) -> None:
    """Function docstring."""
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"not_typ": "KeyError"}]},
    )
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert not ref.raises
