"""Module docstring."""

from typing import Any


from ml_framework_snapshots.models import GhostInspector


def dummy_func_with_docstring(x: int) -> str:
    """Dummy function.

    Args:
        x (int): The x value.

    Returns:
        str: A string.

    Raises:
        ValueError: If x is bad.

    # noqa: DAR202, DAR402
    """
    if x < 0:  # pragma: no cover
        raise ValueError()  # pragma: no cover
    return "test"  # pragma: no cover


def dummy_func_sphinx_docstring(y: Any) -> None:
    """Sphinx doc.

    :param y: The y value.
    :type y: int
    :raises TypeError: If wrong.

    Args:
        y: Parameter.
    """
    pass  # pragma: no cover


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
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    mocker.patch("cdd.docstring.parse.docstring", side_effect=ValueError)
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert ref.name == "dummy_func_with_docstring"


def test_cdd_direct_raises(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
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
        pass  # pragma: no cover

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
            pass  # pragma: no cover

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
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"not_typ": "KeyError"}]},
    )
    mocker.patch(
        "ml_framework_snapshots.models.extract_griffe_docstring_metadata",
        return_value={"params": {}, "returns": None, "raises": []},
    )
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert not ref.raises


def test_ghost_inspector_aten_overload_and_factory_default(mocker: Any) -> None:
    """Test GhostInspector when c_ext_params already has overloads and overload has factory default.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_framework_snapshots.utils import CExtensionSignature

    existing_overload = CExtensionSignature(
        params=[
            ("y", "POSITIONAL_OR_KEYWORD", "<factory_default>", "float"),
            ("z", "POSITIONAL_OR_KEYWORD", "1.0", "float"),
        ]
    )
    existing_sig = CExtensionSignature(
        params=[("x", "POSITIONAL_OR_KEYWORD", None, "int")],
        overloads=[existing_overload],
    )
    aten_sig = CExtensionSignature(
        params=[("x", "POSITIONAL_OR_KEYWORD", None, "int")],
        overloads=[
            CExtensionSignature(params=[("z", "POSITIONAL_OR_KEYWORD", None, "str")])
        ],
    )

    mocker.patch(
        "ml_framework_snapshots.models.extract_c_extension_signature",
        return_value=existing_sig,
    )
    mocker.patch(
        "ml_framework_snapshots.frameworks.torch.extract_aten_c_extension_signature",
        return_value=aten_sig,
    )
    mocker.patch("inspect.signature", side_effect=TypeError("no python signature"))

    def dummy_torch_fn(x: int) -> None:
        """Dummy function for torch inspection test.

        Args:
            x: Integer parameter.
        """
        pass

    ref = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert len(ref.overloads) == 1
    assert ref.overloads[0].params[0].default_factory == "<factory_default>"


def test_ghost_inspector_aten_none(mocker: Any) -> None:
    """Test GhostInspector when extract_aten_c_extension_signature returns None.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch("inspect.signature", side_effect=TypeError("no python signature"))
    mocker.patch(
        "ml_framework_snapshots.frameworks.torch.extract_aten_c_extension_signature",
        return_value=None,
    )
    mocker.patch(
        "ml_framework_snapshots.models.extract_c_extension_signature",
        return_value=None,
    )

    def dummy_torch_fn() -> None:
        """Dummy function for torch inspection test."""
        pass

    ref = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert len(ref.params) == 2
    assert ref.params[0].name == "args"
    assert ref.params[1].name == "kwargs"
