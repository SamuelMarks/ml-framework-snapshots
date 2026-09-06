"""Tests for Griffe reST/Sphinx docstring parser integration and fallback."""

from typing import Any

from ml_framework_snapshots.models import GhostInspector, sanitize_type_str
from ml_framework_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
    parse_docstring_with_griffe,
    extract_griffe_docstring_metadata,
    strip_sphinx_roles,
)
import griffe


def test_griffe_rest_sphinx_field_lists() -> None:
    """Test parsing complex Sphinx field lists with :param:, :type:, :return:, :rtype:, :raises:."""
    docstring = """Compute scaled dot-product attention.

:param query: Query tensor of shape (batch, seq, dim).
:type query: torch.Tensor
:param key: Key tensor of shape (batch, seq, dim).
:type key: torch.Tensor
:param value: Value tensor.
:type value: torch.Tensor
:return: Attended output tensor.
:rtype: torch.Tensor
:raises ValueError: If tensor shapes do not align.
:raises RuntimeError: If out of memory.
"""
    meta = extract_griffe_docstring_metadata(docstring, parser_name="rest")

    # Verify parameters
    assert "query" in meta["params"]
    assert "key" in meta["params"]
    assert "value" in meta["params"]

    assert meta["params"]["query"]["typ"] == "torch.Tensor"
    assert "Query tensor" in meta["params"]["query"]["doc"]

    # Verify return
    assert meta["returns"] is not None
    assert meta["returns"]["typ"] == "torch.Tensor"
    assert "Attended output tensor" in meta["returns"]["doc"]

    # Verify raises
    assert "ValueError" in meta["raises"]
    assert "RuntimeError" in meta["raises"]


def test_get_framework_docstring_parser() -> None:
    """Test framework docstring parser resolution heuristics."""
    assert get_framework_docstring_parser("torch") == "rest"
    assert get_framework_docstring_parser("pytorch") == "rest"
    assert get_framework_docstring_parser("tensorflow") == "rest"
    assert get_framework_docstring_parser("sklearn") == "rest"
    assert get_framework_docstring_parser("scipy") == "rest"
    assert get_framework_docstring_parser("numpy") == "numpy"
    assert get_framework_docstring_parser("jax") == "google"
    assert get_framework_docstring_parser("keras") == "google"
    assert get_framework_docstring_parser("transformers") == "google"
    assert get_framework_docstring_parser("unknown_framework") == "rest"

    # Content-based overrides
    sphinx_doc = """:param a: input
:return: output"""
    assert get_framework_docstring_parser("numpy", docstring=sphinx_doc) == "rest"

    numpy_doc = """Description

Parameters
----------
a : int
"""
    assert get_framework_docstring_parser("torch", docstring=numpy_doc) == "numpy"

    google_doc = """Description

Args:
    a (int): input
"""
    assert get_framework_docstring_parser("torch", docstring=google_doc) == "google"


def test_resolve_griffe_parser() -> None:
    """Test resolving parser strings to Griffe Parser enum instances."""
    rest_parser = resolve_griffe_parser("rest")
    assert rest_parser in (
        getattr(griffe.Parser, "rest", griffe.Parser.sphinx),
        griffe.Parser.sphinx,
    )

    sphinx_parser = resolve_griffe_parser("sphinx")
    assert sphinx_parser == griffe.Parser.sphinx

    google_parser = resolve_griffe_parser("google")
    assert google_parser == griffe.Parser.google

    numpy_parser = resolve_griffe_parser("numpy")
    assert numpy_parser == griffe.Parser.numpy

    auto_parser = resolve_griffe_parser("auto")
    assert auto_parser == griffe.Parser.auto

    invalid_parser = resolve_griffe_parser("invalid_style_123")
    assert invalid_parser is None


def test_parse_docstring_dynamic_fallback() -> None:
    """Test dynamic fallback between conventions when a docstring style differs from requested."""
    google_doc = """Perform matrix multiplication.

Args:
    a (Tensor): Left operand matrix.
    b (Tensor): Right operand matrix.

Returns:
    Tensor: Multiplied result matrix.

Raises:
    ValueError: If dimensions do not match.
"""
    # Ask for rest, but document is in Google style -> dynamic fallback should succeed
    sections = parse_docstring_with_griffe(google_doc, parser_name="rest")
    assert len(sections) > 1
    has_params = any(
        getattr(s.kind, "value", str(s.kind)) == "parameters" for s in sections
    )
    assert has_params

    numpy_doc = """Perform matrix multiplication.

Parameters
----------
a : Tensor
    Left operand matrix.
b : Tensor
    Right operand matrix.

Returns
-------
Tensor
    Multiplied result matrix.

Raises
------
ValueError
    If dimensions do not match.
"""
    # Ask for rest, but document is in NumPy style -> dynamic fallback should succeed
    sections_np = parse_docstring_with_griffe(numpy_doc, parser_name="rest")
    assert len(sections_np) > 1
    has_params_np = any(
        getattr(s.kind, "value", str(s.kind)) == "parameters" for s in sections_np
    )
    assert has_params_np

    # Plain text docstring has no structured sections but returns plain section
    plain_doc = "Simple summary without structured sections."
    plain_sections = parse_docstring_with_griffe(plain_doc, parser_name="rest")
    assert len(plain_sections) >= 1


def test_ghost_inspector_sphinx_docstring() -> None:
    """Test GhostInspector inspection with full Sphinx docstring field lists."""

    def dummy_op(x: Any, y: Any) -> Any:
        """Dummy op summary.

        :param x: Primary input vector.
        :type x: torch.Tensor
        :param y: Offset vector.
        :type y: torch.Tensor
        :return: Shifted result.
        :rtype: torch.Tensor
        :raises ValueError: If dimension mismatch.
        :raises TypeError: If non-tensor passed.
        """
        pass  # pragma: no cover

    ref = GhostInspector.inspect(dummy_op, "torch.ops.dummy_op")

    assert ref.name == "dummy_op"
    assert ref.returns_type == "torch.Tensor"
    assert ref.returns_description == "Shifted result."
    assert "ValueError" in (ref.raises or [])
    assert "TypeError" in (ref.raises or [])

    x_param = next(p for p in ref.params if p.name == "x")
    assert x_param.annotation == "torch.Tensor"
    assert "Primary input vector" in (x_param.description or "")


def test_ghost_inspector_merge_branches(mocker: Any) -> None:
    """Test merging branches: return description only, CDD exception cleanup, param doc/type augmentation."""

    def dummy_return_only(a: int) -> None:
        """Dummy return only summary.

        :param a: An argument.
        :return: Only return description without rtype.
        :raises ValueError: If bad.
        """
        pass  # pragma: no cover

    # Mock cdd to have an existing param without doc, and another without typ, and an exception in params
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={
            "params": {
                "a": {"doc": "", "typ": "int"},
                "b": {"doc": "B doc", "typ": ""},
                "ValueError": {"doc": "Should be pruned from params"},
            },
            "returns": {"return_type": {"doc": None, "typ": None}},
        },
    )

    mocker.patch(
        "ml_framework_snapshots.models.extract_griffe_docstring_metadata",
        return_value={
            "params": {
                "a": {"doc": "Augmented doc", "typ": "int", "default": None},
                "b": {"doc": "B doc", "typ": "float", "default": None},
            },
            "returns": {"doc": "Only return description without rtype.", "typ": None},
            "raises": ["ValueError"],
        },
    )

    ref = GhostInspector.inspect(dummy_return_only, "torch.dummy_return_only")
    assert ref.returns_description == "Only return description without rtype."
    assert "ValueError" in (ref.raises or [])
    assert not ref.has_arg("ValueError")

    # Test hydrate with and without schema_version
    hydrated_with = GhostInspector.hydrate(
        {"name": "op", "api_path": "op", "kind": "function", "schema_version": "2.0"}
    )
    assert hydrated_with.schema_version == "2.0"

    hydrated_without = GhostInspector.hydrate(
        {"name": "op", "api_path": "op", "kind": "function"}
    )
    assert hydrated_without.schema_version == "1.2"


def test_extract_griffe_docstring_metadata_edge_cases(mocker: Any) -> None:
    """Test extract_griffe_docstring_metadata on anonymous parameters, nameless raises, and duplicates."""

    class MockParam:
        """Mock Griffe parameter element."""

        def __init__(self, name: Any, desc: Any, anno: Any) -> None:
            """Initialize MockParam.

            Args:
                name: Name of parameter.
                desc: Parameter description.
                anno: Parameter type annotation.
            """
            self.name = name
            self.description = desc
            self.annotation = anno
            self.value = None

    class MockReturn:
        """Mock Griffe return element."""

        def __init__(self, desc: Any, anno: Any) -> None:
            """Initialize MockReturn.

            Args:
                desc: Return description.
                anno: Return type annotation.
            """
            self.description = desc
            self.annotation = anno

    class MockRaise:
        """Mock Griffe raise element."""

        def __init__(self, anno: Any) -> None:
            """Initialize MockRaise.

            Args:
                anno: Exception annotation name.
            """
            self.annotation = anno

    class MockSection:
        """Mock Griffe docstring section."""

        def __init__(self, kind: str, value: Any) -> None:
            """Initialize MockSection.

            Args:
                kind: Section kind name.
                value: Section contents.
            """
            self.kind = kind
            self.value = value

    mock_sections = [
        MockSection(
            "parameters",
            [MockParam(None, "no name", "int"), MockParam("x", "valid", "int")],
        ),
        MockSection("returns", [MockReturn("desc", "int")]),
        MockSection(
            "raises",
            [
                MockRaise(None),
                MockRaise("ValueError"),
                MockRaise("ValueError"),
            ],
        ),
    ]

    mocker.patch(
        "ml_framework_snapshots.utils.parse_docstring_with_griffe",
        return_value=mock_sections,
    )

    meta = extract_griffe_docstring_metadata("any doc")
    assert "x" in meta["params"]
    assert None not in meta["params"]
    assert meta["raises"] == ["ValueError"]


def test_ghost_inspector_custom_fault_and_doc_return() -> None:
    """Test GhostInspector capturing custom non-Error/Exception raises and doc-only returns."""

    def fn(x: Any) -> Any:
        """Function summary.

        :param x: Value.
        :return: Return doc only.
        :raises CustomFault: When faulted.
        """
        pass  # pragma: no cover

    ref = GhostInspector.inspect(fn, "torch.fn")
    assert ref.returns_description == "Return doc only."
    assert "CustomFault" in (ref.raises or [])
    assert not ref.has_arg("CustomFault")


def test_ghost_inspector_griffe_return_type_fallback(mocker: Any) -> None:
    """Test GhostInspector extracting returns_type from Griffe when CDD does not find one."""

    def dummy_ret(x: Any) -> Any:
        """Dummy ret summary.

        :param x: Input.
        :return: Output.
        :rtype: torch.Tensor
        """
        pass  # pragma: no cover

    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"params": {}, "returns": None},
    )
    mocker.patch(
        "ml_framework_snapshots.models.extract_griffe_docstring_metadata",
        return_value={
            "params": {},
            "returns": {"typ": "torch.Tensor", "doc": "Output."},
            "raises": [],
        },
    )

    ref = GhostInspector.inspect(dummy_ret, "torch.dummy_ret")
    assert ref.returns_type == "torch.Tensor"
    assert ref.returns_description == "Output."


def test_parse_docstring_with_griffe_unresolved(mocker: Any) -> None:
    """Test parse_docstring_with_griffe when a candidate parser cannot be resolved."""
    call_count = 0

    def mock_resolve(name: str) -> Any:
        """Mock resolve_griffe_parser to return None on the first call.

        Args:
            name: The parser name to resolve.

        Returns:
            None on first call, then griffe.Parser.sphinx.
        """
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None
        return griffe.Parser.sphinx

    mocker.patch(
        "ml_framework_snapshots.utils.resolve_griffe_parser",
        side_effect=mock_resolve,
    )

    sections = parse_docstring_with_griffe("doc", parser_name="invalid")
    assert len(sections) >= 1


def test_strip_sphinx_roles() -> None:
    """Test stripping various Sphinx cross-referencing roles."""
    assert strip_sphinx_roles(None) is None
    assert strip_sphinx_roles(":class:`torch.Tensor`") == "torch.Tensor"
    assert strip_sphinx_roles(":class:`~torch.Tensor`") == "torch.Tensor"
    assert (
        strip_sphinx_roles(":func:`~torch.nn.functional.relu`")
        == "torch.nn.functional.relu"
    )
    assert strip_sphinx_roles(":obj:`None`") == "None"
    assert (
        strip_sphinx_roles(":meth:`~torch.Tensor.backward`") == "torch.Tensor.backward"
    )
    assert strip_sphinx_roles(":attr:`data`") == "data"
    assert strip_sphinx_roles("plain text without roles") == "plain text without roles"


def test_sanitize_type_str_with_sphinx_roles() -> None:
    """Test that sanitize_type_str cleans Sphinx roles before parsing type hints."""
    assert sanitize_type_str(":class:`~torch.Tensor`") == "torch.Tensor"
    assert (
        sanitize_type_str("Union[:class:`torch.Tensor`, :obj:`None`]")
        == "torch.Tensor | None"
    )
    assert (
        sanitize_type_str("Optional[:class:`~torch.Tensor`]") == "torch.Tensor | None"
    )


def test_promote_documented_kwargs() -> None:
    """Test promoting documented kwargs from Sphinx/reST docstrings to formal GhostParams."""

    def custom_layer(in_features: int, **kwargs: Any) -> None:
        """Initialize custom layer.

        :param in_features: Size of each input sample.
        :type in_features: int
        :param bias: If set to False, additive bias is omitted.
        :type bias: bool
        :param device: Target compute device.
        :type device: str
        """
        pass

    ref = GhostInspector.inspect(custom_layer, "pkg.custom_layer")
    param_names = [p.name for p in ref.params]
    assert "in_features" in param_names
    assert "bias" in param_names
    assert "device" in param_names
    assert "kwargs" in param_names

    bias_param = next(p for p in ref.params if p.name == "bias")
    assert str(bias_param.kind).endswith("KEYWORD_ONLY")
    assert bias_param.annotation == "bool"
    assert "additive bias" in (bias_param.description or "")

    device_param = next(p for p in ref.params if p.name == "device")
    assert str(device_param.kind).endswith("KEYWORD_ONLY")
    assert device_param.annotation == "str"


def test_pytorch_modules_kwargs_promotion() -> None:
    """Test parameter extraction and kwargs promotion on PyTorch nn.Linear and nn.Conv2d."""
    import torch.nn as nn

    linear_ref = GhostInspector.inspect(nn.Linear, "torch.nn.Linear", kind="class")
    linear_param_names = {p.name for p in linear_ref.params}
    assert "in_features" in linear_param_names
    assert "out_features" in linear_param_names
    assert "bias" in linear_param_names

    conv_ref = GhostInspector.inspect(nn.Conv2d, "torch.nn.Conv2d", kind="class")
    conv_param_names = {p.name for p in conv_ref.params}
    assert "in_channels" in conv_param_names
    assert "out_channels" in conv_param_names
    assert "kernel_size" in conv_param_names
