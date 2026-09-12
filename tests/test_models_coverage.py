"""Coverage tests for models module.

Tests missing lines and branches in ml_framework_snapshots/models.py.
"""

import ast
from typing import Any, List
from unittest.mock import patch

from ml_framework_snapshots.models import (
    GhostInspector,
    extract_accepted_kwargs_from_ast,
)


class BaseParentModel:
    """Base parent class."""

    def __init__(self, arg1: int = 1) -> None:
        """Initialize BaseParentModel.

        Args:
            arg1: Integer parameter.
        """
        pass


class OtherHelperModel:
    """Helper class."""

    def __init__(self) -> None:
        """Initialize helper."""
        pass


helper_instance = OtherHelperModel()


class DerivedClassModel(BaseParentModel):
    """Derived class with custom __init__ calls."""

    def __init__(self, a: int = 10) -> None:
        """Initialize DerivedClassModel.

        Args:
            a: Parameter a.
        """
        OtherHelperModel.__init__(helper_instance)
        super().__init__(arg1=a)


class SimpleClassModel:
    """Simple class.

    Args:
        x: Input.
    """

    def __init__(self, x: int) -> None:
        """Initialize SimpleClassModel.

        Args:
            x: Input.
        """
        pass


def target_fn_ast_edges(
    self: Any, ast_arg: int = 42, broken_arg: str = "x", hint_arg: int = 1
) -> None:
    """Target function for AST default edges.

    Args:
        self: Self parameter.
        ast_arg: AST default parameter.
        broken_arg: Broken default parameter.
        hint_arg: Parameter resolving type from hint.
    """
    pass


def target_fn_sig_failure(x: int) -> None:
    """Target function for signature failure in CDD branch.

    Args:
        x: Parameter.
    """
    pass


class MroParentClass:
    """Parent class."""

    def __init__(self, p: int = 1) -> None:
        """Initialize MroParentClass.

        Args:
            p: Parameter p.
        """
        pass


class MroChildClass(MroParentClass):
    """Child class calling super with **kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MroChildClass.

        Args:
            **kwargs: Keyword args passed to super.
        """
        super().__init__(**kwargs)


def test_extract_accepted_kwargs_subscript_branches() -> None:
    """Test extract_accepted_kwargs_from_ast across all Subscript AST branches.

    Returns:
        None.
    """
    tree = ast.parse("def sample_fn(**kwargs):\n    kwargs[123]\n    kwargs['alpha']\n")
    fn_def = tree.body[0]

    # 1. slice is Index(Constant('alpha')) -> covers hasattr(ast, 'Index') and idx_val as string Constant
    idx2: Any = ast.slice.__new__(ast.Index)
    idx2.value = ast.Constant(value="alpha")
    fn_def.body[1].value.slice = idx2  # type: ignore[attr-defined]

    # 2. slice is Index(Constant(999)) -> covers idx_val with non-string Constant
    idx3: Any = ast.slice.__new__(ast.Index)
    idx3.value = ast.Constant(value=999)
    sub3 = ast.Subscript(
        value=ast.Name(id="kwargs", ctx=ast.Load()), slice=idx3, ctx=ast.Load()
    )
    fn_def.body.append(ast.Expr(value=sub3))  # type: ignore[attr-defined]

    # 3. slice is Index(Name('x')) -> covers idx_val with non-Constant node
    idx4: Any = ast.slice.__new__(ast.Index)
    idx4.value = ast.Name(id="x", ctx=ast.Load())
    sub4 = ast.Subscript(
        value=ast.Name(id="kwargs", ctx=ast.Load()), slice=idx4, ctx=ast.Load()
    )
    fn_def.body.append(ast.Expr(value=sub4))  # type: ignore[attr-defined]

    # 4. slice is Name('var') -> covers slice_node that is neither Constant nor Index
    sub5 = ast.Subscript(
        value=ast.Name(id="kwargs", ctx=ast.Load()),
        slice=ast.Name(id="var", ctx=ast.Load()),
        ctx=ast.Load(),
    )
    fn_def.body.append(ast.Expr(value=sub5))  # type: ignore[attr-defined]

    with patch("inspect.getsource", return_value="def sample_fn(**kwargs): pass"):
        with patch("ml_framework_snapshots.models.ast.parse", return_value=tree):
            res = extract_accepted_kwargs_from_ast(lambda **kwargs: None)
            assert res == ["alpha"]


def test_ghost_inspector_unwrap_max_depth() -> None:
    """Test recursive unwrap loop when max depth 10 is reached without break.

    Returns:
        None.
    """

    def make_wrapper(inner: Any) -> Any:
        """Create function wrapper with __wrapped__ attribute.

        Args:
            inner: Inner callable.

        Returns:
            Wrapped callable.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Wrapped function.

            Args:
                *args: Positional args.
                **kwargs: Keyword args.

            Returns:
                Call result.
            """
            return inner(*args, **kwargs)

        setattr(wrapped, "__wrapped__", inner)
        return wrapped

    def base_func(x: int) -> int:
        """Base function.

        Args:
            x: Input value.

        Returns:
            Result value.
        """
        return x

    curr: Any = base_func
    for _ in range(12):
        curr = make_wrapper(curr)

    ref = GhostInspector.inspect(curr, "test_deep_wrapper")
    assert ref.has_arg("x")


def test_ghost_inspector_cdd_docstring_exception(mocker: Any) -> None:
    """Test GhostInspector handles exception in cdd_docstring_parsers.docstring.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.docstring.parse as cdd_docstring_parsers

    mocker.patch.object(
        cdd_docstring_parsers,
        "docstring",
        side_effect=RuntimeError("simulated cdd failure"),
    )

    def dummy(x: int) -> None:
        """Dummy docstring.

        Args:
            x: Value.
        """
        pass

    ref = GhostInspector.inspect(dummy, "dummy_fn")
    assert ref.has_arg("x")


def test_ghost_inspector_griffe_docstring_exception(mocker: Any) -> None:
    """Test GhostInspector handles exception in extract_griffe_docstring_metadata.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    mocker.patch(
        "ml_framework_snapshots.models.extract_griffe_docstring_metadata",
        side_effect=RuntimeError("simulated griffe doc error"),
    )

    def dummy(x: int) -> None:
        """Dummy docstring.

        Args:
            x: Value.
        """
        pass

    ref = GhostInspector.inspect(dummy, "dummy_griffe_doc_fn")
    assert ref.has_arg("x")


def test_ghost_inspector_super_init_branches() -> None:
    """Test GhostInspector AST inspection branches for super().__init__ calls.

    Returns:
        None.
    """
    ref = GhostInspector.inspect(DerivedClassModel, "DerivedClassModel")
    assert ref.has_arg("a")


def test_ghost_inspector_cdd_parsed_ir_none(mocker: Any) -> None:
    """Test GhostInspector when cdd_parsed_ir has no params attribute.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.class_.parse

    mocker.patch.object(cdd.class_.parse, "class_", return_value={})

    ref = GhostInspector.inspect(SimpleClassModel, "SimpleClassModel")
    assert ref.has_arg("x")


def test_ghost_inspector_cdd_ast_params_edges(mocker: Any) -> None:
    """Test GhostInspector cdd_ast_params handling for self, AST defaults, and hints.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.function.parse

    ast_default = ast.Constant(value=42)

    class BrokenAST(ast.AST):
        """Broken AST node."""

        pass

    broken_default = BrokenAST()

    fake_cdd_ir = {
        "params": {
            "self": {"typ": "Any"},
            "ast_arg": {"default": ast_default, "typ": "int"},
            "broken_arg": {"default": broken_default, "typ": "str"},
            "hint_arg": {"default": None},
        }
    }

    orig_unparse = ast.unparse

    def mock_unparse(node: Any) -> str:
        """Mock unparse raising on BrokenAST.

        Args:
            node: AST node.

        Raises:
            ValueError: Simulated unparse error on BrokenAST.

        Returns:
            Unparsed string.
        """
        if isinstance(node, BrokenAST):
            raise ValueError("unparse failed")
        return orig_unparse(node)

    mocker.patch("ml_framework_snapshots.models.ast.unparse", side_effect=mock_unparse)
    mocker.patch.object(cdd.function.parse, "function", return_value=fake_cdd_ir)

    ref = GhostInspector.inspect(target_fn_ast_edges, "target_fn_ast_edges")
    assert not ref.has_arg("self")
    assert ref.has_arg("ast_arg")
    assert ref.has_arg("broken_arg")
    assert ref.has_arg("hint_arg")


def test_ghost_inspector_standard_fallback_exception_in_cdd_branch(mocker: Any) -> None:
    """Test GhostInspector handles exception in inspect.signature within CDD branch.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.function.parse

    fake_cdd_ir = {
        "params": {
            "x": {"typ": "int"},
        }
    }
    mocker.patch.object(cdd.function.parse, "function", return_value=fake_cdd_ir)
    mocker.patch("inspect.signature", side_effect=TypeError("broken signature"))

    ref = GhostInspector.inspect(target_fn_sig_failure, "target_fn_sig_failure")
    assert ref.has_arg("x")


def test_ghost_inspector_parent_mro_signature_exception(mocker: Any) -> None:
    """Test GhostInspector handles exception when parent signature inspection fails during super kwargs resolution.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    real_signature = __import__("inspect").signature

    def mock_signature(obj: Any, *args: Any, **kwargs: Any) -> Any:
        """Mock signature raising on MroParentClass.__init__.

        Args:
            obj: Target object.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: Simulated signature failure on parent __init__.

        Returns:
            Signature object.
        """
        if obj == MroParentClass.__init__:
            raise TypeError("parent init signature failure")
        return real_signature(obj, *args, **kwargs)

    mocker.patch("inspect.signature", side_effect=mock_signature)

    ref = GhostInspector.inspect(MroChildClass, "MroChildClass")
    assert ref.api_path == "MroChildClass"


def test_ghost_inspector_torch_target_infer_exception(mocker: Any) -> None:
    """Test GhostInspector handles infer_torch_dtype_and_rank exceptions.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    from ml_framework_snapshots.frameworks import torch as torch_fw

    mocker.patch.object(
        torch_fw,
        "infer_torch_dtype_and_rank",
        side_effect=ValueError("simulated infer error"),
    )

    def dummy_torch_fn(x: int) -> None:
        """Dummy function.

        Args:
            x: Input.
        """
        pass

    ref = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert ref.has_arg("x")


def test_ghost_inspector_overload_without_parameters() -> None:
    """Test GhostInspector overload branch when overload object has no parameters attribute.

    Returns:
        None.
    """

    class BareOverload:
        """Overload object without parameters attribute."""

        returns = "int"

    class FakeNodeWithOverload:
        """Fake griffe node containing bare overload."""

        is_class = False
        is_function = True
        parameters: List[Any] = []
        docstring = None
        overloads = [BareOverload()]

    ref = GhostInspector.inspect(FakeNodeWithOverload(), "bare_overload_fn")
    assert len(ref.overloads) == 1
    assert ref.overloads[0].returns_type == "int"


def test_ghost_inspector_c_ext_overload_infer_exception(mocker: Any) -> None:
    """Test GhostInspector c_ext overloads when infer_torch_dtype_and_rank raises.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    from ml_framework_snapshots.frameworks import torch as torch_fw

    mocker.patch.object(
        torch_fw,
        "infer_torch_dtype_and_rank",
        side_effect=RuntimeError("simulated c_ext infer error"),
    )

    class CExtSignature(List[Any]):
        """Mock C-extension signature list subclass."""

        overloads: List[Any] = []
        returns_type: str = "Tensor"

    fake_sig = CExtSignature([("x", "POSITIONAL_OR_KEYWORD", None, "Tensor")])
    fake_sig.overloads = [
        [("x", "POSITIONAL_OR_KEYWORD", None, "Tensor")],
    ]

    mocker.patch(
        "ml_framework_snapshots.models.extract_c_extension_signature",
        return_value=fake_sig,
    )
    mocker.patch("inspect.signature", side_effect=TypeError("no signature"))

    def c_ext_dummy(x: Any) -> None:
        """Dummy C extension function.

        Args:
            x: Tensor.
        """
        pass

    ref = GhostInspector.inspect(c_ext_dummy, "torch.c_ext_dummy")
    assert len(ref.overloads) == 1
    assert ref.overloads[0].params[0].name == "x"
