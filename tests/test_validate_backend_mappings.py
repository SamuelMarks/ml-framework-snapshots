"""Tests for scripts/validate_backend_mappings.py."""

from typing import Any
import ast
import json

from scripts.validate_backend_mappings import (
    load_snapshot,
    build_snapshot_lookup,
    format_template,
    TemplateCallVisitor,
    MockGraph,
    extract_pytorch_mappings,
    extract_numpy_mappings,
    extract_dask_mappings,
    validate_mappings,
    main,
)


def test_load_snapshot(tmp_path: Any, mocker: Any) -> None:
    """Test load_snapshot.

    Args:
        tmp_path: Parameter.
        mocker: Parameter.
    """
    mock_resolve = mocker.patch("scripts.validate_backend_mappings.Path.resolve")
    # Make Path(__file__).resolve().parent.parent point to tmp_path
    mock_resolve.return_value.parent.parent = tmp_path
    # create dummy snapshot
    snap_file = (
        tmp_path / "src" / "ml_framework_snapshots" / "snapshots" / "torch_v1.0.json"
    )
    snap_file.parent.mkdir(parents=True)
    snap_file.write_text(json.dumps({"test": "data"}))

    assert load_snapshot("torch") == {"test": "data"}


def test_load_snapshot_not_found(tmp_path: Any, mocker: Any) -> None:
    """Test load_snapshot when no file exists.

    Args:
        tmp_path: Parameter.
        mocker: Parameter.
    """
    mock_resolve = mocker.patch("scripts.validate_backend_mappings.Path.resolve")
    mock_resolve.return_value.parent.parent = tmp_path
    assert load_snapshot("unknown") == {}


def test_build_snapshot_lookup() -> None:
    """Test build_snapshot_lookup."""
    snap = {
        "categories": {
            "test_cat": [
                {"api_path": "test.func", "aliases": ["test.alias"]},
            ]
        }
    }
    lookup = build_snapshot_lookup(snap)
    assert "test.func" in lookup
    assert "test.alias" in lookup
    assert lookup["test.func"] == snap["categories"]["test_cat"][0]


def test_format_template() -> None:
    """Test format_template."""
    assert format_template("{axis} + {y}") == "v_axis + v_y"


def test_template_call_visitor() -> None:
    """Test TemplateCallVisitor."""
    code = "torch.add(x, y, alpha=True, **kwargs)"
    tree = ast.parse(code)
    visitor = TemplateCallVisitor()
    visitor.visit(tree)
    assert len(visitor.calls) == 1
    call = visitor.calls[0]
    assert call["func_name"] == "torch.add"
    assert "alpha" in call["kwargs"]


def test_mock_graph() -> None:
    """Test MockGraph."""
    graph = MockGraph()
    assert graph.nodes == {}
    assert graph.edges == []


def test_extract_numpy_mappings(mocker: Any) -> None:
    """Test extract_numpy_mappings.

    Args:
        mocker: Parameter.
    """
    res = extract_numpy_mappings()
    assert isinstance(res, dict)
    mocker.patch("scripts.validate_backend_mappings.NumpyGenerator", spec=[])
    res2 = extract_numpy_mappings()
    assert isinstance(res2, dict)


def test_extract_dask_mappings(mocker: Any) -> None:
    """Test extract_dask_mappings.

    Args:
        mocker: Parameter.
    """
    res = extract_dask_mappings()
    assert isinstance(res, dict)
    mocker.patch("scripts.validate_backend_mappings.DaskGenerator", spec=[])
    res2 = extract_dask_mappings()
    assert isinstance(res2, dict)


def test_extract_pytorch_mappings() -> None:
    """Test extract_pytorch_mappings."""
    res = extract_pytorch_mappings()
    assert isinstance(res, dict)


def test_validate_mappings(mocker: Any) -> None:
    """Test validate_mappings.

    Args:
        mocker: Parameter.
    """
    mocker.patch(
        "scripts.validate_backend_mappings.load_snapshot",
        return_value={
            "categories": {
                "cat": [
                    {"api_path": "numpy.add", "kwargs": ["out"], "params": []},
                    {
                        "api_path": "torch.add",
                        "kwargs": ["alpha"],
                        "params": [{"kind": "VAR_KEYWORD"}],
                    },
                ]
            }
        },
    )

    mappings = {
        "add1": "np.add(a, b, out=c)",
        "add2": "np.add(a, b, bad_kw=c)",
        "add3": "torch.add(a, b, other=c)",
        "add4": "int(a)",
        "add5": "torch.hallucinated()",
        "invalid": "def 1",
        "jnp_add": "jnp.add(a, b)",
        "cp_add": "cp.add(a, b)",
        "da_add": "da.add(a, b)",
        "tf_add": "tf.add(a, b)",
        "no_dot": "nodot()",
        "not_str": 123,
        "exec_mode": "a = np.add(x, y)",
        "kwargs_star": "np.add(**kwargs)",
        "complex_attr": "get_obj().add(x, y)",
        "subscript_call": "funcs[0](a, b)",
    }
    count, errors = validate_mappings("test", mappings)  # type: ignore[arg-type]
    assert count == 16
    assert any("Hallucinated kwarg" in e for e in errors)
    assert any("Hallucinated API" in e for e in errors)
    assert any("SyntaxError" in e for e in errors)


def test_extract_pytorch_mappings_errors(mocker: Any) -> None:
    """Test extract_pytorch_mappings with errors.

    Args:
        mocker: Parameter.
    """
    mock_gen = mocker.patch("scripts.validate_backend_mappings.PyTorchCodeGenerator")
    mock_gen.return_value._get_math_ops.side_effect = Exception("math error")
    mock_gen.return_value._get_creation_ops.side_effect = Exception("creation error")
    mock_gen.return_value._get_array_ops.side_effect = Exception("array error")
    del mock_gen.return_value._OP_MAP
    res = extract_pytorch_mappings()
    assert isinstance(res, dict)


def test_validate_mappings_empty_snapshot(mocker: Any) -> None:
    """Test validate_mappings with empty snapshot.

    Args:
        mocker: Parameter.
    """
    mocker.patch("scripts.validate_backend_mappings.load_snapshot", return_value={})
    count, errors = validate_mappings("test", {"add": "add()"})
    assert count == 0
    assert len(errors) == 0


def test_main_success(mocker: Any) -> None:
    """Test main function with no errors.

    Args:
        mocker: Parameter.
    """
    mocker.patch(
        "scripts.validate_backend_mappings.validate_mappings", return_value=(1, [])
    )
    mock_exit = mocker.patch("sys.exit")
    main()
    mock_exit.assert_called_with(0)


def test_main_with_errors(mocker: Any) -> None:
    """Test main function with errors.

    Args:
        mocker: Parameter.
    """
    mocker.patch(
        "scripts.validate_backend_mappings.validate_mappings",
        return_value=(1, ["error1"]),
    )
    mock_exit = mocker.patch("sys.exit")
    main()
    mock_exit.assert_called_with(0)
