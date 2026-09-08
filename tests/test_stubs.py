"""Module docstring."""

from pathlib import Path
import os

from typing import Any


from ml_framework_snapshots.stubs import generate_stubs


def test_generate_stubs(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    snap = {
        "categories": {
            "layer": [
                {
                    "api_path": "test_fw.nn.Linear",
                    "kind": "class",
                    "params": [
                        {
                            "name": "in_features",
                            "annotation": "int",
                            "default": None,
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                        {
                            "name": "out_features",
                            "annotation": "int",
                            "default": None,
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                    ],
                    "returns_type": "None",
                },
                {
                    "api_path": "test_fw.nn.EmptyClass",
                    "kind": "class",
                    "params": [],
                    "returns_type": None,
                },
                {
                    "api_path": "test_fw.functional.relu",
                    "kind": "function",
                    "params": [
                        {
                            "name": "x",
                            "annotation": "Tensor",
                            "default": None,
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                        {
                            "name": "args",
                            "kind": "VAR_POSITIONAL",
                            "default": None,
                            "annotation": None,
                        },
                        {
                            "name": "kwargs",
                            "kind": "VAR_KEYWORD",
                            "default": None,
                            "annotation": None,
                        },
                    ],
                    "has_varargs": True,
                    "returns_type": "Tensor",
                },
                {
                    "api_path": "test_fw.functional.varargs_func",
                    "kind": "function",
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
                    "has_varargs": True,
                    "returns_type": None,
                },
                {"api_path": "", "kind": "function", "params": []},
            ]
        }
    }

    out_dir = Path(os.path.join(tmp_path, "stubs"))
    generate_stubs(snap, str(out_dir))

    nn_init = Path(
        os.path.join(
            Path(os.path.join(Path(os.path.join(out_dir, "test_fw")), "nn")),
            "__init__.pyi",
        )
    )
    assert nn_init.exists()

    content = nn_init.read_text()
    assert "class Linear:" in content
    assert (
        "def __init__(self, in_features: int, out_features: int) -> None: ..."
        in content
    )
    assert "class EmptyClass:" in content
    assert "def __init__(self) -> Any: ..." in content

    func_init = Path(
        os.path.join(
            Path(os.path.join(Path(os.path.join(out_dir, "test_fw")), "functional")),
            "__init__.pyi",
        )
    )
    assert func_init.exists()
    content = func_init.read_text()
    assert "def relu(x: Tensor, *args, **kwargs) -> Tensor: ..." in content
    assert "def varargs_func(x: Any, *args: Any) -> Any: ..." in content


def test_generate_stubs_empty(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    out_dir = Path(os.path.join(tmp_path, "stubs"))
    generate_stubs({}, str(out_dir))
    assert not Path(os.path.join(out_dir, "test_fw")).exists()


def test_generate_stubs_default_vals(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    snap = {
        "categories": {
            "layer": [
                {
                    "api_path": "test_fw.nn.Linear",
                    "kind": "class",
                    "params": [
                        {
                            "name": "in_features",
                            "annotation": "int",
                            "default": "10",
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                    ],
                }
            ]
        }
    }
    out_dir = Path(os.path.join(tmp_path, "stubs"))
    generate_stubs(snap, str(out_dir))

    nn_init = Path(
        os.path.join(
            Path(os.path.join(Path(os.path.join(out_dir, "test_fw")), "nn")),
            "__init__.pyi",
        )
    )
    content = nn_init.read_text()
    assert "def __init__(self, in_features: int = 10) -> Any: ..." in content


def test_generate_stubs_no_module(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    data = {
        "categories": {
            "losses": [
                {
                    "api_path": "NoModuleObj",
                    "name": "NoModuleObj",
                    "kind": "function",
                    "params": [],
                    "has_varargs": False,
                }
            ]
        }
    }

    generate_stubs(data, str(tmp_path))
    assert not list(tmp_path.glob("**/*.pyi"))


def test_generate_stubs_include_nonpublic(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    snap = {
        "categories": {
            "layer": [
                {
                    "api_path": "test_fw.nn._PrivateLinear",
                    "kind": "class",
                    "is_public": False,
                    "params": [],
                    "returns_type": "None",
                }
            ]
        }
    }

    out_dir = Path(os.path.join(tmp_path, "stubs_priv"))
    generate_stubs(snap, str(out_dir))

    nn_init = Path(
        os.path.join(
            Path(os.path.join(Path(os.path.join(out_dir, "test_fw")), "nn")),
            "__init__.pyi",
        )
    )
    assert not nn_init.exists()

    generate_stubs(snap, str(out_dir), include_nonpublic=True)
    assert nn_init.exists()
    content = nn_init.read_text()
    assert "class _PrivateLinear:" in content


def test_generate_stubs_list_and_non_dict(tmp_path: Any) -> None:
    """Test generate_stubs when snapshot_data is a list or an unsupported type.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    import pytest
    from ml_framework_snapshots.stubs import validate_pyi_stub

    # Test snapshot_data as list
    list_data = [
        {
            "api_path": "list_fw.mod.helper",
            "kind": "function",
            "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
        }
    ]
    list_out_dir = Path(os.path.join(tmp_path, "stubs_list"))
    generate_stubs(list_data, str(list_out_dir))
    helper_init = Path(os.path.join(list_out_dir, "list_fw", "mod", "__init__.pyi"))
    assert helper_init.exists()
    assert "def helper(x: Any) -> Any: ..." in helper_init.read_text()

    # Test snapshot_data as unsupported non-dict non-list type (e.g. integer or string)
    invalid_out_dir = Path(os.path.join(tmp_path, "stubs_invalid"))
    generate_stubs(42, str(invalid_out_dir))
    assert not list(invalid_out_dir.glob("**/*.pyi"))

    # Test validate_pyi_stub raising SyntaxError
    with pytest.raises(SyntaxError):
        validate_pyi_stub("def invalid syntax ::::")


def test_generate_stubs_overloads_and_dsl(tmp_path: Any) -> None:
    """Test stub generation with overloads and non-Python domain DSL items."""
    snap = {
        "categories": {
            "ops": [
                {
                    "api_path": "dsl_fw.math.add",
                    "kind": "function",
                    "params": [
                        {
                            "name": "x",
                            "annotation": "Tensor",
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                        {
                            "name": "y",
                            "annotation": "Tensor",
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                    ],
                    "returns_type": "Tensor",
                    "overloads": [
                        {
                            "params": [
                                {
                                    "name": "x",
                                    "annotation": "int",
                                    "kind": "POSITIONAL_OR_KEYWORD",
                                },
                                {
                                    "name": "y",
                                    "annotation": "int",
                                    "kind": "POSITIONAL_OR_KEYWORD",
                                },
                            ],
                            "returns_type": "int",
                        },
                        {
                            "params": [
                                {
                                    "name": "x",
                                    "annotation": "float",
                                    "kind": "POSITIONAL_OR_KEYWORD",
                                },
                                {
                                    "name": "y",
                                    "annotation": "float",
                                    "kind": "POSITIONAL_OR_KEYWORD",
                                },
                            ],
                            "returns_type": "float",
                        },
                        {
                            "params": [
                                {"name": "args", "kind": "VAR_POSITIONAL"},
                                {"name": "kwargs", "kind": "VAR_KEYWORD"},
                            ],
                            "returns_type": "Any",
                        },
                    ],
                },
                {
                    "api_path": "dsl_fw.ops.Dense",
                    "kind": "class",
                    "params": [
                        {
                            "name": "units",
                            "annotation": "int",
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                    ],
                    "overloads": [
                        {
                            "params": [
                                {
                                    "name": "units",
                                    "annotation": "int",
                                    "kind": "POSITIONAL_OR_KEYWORD",
                                },
                                {
                                    "name": "activation",
                                    "annotation": "str",
                                    "default": "'relu'",
                                },
                            ],
                            "returns_type": "None",
                        }
                    ],
                },
                {
                    "api_path": "HMMA16816",
                    "framework": "nvidia_sass",
                    "kind": "function",
                    "params": [
                        {
                            "name": "d",
                            "annotation": "str",
                            "kind": "POSITIONAL_OR_KEYWORD",
                        },
                    ],
                },
            ]
        }
    }

    out_dir = Path(os.path.join(tmp_path, "stubs_overload"))
    generate_stubs(snap, str(out_dir))

    # Math stubs with overloads
    math_pyi = Path(os.path.join(out_dir, "dsl_fw", "math", "__init__.pyi"))
    assert math_pyi.exists()
    math_text = math_pyi.read_text()
    assert "@overload" in math_text
    assert "def add(x: int, y: int) -> int: ..." in math_text
    assert "def add(x: float, y: float) -> float: ..." in math_text
    assert "def add(x: Tensor, y: Tensor) -> Tensor: ..." in math_text

    # Class overloads
    ops_pyi = Path(os.path.join(out_dir, "dsl_fw", "ops", "__init__.pyi"))
    assert ops_pyi.exists()
    ops_text = ops_pyi.read_text()
    assert "@overload" in ops_text
    assert (
        "def __init__(self, units: int, activation: str = 'relu') -> None: ..."
        in ops_text
    )

    # DSL instruction with single token api_path
    sass_pyi = Path(os.path.join(out_dir, "nvidia_sass", "__init__.pyi"))
    assert sass_pyi.exists()
    sass_text = sass_pyi.read_text()
    assert "def HMMA16816(d: str) -> Any: ..." in sass_text
