from typing import Any
"""Module docstring."""

from ml_framework_snapshots.stubs import generate_stubs


def test_generate_stubs(tmp_path: Any) -> None:
    """Function docstring."""
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

    out_dir = tmp_path / "stubs"
    generate_stubs(snap, str(out_dir))

    nn_init = out_dir / "test_fw" / "nn" / "__init__.pyi"
    assert nn_init.exists()

    content = nn_init.read_text()
    assert "class Linear:" in content
    assert (
        "def __init__(self, in_features: int, out_features: int) -> None: ..."
        in content
    )
    assert "class EmptyClass:" in content
    assert "def __init__(self) -> Any: ..." in content

    func_init = out_dir / "test_fw" / "functional" / "__init__.pyi"
    assert func_init.exists()
    content = func_init.read_text()
    assert "def relu(x: Tensor, *args, **kwargs) -> Tensor: ..." in content
    assert "def varargs_func(x: Any, *args: Any) -> Any: ..." in content


def test_generate_stubs_empty(tmp_path: Any) -> None:
    """Function docstring."""
    out_dir = tmp_path / "stubs"
    generate_stubs({}, str(out_dir))
    assert not (out_dir / "test_fw").exists()


def test_generate_stubs_default_vals(tmp_path: Any) -> None:
    """Function docstring."""
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
    out_dir = tmp_path / "stubs"
    generate_stubs(snap, str(out_dir))

    nn_init = out_dir / "test_fw" / "nn" / "__init__.pyi"
    content = nn_init.read_text()
    assert "def __init__(self, in_features: int = 10) -> Any: ..." in content


def test_generate_stubs_no_module(tmp_path: Any) -> None:
    """Function docstring."""
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
    """Function docstring."""
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

    out_dir = tmp_path / "stubs_priv"
    generate_stubs(snap, str(out_dir))

    nn_init = out_dir / "test_fw" / "nn" / "__init__.pyi"
    assert not nn_init.exists()

    generate_stubs(snap, str(out_dir), include_nonpublic=True)
    assert nn_init.exists()
    content = nn_init.read_text()
    assert "class _PrivateLinear:" in content
