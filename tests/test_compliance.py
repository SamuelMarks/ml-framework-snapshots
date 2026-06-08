"""Tests for the compliance checking module."""

from typing import Dict, Any


import pytest
from pathlib import Path
from ml_framework_snapshots.compliance import (
    get_module_info_from_path,
    extract_target_ast,
    align_namespace,
)
import griffe
from ml_framework_snapshots.compliance import extract_target_refs, score_compliance


def test_get_module_info_from_path_file(tmp_path: Path) -> None:
    """Test getting module info from a python file."""
    # Create a simple structure
    # tmp_path/my_pkg/__init__.py
    # tmp_path/my_pkg/sub_module.py
    pkg_dir = tmp_path / "my_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "sub_module.py"
    sub_mod.write_text("def foo(): pass")

    search_path, mod_name = get_module_info_from_path(str(sub_mod))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg.sub_module"


def test_get_module_info_from_path_dir(tmp_path: Path) -> None:
    """Test getting module info from a directory package."""
    # Create a simple structure
    # tmp_path/my_pkg/__init__.py
    pkg_dir = tmp_path / "my_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()

    search_path, mod_name = get_module_info_from_path(str(pkg_dir))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg"


def test_get_module_info_from_path_init_file(tmp_path: Path) -> None:
    """Test getting module info from an __init__.py file."""
    # Create a simple structure
    # tmp_path/my_pkg/__init__.py
    pkg_dir = tmp_path / "my_pkg"
    pkg_dir.mkdir()
    init_file = pkg_dir / "__init__.py"
    init_file.touch()

    search_path, mod_name = get_module_info_from_path(str(init_file))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg"


def test_get_module_info_from_path_not_exist() -> None:
    """Test getting module info for non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        get_module_info_from_path("does_not_exist_xyz.py")


def test_get_module_info_from_path_value_error(tmp_path: Path) -> None:
    """Test getting module info raises ValueError when module name is empty."""
    # This happens if you pass the root directory
    with pytest.raises(ValueError, match="Could not derive module name"):
        get_module_info_from_path(str(tmp_path))


def test_extract_target_ast(tmp_path: Path) -> None:
    """Test extracting target AST via griffe."""
    file_path = tmp_path / "simple_mod.py"
    file_path.write_text("def my_func(): pass")

    ast_node = extract_target_ast(str(file_path))
    assert isinstance(ast_node, griffe.Module)
    assert ast_node.name == "simple_mod"
    assert "my_func" in ast_node.members


def test_align_namespace() -> None:
    """Test namespace alignment."""
    assert (
        align_namespace("ml_switcheroo.jax.nn", "ml_switcheroo.jax", "jax") == "jax.nn"
    )
    assert align_namespace("ml_switcheroo.jax", "ml_switcheroo.jax", "jax") == "jax"
    assert align_namespace("other.module", "ml_switcheroo.jax", "jax") == "other.module"


def test_get_module_info_from_path_no_py_ext(tmp_path: Path) -> None:
    """Test getting module info from a file without .py extension."""
    pkg_dir = tmp_path / "my_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "sub_module"
    sub_mod.write_text("def foo(): pass")

    search_path, mod_name = get_module_info_from_path(str(sub_mod))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg.sub_module"


def test_extract_target_refs(tmp_path: Path) -> None:
    """Test extracting target refs dynamically."""
    pkg_dir = tmp_path / "mock_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    sub_mod.write_text("def my_func(a: int) -> int: return a")

    refs = extract_target_refs(str(sub_mod), "mock_pkg.api", "reference")
    assert len(refs) == 1
    assert refs[0].api_path == "reference.my_func"
    assert refs[0].name == "my_func"
    assert refs[0].params[0].name == "a"


def test_score_compliance_empty() -> None:
    """Test compliance scoring with empty reference."""
    ref_snap: Dict[str, Any] = {"categories": {}}
    res = score_compliance(ref_snap, [])
    assert res["score_percentage"] == 0.0


def test_score_compliance_matches() -> None:
    """Test compliance scoring with matches and missing."""
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    ref_snap = {
        "categories": {
            "LAYER": [
                {
                    "name": "func1",
                    "api_path": "reference.func1",
                    "kind": "function",
                    "params": [
                        {
                            "name": "a",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "int",
                        }
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "name": "func2",
                    "api_path": "reference.func2",
                    "kind": "function",
                    "params": [],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
            ]
        }
    }

    target_refs = [
        GhostRef(
            name="func1",
            api_path="reference.func1",
            kind="function",
            params=[
                GhostParam(name="a", kind="POSITIONAL_OR_KEYWORD", annotation="int")
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        )
    ]

    res = score_compliance(ref_snap, target_refs)
    assert res["total_endpoints"] == 2
    assert res["score_percentage"] == 50.0
    assert "reference.func1" in res["matched"]
    assert "reference.func2" in res["missing"]


def test_score_compliance_mismatch() -> None:
    """Test compliance scoring with mismatch and varargs fallback."""
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    ref_snap = {
        "categories": {
            "LAYER": [
                {
                    "name": "func1",
                    "api_path": "reference.func1",
                    "kind": "function",
                    "params": [
                        {
                            "name": "a",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "int",
                        }
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "name": "func2",
                    "api_path": "reference.func2",
                    "kind": "function",
                    "params": [
                        {
                            "name": "b",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "int",
                        }
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
            ]
        }
    }

    target_refs = [
        GhostRef(
            name="func1",
            api_path="reference.func1",
            kind="function",
            params=[
                GhostParam(name="wrong", kind="POSITIONAL_OR_KEYWORD", annotation="str")
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
        GhostRef(
            name="func2",
            api_path="reference.func2",
            kind="function",
            params=[
                GhostParam(name="args", kind="VAR_POSITIONAL"),
                GhostParam(name="kwargs", kind="VAR_KEYWORD"),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
    ]

    res = score_compliance(ref_snap, target_refs)
    assert res["total_endpoints"] == 2
    assert "reference.func1" not in res["matched"]
    assert "reference.func2" in res["matched"]  # because varargs fallback
    assert len(res["mismatched"]) == 1
    assert res["mismatched"][0]["api_path"] == "reference.func1"


def test_extract_target_refs_import_error(tmp_path: Path) -> None:
    """Test extracting target refs skips bad imports."""
    pkg_dir = tmp_path / "bad_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    # missing_module doesn't exist
    sub_mod.write_text("import missing_module\ndef my_func(): pass")

    refs = extract_target_refs(str(sub_mod), "bad_pkg.api", "reference")
    # Actually Griffe might still parse it, but dynamic import will fail.
    # It should silently skip.
    assert len(refs) == 0


def test_extract_target_refs_aliases(tmp_path: Path) -> None:
    """Test extraction with aliases mapping correctly."""
    from ml_switcheroo_ir.schema.ghost import GhostRef

    ref_snap = {
        "categories": {
            "LAYER": [
                {
                    "name": "func1",
                    "api_path": "reference.func1",
                    "kind": "function",
                    "params": [],
                    "docstring": "",
                    "aliases": ["reference.alias1"],
                    "returns_type": "None",
                }
            ]
        }
    }

    target_refs = [
        GhostRef(
            name="func1",
            api_path="reference.alias1",
            kind="function",
            params=[],
            docstring="",
            aliases=[],
            returns_type="None",
        )
    ]

    res = score_compliance(ref_snap, target_refs)
    assert res["total_endpoints"] == 1
    assert res["score_percentage"] == 100.0


def test_extract_target_refs_skip_private(tmp_path: Path) -> None:
    """Test extracting target refs skips private members."""
    pkg_dir = tmp_path / "priv_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    sub_mod.write_text("def _private_func(): pass\nclass _PrivateClass: pass")

    refs = extract_target_refs(str(sub_mod), "priv_pkg.api", "reference")
    assert len(refs) == 0


def test_extract_target_refs_nested_import_error(tmp_path: Path) -> None:
    """Test extracting target refs fails on nested import cleanly."""
    pkg_dir = tmp_path / "nested_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    # Create something that looks like an attribute but actually triggers ImportError logic
    # that doesn't resolve to a live object.
    # In `extract_target_refs`, if import fails we fallback to getattr.
    # We mock it by not instantiating the object, but letting griffe parse it.

    # Actually just a normal failure that is caught in `except Exception:` block:
    # We can mock this by trying to extract a fake file
    sub_mod.write_text(
        "import sys\nsys.modules['nested_pkg.api.broken'] = None\ndef broken(): pass"
    )
    refs = extract_target_refs(str(sub_mod), "nested_pkg.api", "reference")

    # We only care that it doesn't crash.
    assert isinstance(refs, list)


def test_extract_target_refs_sys_path(tmp_path: Path) -> None:
    """Test extracting target refs when search path is already in sys.path."""
    import sys

    pkg_dir = tmp_path / "sys_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    sub_mod.write_text("def my_func(): pass")

    sys.path.insert(0, str(tmp_path))
    refs = extract_target_refs(str(sub_mod), "sys_pkg.api", "reference")
    assert len(refs) == 1
    sys.path.remove(str(tmp_path))


def test_extract_target_refs_empty_members(tmp_path: Path) -> None:
    """Test extracting from a module with no members."""
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()

    refs = extract_target_refs(str(pkg_dir), "empty_pkg", "reference")
    assert len(refs) == 0


def test_extract_target_refs_catch_all_exception(tmp_path: Path) -> None:
    """Test extracting target refs skips items that raise general exceptions."""
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = tmp_path / "broken_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    sub_mod.write_text(
        "class BrokenClass:\n    @property\n    def prop(self): raise RuntimeError('Broken')"
    )

    refs = extract_target_refs(str(sub_mod), "broken_pkg.api", "reference")
    # Griffe parses it, GhostInspector will try to inspect and might fail, but it's wrapped in a general catch block.
    # It should survive.
    assert isinstance(refs, list)


def test_extract_target_refs_catch_inner_exception(tmp_path) -> None:
    """Test extracting target refs fails cleanly on inner node walk."""
    from ml_framework_snapshots.compliance import extract_target_refs
    from unittest.mock import patch

    pkg_dir = tmp_path / "inner_broken_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    # Create an invalid state for Griffe so dynamic import fails but walk continues
    sub_mod.write_text("class Outer:\n    class Inner:\n        pass\n")

    import ml_framework_snapshots.models

    orig_inspect = ml_framework_snapshots.models.GhostInspector.inspect

    def mock_inspect(obj, api_path, is_public=None):
        """Mock inspect to test inner node failure.

        Args:
            obj: The object.
            api_path: The api path.
            is_public: Public flag.
        """
        if "Inner" in api_path:
            raise RuntimeError("inner broken")
        return orig_inspect(obj, api_path, is_public)

    with patch.object(
        ml_framework_snapshots.models.GhostInspector, "inspect", mock_inspect
    ):
        refs = extract_target_refs(str(sub_mod), "inner_broken_pkg.api", "reference")
        # Should have extracted Outer, skipped Inner due to exception
        assert len(refs) == 1


def test_extract_target_refs_no_parts(tmp_path) -> None:
    """Test extracting target refs handles paths with no inner parts (just module)."""
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = tmp_path / "top_level_pkg"
    pkg_dir.mkdir()
    sub_mod = pkg_dir / "__init__.py"
    sub_mod.write_text("def top_func(): pass")

    refs = extract_target_refs(str(sub_mod), "top_level_pkg", "reference")
    assert len(refs) == 1


def test_extract_target_refs_break_loop(tmp_path) -> None:
    """Test extracting target refs loop breaking logic."""
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = tmp_path / "break_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    # This will have parts "break_pkg", "api", "MyClass"
    sub_mod.write_text("class MyClass:\n    pass")

    refs = extract_target_refs(str(sub_mod), "break_pkg.api", "reference")
    # Loop should evaluate `parts` fully.
    # It hits `except ImportError:` when it hits `MyClass`, sets object, and `break` is called.
    assert len(refs) == 1


def test_extract_target_refs_continue_loop(tmp_path) -> None:
    """Test extracting target refs handles the loop fully exiting."""
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = tmp_path / "cont_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    # Provide a function that gets completely imported but isn't part of the target loop break
    sub_mod.write_text("import sys\ndef func(): pass")

    refs = extract_target_refs(str(sub_mod), "cont_pkg.api", "reference")
    # if `importlib.import_module` works for ALL parts, then `for i in range` finishes
    # without `break` and doesn't trigger `except ImportError`.
    # Let's ensure the `except ImportError` isn't required for correct execution.
    assert isinstance(refs, list)
