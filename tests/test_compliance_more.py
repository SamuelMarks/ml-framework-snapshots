"""Module docstring."""

from typing import Any


def test_align_namespace_exact_mapping() -> None:
    """Function docstring."""
    from ml_framework_snapshots.compliance import align_namespace

    assert align_namespace("zero_jax", "target_prefix", "ref") == "jax"
    assert align_namespace("zero_optax", "target_prefix", "ref") == "optax"


def test_score_compliance_sig_tuple_edge_cases() -> None:
    """Function docstring."""
    from ml_framework_snapshots.compliance import score_compliance
    from ml_switcheroo_ir.schema.ghost import GhostRef, GhostParam

    ref_snap = {
        "categories": {
            "test": [
                {
                    "api_path": "ref.func1",
                    "name": "func1",
                    "kind": "function",
                    "params": [
                        {
                            "name": "a",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "'```(None)```'",
                            "annotation": '"str"',
                        },
                        {
                            "name": "b",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "```(None)```",
                            "annotation": "'int'",
                        },
                        {
                            "name": "c",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "'___NONE___'",
                            "annotation": "float",
                        },
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                }
            ]
        }
    }
    target_refs = [
        GhostRef(
            api_path="ref.func1",
            name="func1",
            kind="function",
            params=[
                GhostParam(
                    name="a",
                    kind="POSITIONAL_OR_KEYWORD",
                    default="None",
                    annotation="str",
                ),
                GhostParam(
                    name="b",
                    kind="POSITIONAL_OR_KEYWORD",
                    default="None",
                    annotation="int",
                ),
                GhostParam(
                    name="c",
                    kind="POSITIONAL_OR_KEYWORD",
                    default="None",
                    annotation="float",
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        )
    ]
    res = score_compliance(ref_snap, target_refs)
    assert "ref.func1" in res["matched"]


def test_score_compliance_varargs_fallback_special_cases() -> None:
    """Function docstring."""
    from ml_framework_snapshots.compliance import score_compliance
    from ml_switcheroo_ir.schema.ghost import GhostRef, GhostParam

    ref_snap = {
        "categories": {
            "test": [
                {
                    "api_path": "chex.assert_shape",
                    "name": "assert_shape",
                    "kind": "function",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                            "annotation": None,
                        },
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "api_path": "jax.nn.relu",
                    "name": "relu",
                    "kind": "function",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                            "annotation": None,
                        },
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
            api_path="chex.assert_shape",
            name="assert_shape",
            kind="function",
            params=[
                GhostParam(
                    name="x",
                    kind="POSITIONAL_OR_KEYWORD",
                    default=None,
                    annotation=None,
                ),
                GhostParam(
                    name="args", kind="VAR_POSITIONAL", default=None, annotation=None
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
        GhostRef(
            api_path="jax.nn.relu",
            name="relu",
            kind="function",
            params=[
                GhostParam(
                    name="x",
                    kind="POSITIONAL_OR_KEYWORD",
                    default=None,
                    annotation=None,
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
    ]

    res = score_compliance(ref_snap, target_refs)
    assert "chex.assert_shape" in res["matched"]
    assert "jax.nn.relu" in res["matched"]


def test_extract_target_refs_string_path(tmp_path: Any) -> None:
    """Function docstring."""
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = tmp_path / "str_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    sub_mod.write_text("def my_func(a: int) -> None: pass")

    refs = extract_target_refs(str(sub_mod), "str_pkg.api", "ref")  # type: ignore
    assert len(refs) == 1
    assert refs[0].api_path == "ref.my_func"


def test_get_module_info_from_path_no_src_dir(tmp_path: Any) -> None:
    """Function docstring."""
    from ml_framework_snapshots.compliance import get_module_info_from_path

    # mod_name will be empty if we point to a root dir without __init__.py
    path, mod = get_module_info_from_path(str(tmp_path), target_prefix="my_pkg")
    assert path == str(tmp_path)
    assert mod == "my_pkg"


def test_extract_target_refs_list_path(tmp_path: Any) -> None:
    """Function docstring."""
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = tmp_path / "list_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()
    sub_mod = pkg_dir / "api.py"
    sub_mod.write_text("def list_func(): pass")

    refs = extract_target_refs([str(sub_mod)], "list_pkg.api", "ref")
    assert len(refs) == 1
    assert refs[0].api_path == "ref.list_func"
