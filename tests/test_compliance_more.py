"""Module docstring."""

from pathlib import Path
import os

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
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "str_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("def my_func(a: int) -> None: pass")

    refs = extract_target_refs(str(sub_mod), "str_pkg.api", "ref")  # type: ignore
    assert len(refs) == 1
    assert refs[0].api_path == "ref.my_func"


def test_get_module_info_from_path_no_src_dir(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    from ml_framework_snapshots.compliance import get_module_info_from_path

    # mod_name will be empty if we point to a root dir without __init__.py
    path, mod = get_module_info_from_path(str(tmp_path), target_prefix="my_pkg")
    assert path == str(tmp_path)
    assert mod == "my_pkg"


def test_extract_target_refs_list_path(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    from ml_framework_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "list_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("def list_func(): pass")

    refs = extract_target_refs([str(sub_mod)], "list_pkg.api", "ref")
    assert len(refs) == 1
    assert refs[0].api_path == "ref.list_func"


def test_check_mlir_text_compliance() -> None:
    """Test check_mlir_text_compliance with valid and invalid MLIR snippets."""
    from ml_framework_snapshots.compliance import check_mlir_text_compliance

    # 1. Valid MLIR snippet
    valid_snippet = """
    // Module containing standard arith ops
    module {
        func.func @test(%a: f32, %b: f32) -> f32 {
            %0 = arith.addf(%a, %b) : f32
            return %0 : f32
        }
    }
    """
    res_valid = check_mlir_text_compliance(valid_snippet)
    assert res_valid["is_compliant"] is True
    assert res_valid["verified_ops"] >= 1

    # 2. Invalid MLIR snippet with non-existent op
    invalid_snippet = """
    %0 = non_existent_dialect.non_existent_op(%a) : i32
    """
    res_invalid = check_mlir_text_compliance(invalid_snippet)
    assert res_invalid["is_compliant"] is False
    assert len(res_invalid["errors"]) > 0

    # 3. Existing op with invalid operand count (branch 404)
    res_bad_count = check_mlir_text_compliance("%0 = arith.addf(%a) : f32")
    assert res_bad_count["is_compliant"] is False
    assert any("Operand count mismatch" in err for err in res_bad_count["errors"])


def test_check_sass_assembly_compliance() -> None:
    """Test check_sass_assembly_compliance with valid, predicated, and invalid SASS instructions."""
    from ml_framework_snapshots.compliance import check_sass_assembly_compliance

    # 1. Valid SASS snippet on sm_80
    valid_sass = """
    /*0010*/ FADD.FTZ R0, R1, R2;
    /*0020*/ @P0 FADD R3, R4, R5;
    """
    res_valid = check_sass_assembly_compliance(valid_sass, sm_arch="sm_80")
    assert res_valid["is_compliant"] is True
    assert res_valid["verified_instructions"] == 2

    # 2. Incompatible instruction for architecture: WGMMA on sm_80
    invalid_sass = """
    WGMMA R0, R1, R2, R3;
    """
    res_invalid = check_sass_assembly_compliance(invalid_sass, sm_arch="sm_80")
    assert res_invalid["is_compliant"] is False
    assert any(
        "WGMMA instructions are strictly supported on sm_90+" in err
        for err in res_invalid["errors"]
    )

    # 3. Comment lines starting with '#' and lines that do not match instruction regex
    sass_comments = "# A comment line\n// Another comment\n@@ invalid label !@#\n\n"
    res_comments = check_sass_assembly_compliance(sass_comments)
    assert res_comments["is_compliant"] is True
    assert res_comments["total_instructions"] == 0


def test_check_rdna_assembly_compliance() -> None:
    """Test check_rdna_assembly_compliance with valid and misaligned RDNA instructions."""
    from ml_framework_snapshots.compliance import check_rdna_assembly_compliance

    # 1. Valid RDNA snippet
    valid_rdna = """
    // Vector addition
    v_add_f32 v0, v1, v2
    """
    res_valid = check_rdna_assembly_compliance(valid_rdna)
    assert res_valid["is_compliant"] is True
    assert res_valid["verified_instructions"] == 1

    # 2. Misaligned 64-bit register pair (odd start index)
    invalid_rdna = """
    v_add_f32 v[1:2], v0, v1
    """
    res_invalid = check_rdna_assembly_compliance(invalid_rdna)
    assert res_invalid["is_compliant"] is False
    assert any("Register alignment error" in err for err in res_invalid["errors"])

    # 3. Comment lines starting with ';' or '#' and lines that do not match instruction regex
    rdna_comments = "; Semicolon comment\n# Hash comment\n!! invalid rdna line !@#\n\n"
    res_comments = check_rdna_assembly_compliance(rdna_comments)
    assert res_comments["is_compliant"] is True
    assert res_comments["total_instructions"] == 0


def test_validate_broadcast_and_matmul_shapes() -> None:
    """Test validate_broadcast_shapes and validate_matmul_shapes rules and error branches."""
    from ml_framework_snapshots.compliance import (
        validate_broadcast_shapes,
        validate_matmul_shapes,
    )

    # 1. Broadcasting
    compat1, res1, _ = validate_broadcast_shapes([2, 3], [1, 3])
    assert compat1 is True
    assert res1 == [2, 3]

    compat2, _, err2 = validate_broadcast_shapes([2, 3], [3, 2])
    assert compat2 is False
    assert "Shape mismatch" in (err2 or "")

    compat3, res3, _ = validate_broadcast_shapes([1, 4, 1], [2, 1, 5])
    assert compat3 is True
    assert res3 == [2, 4, 5]

    # Dynamic / wildcard and non-int dimensions
    compat_dyn, res_dyn, _ = validate_broadcast_shapes(["?"], [4])
    assert compat_dyn is True
    assert res_dyn == [-1]

    compat_str, res_str, _ = validate_broadcast_shapes(["invalid_dim"], [4])
    assert compat_str is True
    assert res_str == [-1]

    # 2. Matmul strict 2D
    compat_s2d, _, err_s2d = validate_matmul_shapes(
        [2, 3, 4], [2, 4, 5], strict_2d=True
    )
    assert compat_s2d is False
    assert "Strict 2D matmul error" in (err_s2d or "")

    # 1D vector dot products
    compat_1d, _, _ = validate_matmul_shapes([4], [4])
    assert compat_1d is True

    compat_1d_sym, _, _ = validate_matmul_shapes(["?"], [4])
    assert compat_1d_sym is True

    compat_1d_sym2, _, _ = validate_matmul_shapes([4], ["?"])
    assert compat_1d_sym2 is True

    compat_1d_sym3, _, _ = validate_matmul_shapes(["?"], ["?"])
    assert compat_1d_sym3 is True

    compat_1d_err, _, err_1d = validate_matmul_shapes([4], [5])
    assert compat_1d_err is False
    assert "dot product mismatch" in (err_1d or "")

    # Rank < 2 error (left < 2 vs right < 2)
    compat_r1, _, err_r1 = validate_matmul_shapes([4], [4, 5])
    assert compat_r1 is False
    assert "requires at least 2D operands" in (err_r1 or "")

    compat_r2, _, err_r2 = validate_matmul_shapes([4, 5], [4])
    assert compat_r2 is False
    assert "requires at least 2D operands" in (err_r2 or "")

    # 2D matmul valid and contracting mismatch
    compat_2d, res_2d, _ = validate_matmul_shapes([2, 3], [3, 4])
    assert compat_2d is True
    assert res_2d == [2, 4]

    compat_2d_sym, res_2d_sym, _ = validate_matmul_shapes([2, "?"], [3, 4])
    assert compat_2d_sym is True

    compat_k_err, _, err_k = validate_matmul_shapes([2, 3], [4, 5])
    assert compat_k_err is False
    assert "contracting dimension mismatch" in (err_k or "")

    # Batch matmul valid and batch mismatch
    compat_bm, res_bm, _ = validate_matmul_shapes([2, 3, 4], [2, 4, 5])
    assert compat_bm is True
    assert res_bm == [2, 3, 5]

    compat_bm_err, _, err_bm = validate_matmul_shapes([2, 3, 4], [3, 4, 5])
    assert compat_bm_err is False
    assert "Batch dimension broadcasting error" in (err_bm or "")
