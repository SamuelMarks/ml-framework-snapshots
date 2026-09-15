"""Exhaustive test suite for the Anti-Hallucination Grounding SDK."""

import gzip
import json
import os
from typing import Any

from ml_framework_snapshots.grounding import (
    DiagnosticSeverity,
    GroundingDiagnostic,
    GroundingEngine,
    GroundingReport,
    compute_levenshtein,
    validate_mlir_op,
    validate_python_call,
    validate_rdna_instruction,
    validate_sass_instruction,
    validate_stablehlo_op,
)


def test_compute_levenshtein() -> None:
    """Test string distance computation with edge cases."""
    assert compute_levenshtein("", "") == 0
    assert compute_levenshtein("abc", "") == 3
    assert compute_levenshtein("", "xyz") == 3
    assert compute_levenshtein("kitten", "sitting") == 3
    assert compute_levenshtein("stablehlo.matmul", "stablehlo.dot_general") > 0
    assert compute_levenshtein("addf", "addf") == 0


def test_grounding_models() -> None:
    """Test GroundingDiagnostic and GroundingReport models and methods."""
    diag = GroundingDiagnostic(
        field="operands",
        message="Invalid operand count",
        severity=DiagnosticSeverity.ERROR,
        suggested_fix="add",
    )
    assert diag.field == "operands"
    assert diag.severity == DiagnosticSeverity.ERROR
    assert diag.suggested_fix == "add"

    report = GroundingReport(
        is_grounded=True,
        target="stablehlo",
        symbol="stablehlo.add",
    )
    assert not report.has_errors
    assert report.is_grounded

    report.add_diagnostic("info_field", "Informational note", DiagnosticSeverity.INFO)
    assert not report.has_errors
    assert report.is_grounded

    report.add_diagnostic("warn_field", "Warning message", DiagnosticSeverity.WARNING)
    assert not report.has_errors
    assert report.is_grounded

    report.add_diagnostic("err_field", "Error message", DiagnosticSeverity.ERROR)
    assert report.has_errors
    assert not report.is_grounded


def test_grounding_engine_custom_dir(tmp_path: Any) -> None:
    """Test GroundingEngine loading from custom directory with json and json.gz."""
    custom_dir = str(tmp_path / "custom_snaps")
    os.makedirs(custom_dir, exist_ok=True)

    # 1. Plain json snapshot
    plain_data = [
        {
            "name": "custom_op",
            "api_path": "custom.custom_op",
            "kind": "function",
            "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
        }
    ]
    with open(os.path.join(custom_dir, "custom_exhaustive.json"), "w") as f:
        json.dump(plain_data, f)

    # 2. Gzipped json snapshot with categories layout
    gz_data = {
        "categories": {
            "math": [
                {
                    "name": "gz_op",
                    "api_path": "gz.gz_op",
                    "kind": "function",
                    "params": [],
                }
            ]
        }
    }
    with gzip.open(
        os.path.join(custom_dir, "gz_exhaustive.json.gz"), "wt", encoding="utf-8"
    ) as gf:
        json.dump(gz_data, gf)

    # 3. Snapshot with items list
    items_data = {
        "items": [
            {
                "name": "item_op",
                "api_path": "items.item_op",
                "kind": "function",
                "params": [],
            }
        ]
    }
    with open(os.path.join(custom_dir, "items_v1.0.0.json"), "w") as f:
        json.dump(items_data, f)

    # 4. Single item dictionary and items with mnemonic and malformed entries
    single_data = {
        "name": "single_op",
        "api_path": "single.single_op",
        "kind": "function",
        "params": [],
    }
    with open(os.path.join(custom_dir, "single.json"), "w") as f:
        json.dump(single_data, f)

    # 5. File with mnemonics, non-dict entries, and items causing hydrate errors
    try:
        with open(os.path.join(custom_dir, "mixed_exhaustive.json"), "w") as f:
            json.dump(
                [
                    "not_a_dict",
                    {
                        "mnemonic": "INST_M",
                        "api_path": "isa.inst.INST_M",
                        "operands": [["R1", "R2"]],
                    },
                    123,
                ],
                f,
            )
    except Exception:
        pass

    # 6. Non-json file
    with open(os.path.join(custom_dir, "ignored.txt"), "w") as f:
        f.write("ignored")

    engine = GroundingEngine(base_dirs=[custom_dir, "/non_existent_path_xyz"])

    # Verify custom_op
    assert engine.has_symbol("custom", "custom.custom_op")
    assert engine.has_symbol("custom", "custom_op")
    sym = engine.get_symbol("custom", "custom.custom_op")
    assert sym is not None
    assert sym.name == "custom_op"
    # Test short name lookup via get_symbol
    sym_short = engine.get_symbol("custom", "custom_op")
    assert sym_short is not None

    # Test non-existent symbol via get_symbol
    assert engine.get_symbol("custom", "non_existent_sym") is None

    # Verify mixed target with mnemonic
    assert engine.has_symbol("mixed", "INST_M")
    inst_sym = engine.get_symbol("mixed", "INST_M")
    assert inst_sym is not None

    # Verify gz_op
    assert engine.has_symbol("gz", "gz.gz_op")
    assert engine.has_symbol("gz", "gz_op")

    # Verify item_op
    assert engine.has_symbol("items", "items.item_op")

    # Verify single_op
    assert engine.has_symbol("single", "single.single_op")

    # Verify loaded targets
    targets = engine.get_loaded_targets()
    assert "custom" in targets
    assert "gz" in targets
    assert "items" in targets
    assert "single" in targets


def test_grounding_engine_suggestions_and_fuzzy(tmp_path: Any) -> None:
    """Test suggestion and fuzzy searching in GroundingEngine."""
    custom_dir = str(tmp_path / "fuzzy_dir")
    os.makedirs(custom_dir, exist_ok=True)

    data = [
        {
            "name": "dot_general",
            "api_path": "stablehlo.dot_general",
            "kind": "function",
            "params": [],
        },
        {
            "name": "add",
            "api_path": "stablehlo.add",
            "kind": "function",
            "params": [],
        },
        {
            "name": "multiply",
            "api_path": "stablehlo.multiply",
            "kind": "function",
            "params": [],
        },
    ]
    with open(os.path.join(custom_dir, "testtarget_exhaustive.json"), "w") as f:
        json.dump(data, f)

    engine = GroundingEngine(base_dirs=[custom_dir])

    # Test suggest_closest_symbol
    suggested = engine.suggest_closest_symbol("testtarget", "stablehlo.dot_genera")
    assert suggested == "stablehlo.dot_general"

    # Distance too large returns None
    assert (
        engine.suggest_closest_symbol(
            "testtarget", "completely_unrelated_xyz", max_distance=2
        )
        is None
    )

    # Unknown target returns None
    assert engine.suggest_closest_symbol("unknown_target", "abc") is None

    # Test fuzzy search ranking
    results = engine.fuzzy_search_symbol("testtarget", "add")
    assert "stablehlo.add" in results or "add" in results

    # Exact, prefix, substring, and distance matches
    results_dot = engine.fuzzy_search_symbol("testtarget", "dot")
    assert any("dot_general" in r for r in results_dot)

    # Empty target fuzzy search
    assert engine.fuzzy_search_symbol("nonexistent", "abc") == []


def test_validate_sass_instruction() -> None:
    """Test validate_sass_instruction with valid and invalid combinations."""
    # Test valid SASS instruction
    report_valid = validate_sass_instruction(
        mnemonic="FFMA",
        architecture="sm_80",
        operands=["R1", "R2", "R3", "R4"],
        modifiers=[".F32"],
    )
    assert report_valid.is_grounded
    assert report_valid.symbol == "FFMA"

    # Test unrecognized mnemonic with typo suggestion
    report_invalid_mnem = validate_sass_instruction(
        mnemonic="FFM",
        architecture="sm_80",
        operands=["R1", "R2"],
    )
    assert not report_invalid_mnem.is_grounded
    assert any(
        d.field == "mnemonic" and d.suggested_fix == "FFMA"
        for d in report_invalid_mnem.diagnostics
    )

    # Test invalid architecture
    report_bad_arch = validate_sass_instruction(
        mnemonic="WGMMA",
        architecture="sm_70",
        operands=["R1", "R2", "R3", "R4"],
    )
    assert not report_bad_arch.is_grounded
    assert any(d.field == "architecture" for d in report_bad_arch.diagnostics)

    # Test unrecognized modifier
    report_bad_mod = validate_sass_instruction(
        mnemonic="FFMA",
        architecture="sm_80",
        operands=["R1", "R2", "R3", "R4"],
        modifiers=[".NONEXISTENT_MODIFIER"],
    )
    assert any(d.field == "modifiers" for d in report_bad_mod.diagnostics)

    # Test operand count mismatch
    report_bad_ops = validate_sass_instruction(
        mnemonic="FFMA",
        architecture="sm_80",
        operands=["R1"],
    )
    assert not report_bad_ops.is_grounded
    assert any(d.field == "operands" for d in report_bad_ops.diagnostics)


def test_validate_rdna_instruction() -> None:
    """Test validate_rdna_instruction including register alignment checking."""
    # Test valid RDNA instruction with even-aligned register pair
    report_valid = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="GFX11",
        operands=["v[0:1]", "v[2:3]"],
    )
    assert report_valid.is_grounded

    # Test unrecognized mnemonic with suggestion
    report_typo = validate_rdna_instruction(
        mnemonic="V_ADD_F3",
        gfx_arch="GFX11",
        operands=["v0", "v1"],
    )
    assert not report_typo.is_grounded
    assert any(d.field == "mnemonic" for d in report_typo.diagnostics)

    # Test odd-aligned 64-bit register pair violation
    report_odd_reg = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="GFX11",
        operands=["v[1:2]", "v[2:3]"],
    )
    assert not report_odd_reg.is_grounded
    assert any(
        "operands[0]" in d.field and "even-aligned" in d.message
        for d in report_odd_reg.diagnostics
    )


def test_validate_stablehlo_op() -> None:
    """Test validate_stablehlo_op verifying attribute schemas and operand counts."""
    # Test valid dot_general with dot_dimension_numbers
    report_dot = validate_stablehlo_op(
        op_name="dot_general",
        operand_types=["tensor<128x64xf32>", "tensor<64x256xf32>"],
        attributes={"dot_dimension_numbers": {}},
    )
    assert report_dot.is_grounded

    # Test dot_general missing dot_dimension_numbers
    report_dot_missing = validate_stablehlo_op(
        op_name="stablehlo.dot_general",
        operand_types=["tensor<128x64xf32>", "tensor<64x256xf32>"],
        attributes={},
    )
    assert not report_dot_missing.is_grounded
    assert any(
        d.field == "attributes.dot_dimension_numbers"
        for d in report_dot_missing.diagnostics
    )

    # Test convolution missing dimension_numbers
    report_conv_missing = validate_stablehlo_op(
        op_name="stablehlo.convolution",
        operand_types=["tensor<1x28x28x1xf32>", "tensor<3x3x1x32xf32>"],
        attributes={},
    )
    assert not report_conv_missing.is_grounded
    assert any(
        d.field == "attributes.dimension_numbers"
        for d in report_conv_missing.diagnostics
    )

    # Test unrecognized op with suggestion
    report_typo = validate_stablehlo_op(
        op_name="stablehlo.matmul",
        operand_types=["t1", "t2"],
        attributes={},
    )
    assert not report_typo.is_grounded
    assert any(d.field == "operation" for d in report_typo.diagnostics)

    # Test operand count mismatch
    report_op_count = validate_stablehlo_op(
        op_name="stablehlo.add",
        operand_types=["t1"],  # add requires 2 operands
        attributes={},
    )
    assert not report_op_count.is_grounded
    assert any(d.field == "operands" for d in report_op_count.diagnostics)


def test_validate_mlir_op() -> None:
    """Test validate_mlir_op checking dialect operations and operand count."""
    # Test valid arith.addf
    report_valid = validate_mlir_op(
        dialect="arith",
        op_name="addf",
        operand_count=2,
        attributes={},
    )
    assert report_valid.is_grounded

    # Test unrecognized op
    report_unknown = validate_mlir_op(
        dialect="arith",
        op_name="unknown_arith_op_xyz",
        operand_count=2,
        attributes={},
    )
    assert not report_unknown.is_grounded

    # Test operand count mismatch
    report_count = validate_mlir_op(
        dialect="arith",
        op_name="arith.addf",
        operand_count=1,
        attributes={},
    )
    assert not report_count.is_grounded
    assert any(d.field == "operand_count" for d in report_count.diagnostics)


def test_validate_python_call(tmp_path: Any) -> None:
    """Test validate_python_call with mock framework snapshot."""
    custom_dir = str(tmp_path / "py_dir")
    os.makedirs(custom_dir, exist_ok=True)

    torch_data = {
        "categories": {
            "math": [
                {
                    "name": "sum",
                    "api_path": "torch.sum",
                    "kind": "function",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "dim", "kind": "KEYWORD_ONLY"},
                        {"name": "keepdim", "kind": "KEYWORD_ONLY"},
                    ],
                }
            ]
        }
    }
    with open(os.path.join(custom_dir, "torch_v2.0.0.json"), "w") as f:
        json.dump(torch_data, f)

    numpy_data = {
        "categories": {
            "math": [
                {
                    "name": "sum",
                    "api_path": "numpy.sum",
                    "kind": "function",
                    "params": [
                        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "axis", "kind": "KEYWORD_ONLY"},
                    ],
                }
            ]
        }
    }
    with open(os.path.join(custom_dir, "numpy_v1.0.0.json"), "w") as f:
        json.dump(numpy_data, f)

    engine = GroundingEngine(base_dirs=[custom_dir])

    # Valid call
    res_valid = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3]],
        kwargs={"dim": 0, "keepdim": True},
        engine=engine,
    )
    assert res_valid.is_grounded

    # Unrecognized API path
    res_unrec = validate_python_call(
        framework="torch",
        api_path="torch.summ",
        args=[],
        kwargs={},
        engine=engine,
    )
    assert not res_unrec.is_grounded
    assert any(
        d.field == "api_path" and d.suggested_fix == "torch.sum"
        for d in res_unrec.diagnostics
    )

    # Hallucinated axis kwarg on torch -> suggests dim
    res_halluc_axis = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3]],
        kwargs={"axis": 0},
        engine=engine,
    )
    assert not res_halluc_axis.is_grounded
    assert any(
        d.field == "kwargs.axis" and d.suggested_fix == "dim"
        for d in res_halluc_axis.diagnostics
    )

    # Hallucinated dim kwarg on numpy -> suggests axis
    res_halluc_dim = validate_python_call(
        framework="numpy",
        api_path="numpy.sum",
        args=[[1, 2, 3]],
        kwargs={"dim": 0},
        engine=engine,
    )
    assert not res_halluc_dim.is_grounded
    assert any(
        d.field == "kwargs.dim" and d.suggested_fix == "axis"
        for d in res_halluc_dim.diagnostics
    )

    # Hallucinated arbitrary kwarg
    res_rand_kw = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3]],
        kwargs={"nonexistent_param": 123},
        engine=engine,
    )
    assert not res_rand_kw.is_grounded

    # Too many positional arguments
    res_too_many_args = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3], 4, 5],
        kwargs={},
        engine=engine,
    )
    assert not res_too_many_args.is_grounded
    assert any(d.field == "args" for d in res_too_many_args.diagnostics)

    # Overloads (dict & object) and accepted_kwargs coverage
    from ml_switcheroo_ir.schema.ghost import ParameterKind
    from ml_framework_snapshots.models import ExtendedGhostRef, GhostParam, GhostRef

    mock_ref = ExtendedGhostRef(
        name="fn",
        api_path="mock.fn",
        kind="function",
        params=[GhostParam(name="a", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
        overloads=[
            {"params": [{"name": "b"}, {"name": ""}]},
            GhostRef(
                name="fn",
                api_path="mock.fn",
                kind="function",
                params=[GhostParam(name="c", kind=ParameterKind.KEYWORD_ONLY)],
            ),
        ],
        accepted_kwargs=["extra_kw"],
    )

    engine._target_cache["mock"] = {"mock.fn": mock_ref}
    res_ov = validate_python_call(
        framework="mock",
        api_path="mock.fn",
        args=[],
        kwargs={"a": 1, "b": 2, "c": 3, "extra_kw": 4},
        engine=engine,
    )
    assert res_ov.is_grounded


def test_grounding_engine_edge_branches(tmp_path: Any, monkeypatch: Any) -> None:
    """Test edge branches in compiler, hardware, and engine modules."""
    # 1. Compiler: convolution with dimension_numbers present and zero expected operands
    conv_report = validate_stablehlo_op(
        op_name="stablehlo.convolution",
        operand_types=["tensor<1x28x28x1xf32>", "tensor<3x3x1x32xf32>"],
        attributes={"dimension_numbers": {}},
    )
    assert conv_report.is_grounded

    # Mock op with no operands field and empty params
    mock_dir = str(tmp_path / "branch_cov")
    os.makedirs(mock_dir, exist_ok=True)
    with open(os.path.join(mock_dir, "stablehlo_exhaustive.json"), "w") as f:
        json.dump(
            [
                {
                    "name": "noop",
                    "api_path": "stablehlo.noop",
                    "kind": "function",
                    "params": [],
                }
            ],
            f,
        )

    eng = GroundingEngine(base_dirs=[mock_dir])
    noop_report = validate_stablehlo_op("stablehlo.noop", [], {}, engine=eng)
    assert noop_report.is_grounded

    # 2. Hardware: SASS with valid modifiers and RDNA operand combinations
    report_sass_mod = validate_sass_instruction(
        mnemonic="FADD",
        architecture="sm_80",
        operands=["R1", "R2", "R3"],
        modifiers=[".FTZ"],
    )
    assert report_sass_mod.is_grounded

    # Mock SASS instruction with no operand_signatures
    no_sig_dir = str(tmp_path / "no_sig_dir")
    os.makedirs(no_sig_dir, exist_ok=True)
    with open(os.path.join(no_sig_dir, "nvidia_sass_exhaustive.json"), "w") as f:
        json.dump(
            [
                {
                    "mnemonic": "NOSIG",
                    "name": "NOSIG",
                    "api_path": "isa.inst.NOSIG",
                    "kind": "function",
                    "domain_metadata": {"modifiers": [".M1"], "operand_signatures": []},
                    "params": [],
                }
            ],
            f,
        )
    no_sig_eng = GroundingEngine(base_dirs=[no_sig_dir])
    report_no_sig = validate_sass_instruction(
        mnemonic="NOSIG",
        architecture="sm_80",
        operands=["R1", "R2"],
        engine=no_sig_eng,
    )
    assert report_no_sig.is_grounded

    # RDNA: test multiple operand forms: scalar v0, 4-reg range v[0:3], even-aligned v[0:1]
    report_rdna_range = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="GFX11",
        operands=["v0", "v[0:3]", "v[0:1]"],
    )
    assert report_rdna_range.is_grounded

    # 3. Engine: tie-breaking in suggest_closest_symbol, category edge cases, and empty symbol fields
    tie_dir = str(tmp_path / "tie_dir")
    os.makedirs(tie_dir, exist_ok=True)
    tie_data = {
        "categories": {
            "valid": [
                # Short candidate first, longer candidate with same distance second
                {"name": "ab", "api_path": "tie.ab", "kind": "function", "params": []},
                {
                    "name": "abc",
                    "api_path": "tie.abc",
                    "kind": "function",
                    "params": [],
                },
                {"name": "only_name", "api_path": "", "kind": "function", "params": []},
                {
                    "name": "",
                    "api_path": "tie.only_api",
                    "kind": "function",
                    "params": [],
                },
                {
                    "name": "aaaaa",
                    "api_path": "tie.aaaaa",
                    "kind": "function",
                    "params": [],
                },
                {
                    "name": "aaa",
                    "api_path": "tie.aaa",
                    "kind": "function",
                    "params": [],
                },
            ],
            "not_a_list": "invalid_value",
        }
    }
    with open(os.path.join(tie_dir, "tie_exhaustive.json"), "w") as f:
        json.dump(tie_data, f)

    tie_eng = GroundingEngine(base_dirs=[tie_dir])
    # Query 'a': distance is 1 to 'ab' (len 2). Later candidate 'abc' (len 3) has distance 2 (> 1).
    # Query 'a' vs 'ab' and 'abc':
    sugg = tie_eng.suggest_closest_symbol("tie", "a", max_distance=3)
    assert sugg == "ab"

    # Query 'aaaa': distance is 1 for 'aaaaa' (len 5), 1 for 'aaa' (len 3)
    sugg2 = tie_eng.suggest_closest_symbol("tie", "aaaa", max_distance=3)
    assert sugg2 == "aaa"

    # Test short-map case-insensitive lookup via get_symbol and has_symbol
    assert tie_eng.has_symbol("tie", "AAA") is True
    assert tie_eng.get_symbol("tie", "AAA") is not None

    # Test suggestion with mnemonic
    sugg_mnem = no_sig_eng.suggest_closest_symbol("nvidia_sass", "NOSI", max_distance=2)
    assert sugg_mnem == "NOSIG"

    # Test candidate tie-breaking where first candidate is longer and second is shorter
    tie_break_dir = str(tmp_path / "tie_break_dir")
    os.makedirs(tie_break_dir, exist_ok=True)
    with open(os.path.join(tie_break_dir, "tb_exhaustive.json"), "w") as f:
        json.dump(
            [
                {
                    "name": "abcd",
                    "api_path": "tb.abcd",
                    "kind": "function",
                    "params": [],
                },
                {"name": "xy", "api_path": "tb.xy", "kind": "function", "params": []},
            ],
            f,
        )
    tb_eng = GroundingEngine(base_dirs=[tie_break_dir])
    # Query 'ab': distance to 'abcd' is 2 (len 4), distance to 'xy' is 2 (len 2).
    # 'abcd' comes before 'xy' alphabetically, so 'xy' triggers len(cand) < len(best_candidate).
    sugg_tb = tb_eng.suggest_closest_symbol("tb", "ab", max_distance=3)
    assert sugg_tb == "xy"

    # Test raw_data not list/dict in file
    primitive_dir = str(tmp_path / "prim_dir")
    os.makedirs(primitive_dir, exist_ok=True)
    with open(os.path.join(primitive_dir, "prim_v1.0.0.json"), "w") as f:
        f.write('"just_a_string"')
    prim_eng = GroundingEngine(base_dirs=[primitive_dir])
    assert prim_eng.load_target("prim") == {}

    # Test hydrate exception handling
    from ml_framework_snapshots.models import GhostInspector

    orig_hydrate = GhostInspector.hydrate

    def bad_hydrate(d: Any) -> Any:
        """Simulate failure during GhostInspector.hydrate.

        Args:
            d: Dictionary to hydrate.

        Returns:
            Hydrated GhostRef.

        Raises:
            ValueError: If dictionary name is 'fail'.
        """
        if d.get("name") == "fail":
            raise ValueError("simulated hydrate failure")
        return orig_hydrate(d)

    monkeypatch.setattr(GhostInspector, "hydrate", bad_hydrate)
    fail_dir = str(tmp_path / "fail_dir")
    os.makedirs(fail_dir, exist_ok=True)
    with open(os.path.join(fail_dir, "fail_v1.0.0.json"), "w") as f:
        json.dump([{"name": "fail", "kind": "function"}], f)
    fail_eng = GroundingEngine(base_dirs=[fail_dir])
    assert fail_eng.load_target("fail") == {}


def test_grounding_engine_collector_fallback(tmp_path: Any, monkeypatch: Any) -> None:
    """Test GroundingEngine collector fallback when files are absent on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    eng = GroundingEngine(base_dirs=[empty_dir])

    # 1. Fallback succeeds for known collector without files on disk
    html_target = eng.load_target("html_dsl")
    assert len(html_target) > 0
    assert "div" in html_target

    # 2. Unknown target returns empty dict
    assert eng.load_target("nonexistent_framework") == {}

    # 3. Exception during extract_snapshot returns empty dict
    import ml_framework_snapshots.api as api_mod

    def mock_extract_fail(target: str) -> Any:
        """Simulate extraction failure.

        Args:
            target: Target framework name.

        Raises:
            RuntimeError: Simulated failure.
        """
        raise RuntimeError("extraction error")

    monkeypatch.setattr(api_mod, "extract_snapshot", mock_extract_fail)
    eng_fail = GroundingEngine(base_dirs=[empty_dir])
    assert eng_fail.load_target("tikz") == {}

    # 4. Fallback with malformed items and hydrate exception
    def mock_extract_malformed(target: str) -> Any:
        """Return snapshot with malformed items for coverage.

        Args:
            target: Target framework name.

        Returns:
            Snapshot dict with non-dict and faulty items.
        """
        return {
            "categories": {
                "scalar_cat": "not_a_list",
                "tags": [
                    "not_a_dict",
                    {"name": "fail", "kind": "function"},
                    {
                        "name": "valid_tag",
                        "kind": "function",
                        "api_path": "valid_tag",
                        "mnemonic": "VT",
                    },
                    {
                        "name": "",
                        "kind": "function",
                        "api_path": "path_only",
                    },
                    {
                        "name": "name_only",
                        "kind": "function",
                        "api_path": "",
                    },
                ],
            }
        }

    from ml_framework_snapshots.models import GhostInspector

    orig_hydrate = GhostInspector.hydrate

    def bad_hydrate(d: Any) -> Any:
        """Simulate hydrate error for item named fail.

        Args:
            d: Dictionary to hydrate.

        Returns:
            Hydrated GhostRef.

        Raises:
            ValueError: If item is named 'fail'.
        """
        if isinstance(d, dict) and d.get("name") == "fail":
            raise ValueError("simulated error")
        return orig_hydrate(d)

    monkeypatch.setattr(GhostInspector, "hydrate", bad_hydrate)
    monkeypatch.setattr(api_mod, "extract_snapshot", mock_extract_malformed)
    eng_malformed = GroundingEngine(base_dirs=[empty_dir])
    res = eng_malformed.load_target("tikz")
    assert "valid_tag" in res
    assert "VT" in res
    assert "path_only" in res
    assert "name_only" in res

    # 5. Fallback when extract_snapshot returns non-dict or missing categories
    monkeypatch.setattr(api_mod, "extract_snapshot", lambda t: {"no_categories": 123})
    eng_non_dict = GroundingEngine(base_dirs=[empty_dir])
    assert eng_non_dict.load_target("latex_dsl") == {}
