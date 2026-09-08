"""Tests for the MLIR documentation scraper."""

import json
from typing import Any
from unittest import mock
from ml_framework_snapshots.tools import scrape_mlir


def test_parse_table() -> None:
    """Test parsing a table for operands/attributes."""
    html = """
    <table>
    <tbody>
    <tr><td><code>lhs</code></td><td>floating-point-like</td></tr>
    <tr><td><code>rhs</code></td><td>floating-point-like</td></tr>
    <tr><td>&amp;nbsp;</td><td>empty</td></tr>
    <tr></tr>
    </tbody>
    </table>
    """
    cols = scrape_mlir.parse_table(html, 2)
    assert cols == ["lhs", "rhs"]


def test_parse_dialect_page(mocker: Any) -> None:
    """Test parsing a mock MLIR dialect page.

    Args:
        mocker: Pytest mocker fixture.
    """
    html = """
    <h3 id="arithaddf-arithaddfop"><code>arith.addf</code> (arith::AddFOp)</h3>
    <p><em>Floating point addition operation</em></p>
    <h4>Operands:</h4>
    <table>
    <tbody><tr><td><code>lhs</code></td><td>type</td></tr></tbody>
    </table>
    <h4>Attributes:</h4>
    <table>
    <tbody><tr><td><code>fastmath</code></td><td>type</td></tr></tbody>
    </table>
    <h3 id="no-header">No Header Here</h3>
    <h3 id="minimal"><code>arith.min</code> (arith::MinOp)</h3>
    """
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.fetch_html", return_value=html
    )
    ops = scrape_mlir.parse_dialect_page("http://fake", "arith")
    assert len(ops) == 2
    assert ops[0]["api_path"] == "arith.addf"
    assert ops[0]["class_name"] == "AddFOp"
    assert ops[0]["operands"] == ["lhs"]
    assert ops[0]["attributes"] == ["fastmath"]
    assert ops[0]["description"] == "Floating point addition operation"
    assert ops[1]["class_name"] == "MinOp"
    assert ops[1]["operands"] == []
    assert ops[1]["attributes"] == []

    # Test empty page and page without h3 tags
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.fetch_html",
        return_value="No h3 header",
    )
    assert scrape_mlir.parse_dialect_page("http://fake", "empty") == []


def test_scrape_stablehlo(mocker: Any) -> None:
    """Test scraping a mock StableHLO markdown doc.

    Args:
        mocker: Pytest mocker fixture.
    """
    md = """
# Spec
### functions
Ignored section.
### INVALID-NAME
Ignored op.
### abs
#### Semantics

# Comment to ignore
Does an abs operation.

#### Inputs

| Label | Name      | Type |
|-------|-----------|------|
| (I1)  | `operand` | tensor |
| direct_operand | tensor |
| | tensor |
|---|---|---|

#### Outputs
"""
    mocker.patch("ml_framework_snapshots.tools.scrape_mlir.fetch_html", return_value=md)
    ops = scrape_mlir.scrape_stablehlo()
    assert len(ops) == 1
    assert ops[0]["api_path"] == "stablehlo.abs"
    assert ops[0]["class_name"] == "AbsOp"
    assert ops[0]["operands"] == ["operand", "direct_operand"]
    assert ops[0]["description"] == "Does an abs operation."


def test_main_docs(mocker: Any) -> None:
    """Test the main entrypoint via HTML scraping fallback.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.inspect_mlir_python_module",
        return_value=[],
    )
    # Quoted and unquoted hrefs to test both branches
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.fetch_html",
        return_value="""<a href="/docs/Dialects/ArithOps/">'arith' Dialect</a>""",
    )
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.parse_dialect_page", return_value=[]
    )
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.scrape_stablehlo", return_value=[]
    )
    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    scrape_mlir.main()
    mock_open.assert_called_once()

    # Second pass with unquoted href
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.fetch_html",
        return_value="""<a href=/docs/Dialects/ArithOps/>'arith' Dialect</a>""",
    )
    scrape_mlir.main()


def test_main_live_inspection(mocker: Any) -> None:
    """Test the main entrypoint when live inspection discovers operations.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.inspect_mlir_python_module",
        return_value=[{"api_path": "arith.addf"}],
    )
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    scrape_mlir.main()
    mock_open.assert_called_once()


def test_parse_mlir_tablegen() -> None:
    """Test parsing MLIR TableGen ODS definitions into structured operations."""
    sample_tablegen = """
def Arith_AddFOp : Arith_Op<"addf", [Commutative, Pure]> {
  let summary = "Floating point add";
  let arguments = (ins
    FloatType:$lhs,
    FloatType:$rhs,
    FastMathFlagsAttr:$fastmath
  );
  let results = (outs
    FloatType:$result
  );
}

def Linalg_GenericOp : Linalg_Op<"generic", []> {
  let summary = "Generic structured operation";
  let arguments = (ins
    Variadic<AnyShaped>:$inputs,
    Variadic<AnyShaped>:$outputs
  );
  let results = (outs
    Variadic<AnyRankedTensor>:$results
  );
  let regions = (region
    SizedRegion<1>:$region
  );
}

def Empty_Op : Op<"empty"> {
  let summary = "Empty op";
}

def TrailingOp : Op<"trailing"> {
  let arguments = (ins FloatType:no_dollar, FloatType, );
  let results = (outs FloatType, );
  let regions = (region single_reg, );
}
"""
    ops = scrape_mlir.parse_mlir_tablegen(sample_tablegen, "arith")
    assert len(ops) == 4

    addf = next(o for o in ops if o["class_name"] == "AddFOp")
    assert addf["api_path"] == "arith.addf"
    assert len(addf["operands"]) == 2
    assert len(addf["attributes"]) == 1
    assert addf["attributes"][0]["name"] == "fastmath"
    assert len(addf["results"]) == 1
    assert "Commutative" in addf["traits"]

    generic = next(o for o in ops if o["class_name"] == "GenericOp")
    assert generic["api_path"] == "arith.generic"
    assert "region" in generic["regions"]
    assert len(generic["operands"]) == 2

    empty = next(o for o in ops if o["class_name"] == "Op")
    assert len(empty["operands"]) == 0
    assert len(empty["results"]) == 0

    trailing = next(o for o in ops if o["class_name"] == "TrailingOp")
    assert len(trailing["operands"]) == 2
    assert len(trailing["results"]) == 1
    assert len(trailing["regions"]) == 1


def test_inspect_mlir_python_module(mocker: Any) -> None:
    """Test inspect_mlir_python_module with mocked modules.

    Args:
        mocker: Pytest mocker fixture.
    """

    class MockAddFOp:
        """AddF op docstring."""

        OPERATION_NAME = "arith.addf"

    class MockDialectModule:
        """Mock dialect module with dir method."""

        AddFOp = MockAddFOp

        def __dir__(self) -> list[str]:
            """Return attributes of the mock dialect module.

            Returns:
                List of member attribute names.
            """
            return ["AddFOp", "_internal"]

    mock_mod = MockDialectModule()

    def mock_import(name: str) -> Any:
        """Mock import_module for MLIR dialects.

        Args:
            name: The module name.

        Returns:
            Mocked dialect module or raises ImportError.

        Raises:
            ImportError: If the module is not mocked.
        """
        if name == "mlir.dialects.arith":
            return mock_mod
        raise ImportError(f"No module named {name}")

    mocker.patch("importlib.import_module", side_effect=mock_import)

    ops = scrape_mlir.inspect_mlir_python_module(dialects=["arith", "math"])
    assert len(ops) == 1
    assert ops[0]["class_name"] == "AddFOp"
    assert ops[0]["api_path"] == "arith.addf"


def test_tablegen_helpers() -> None:
    """Test TableGen utility functions for stripping comments and parsing arguments."""
    src = "/* comment */ let x = 1; // line comment\nlet y = 2;"
    clean = scrape_mlir.strip_tablegen_comments(src)
    assert "/*" not in clean
    assert "//" not in clean
    assert "let y = 2;" in clean

    operands, attributes = scrape_mlir.parse_tablegen_args(
        "FloatLike:$lhs, FloatLike:$rhs, FastMathFlagsAttr:$fastmath, I64Attr:$dim, anonymous"
    )
    assert len(operands) == 3
    assert operands[0]["name"] == "lhs"
    assert len(attributes) == 2
    assert attributes[0]["name"] == "fastmath"

    empty_ops, empty_attrs = scrape_mlir.parse_tablegen_args(None)
    assert empty_ops == []
    assert empty_attrs == []

    results = scrape_mlir.parse_tablegen_results("FloatLike:$res1, FloatLike, :$res3")
    assert len(results) == 3
    assert results[0]["name"] == "res1"

    assert scrape_mlir.parse_tablegen_results(None) == []

    regions = scrape_mlir.parse_tablegen_regions("Region:$body, anonymous_reg, :$loop")
    assert len(regions) == 3
    assert regions[0] == "body"

    assert scrape_mlir.parse_tablegen_regions(None) == []


def test_tablegen_inheritance_resolution() -> None:
    """Test that defs inherit arguments and results from superclasses defined with Arguments/Results."""
    td_source = """
    class BaseMathOp<string mnem> :
        Op<mnem>,
        Arguments<(ins FloatType:$lhs, FloatType:$rhs)>,
        Results<(outs FloatType:$result)>;

    def DerivedAddOp : BaseMathOp<"derived_add"> {
      let summary = "Derived add operation";
    }
    """
    ops = scrape_mlir.parse_mlir_tablegen(td_source, "math")
    assert len(ops) == 1
    derived = ops[0]
    assert derived["api_path"] == "math.derived_add"
    assert len(derived["operands"]) == 2
    assert derived["operands"][0]["name"] == "lhs"
    assert len(derived["results"]) == 1
    assert derived["results"][0]["name"] == "result"


def test_tablegen_parsing_missing_branches() -> None:
    """Test parse_tablegen_args, results, and regions edge-case formatting branches."""
    # Empty token in args and empty arg name after colon (lines 162, 174)
    ops, attrs = scrape_mlir.parse_tablegen_args(
        "FloatType:$lhs, , FloatType:, FastMathAttr:$"
    )
    assert len(ops) == 2
    assert len(attrs) == 1

    # Empty token and single colon without dollar in results (lines 231, 235-236)
    res = scrape_mlir.parse_tablegen_results("FloatType:$res1, , FloatType:res2")
    assert len(res) == 2
    assert res[1]["name"] == "res2"

    # Single colon without dollar in regions (line 264)
    regions = scrape_mlir.parse_tablegen_regions("Region:body_reg")
    assert regions == ["body_reg"]


def test_parse_mlir_tablegen_mnemonic_traits_and_unary() -> None:
    """Test parse_mlir_tablegen with let mnemonic, body traits, and Unary base fallback."""
    td_source = """
    def Arith_ExplicitMnemonicOp : Op<"ignored_first_arg"> {
      let mnemonic = "explicit_add";
      let traits = [Pure, Commutative];
      let arguments = (ins FloatType:$in);
      let results = (outs FloatType:$out);
    }

    def Arith_UnaryFallbackOp : Unary<"unary_op"> {
      let summary = "Unary fallback test";
    }
    """
    ops = scrape_mlir.parse_mlir_tablegen(td_source, "arith")
    assert len(ops) == 2

    explicit_op = next(o for o in ops if o["class_name"] == "ExplicitMnemonicOp")
    assert explicit_op["api_path"] == "arith.explicit_add"
    assert "Pure" in explicit_op["traits"]
    assert "Commutative" in explicit_op["traits"]

    unary_op = next(o for o in ops if o["class_name"] == "UnaryFallbackOp")
    assert len(unary_op["operands"]) == 1
    assert unary_op["operands"][0]["name"] == "in"


def test_main_tablegen_empty_and_duplicate_ops(mocker: Any) -> None:
    """Test main() with empty TableGen responses, duplicate ops, live ops, and stablehlo duplicates.

    Args:
        mocker: Pytest mocker fixture.
    """
    # Simulate fetch_html returning empty string for some URLs (branch 634 -> 631)
    # and duplicate ops in parsed TableGen (branch 637 -> 636)
    td_with_dups = """
    def Dialect_OpA : Op<"op_a"> { let summary = "Op A"; }
    def Dialect_OpADup : Op<"op_a"> { let summary = "Op A duplicate"; }
    """

    call_count = [0]

    def mock_fetch(url: str) -> str:
        """Mock fetching TableGen URL returning empty then duplicates.

        Args:
            url: Requested URL.

        Returns:
            TableGen string content.
        """
        call_count[0] += 1
        if call_count[0] == 1:
            return ""  # Empty content
        return td_with_dups

    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.fetch_html", side_effect=mock_fetch
    )

    # inspect_mlir_python_module returns an unseen op (lines 645-646)
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.inspect_mlir_python_module",
        return_value=[{"api_path": "arith.live_unique_op", "name": "live_unique_op"}],
    )

    # scrape_stablehlo returns a duplicate op (arith.live_unique_op) already in seen_paths (branch 677 -> 676)
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.scrape_stablehlo",
        return_value=[{"api_path": "arith.live_unique_op", "name": "live_unique_op"}],
    )

    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    scrape_mlir.main()
    mock_open.assert_called()


def test_parse_mlir_tablegen_sized_segments() -> None:
    """Test TableGen extraction of AttrSizedOperandSegments and AttrSizedResultSegments traits and attributes."""
    content = """
    class Op<string mnem, list<string> traits = []>;

    def SizedSegmentsOp : Op<"sized_op", ["AttrSizedOperandSegments", "AttrSizedResultSegments"]> {
      let arguments = (ins AnyType:$arg0);
      let results = (outs AnyType:$res0);
    }
    """
    ops = scrape_mlir.parse_mlir_tablegen(content, "test")
    assert len(ops) == 1
    attr_names = [a["name"] for a in ops[0]["attributes"]]
    assert "operand_segment_sizes" in attr_names
    assert "result_segment_sizes" in attr_names
    assert "AttrSizedOperandSegments" in ops[0]["traits"]
    assert "AttrSizedResultSegments" in ops[0]["traits"]


def test_parse_mlir_tablegen_existing_sized_segments_attributes() -> None:
    """Test TableGen extraction when operand_segment_sizes and result_segment_sizes already exist."""
    content = """
    class Op<string mnem, list<string> traits = []>;

    def ExistingSegmentsOp : Op<"existing_op", [AttrSizedOperandSegments, AttrSizedResultSegments]> {
      let arguments = (ins DenseI32ArrayAttr:$operand_segment_sizes, DenseI32ArrayAttr:$result_segment_sizes);
      let results = (outs AnyType:$res0);
    }
    """
    ops = scrape_mlir.parse_mlir_tablegen(content, "test")
    assert len(ops) == 1
    attr_names = [a["name"] for a in ops[0]["attributes"]]
    assert attr_names.count("operand_segment_sizes") == 1
    assert attr_names.count("result_segment_sizes") == 1
    assert "AttrSizedOperandSegments" in ops[0]["traits"]
    assert "AttrSizedResultSegments" in ops[0]["traits"]


def test_tablegen_ast_parser() -> None:
    """Test TableGenASTParser handling multiclass, defm, foreach, and string concatenations."""
    tblgen_code = """
    // Multiclass template
    multiclass BinaryOps<string suffix> {
      def NAME # "_" # suffix : Op<NAME # "_" # suffix>;
    }

    // Foreach unrolling
    foreach dt = ["f32", "f64"] in {
      defm Add # "_" # dt : BinaryOps<dt>;
    }

    // Unknown multiclass fallback
    defm Custom : UnknownMulticlass;
    """
    parser = scrape_mlir.TableGenASTParser(tblgen_code)
    expanded = parser.expand()
    assert "Add_f32_f32 : Op<Add_f32_f32>" in expanded
    assert "Add_f64_f64 : Op<Add_f64_f64>" in expanded
    assert "defm Custom : UnknownMulticlass" in expanded

    # Test standalone concatenation
    assert parser._resolve_concatenations('"foo" # "bar"') == '"foobar"'
    assert parser._resolve_concatenations('Ident # "bar"') == "Identbar"
    assert parser._resolve_concatenations('"foo" # Ident') == "fooIdent"
    assert parser._resolve_concatenations("Id1 # Id2") == "Id1Id2"

    # Test nested braces inside multiclass body
    nested_mc = """
    multiclass NestedMC {
      def NestedOp {
        let nested = { 1, 2 };
      }
    }
    """
    nested_parser = scrape_mlir.TableGenASTParser(nested_mc)
    nested_parser.expand()
    assert "NestedMC" in nested_parser.multiclasses


def test_parse_llvm_tblgen_json() -> None:
    """Test parse_llvm_tblgen_json parsing DAG and list records from llvm-tblgen."""
    mock_json = {
        "!instanceof": {"Op": ["Arith_AddFOp", "Arith_SubFOp"]},
        "Arith_AddFOp": {
            "!name": "Arith_AddFOp",
            "!superclasses": ["Arith_Op", "Op"],
            "mnemonic": "addf",
            "summary": "Floating point addition",
            "description": "Performs element-wise float addition.",
            "arguments": {
                "kind": "dag",
                "args": [
                    [{"def": "FloatType"}, "lhs"],
                    [{"def": "FloatType"}, "rhs"],
                    [{"def": "FastMathFlagsAttr"}, "fastmath"],
                    ["custom_tensor", "extra_arg"],
                ],
            },
            "results": {
                "kind": "dag",
                "args": [
                    [{"def": "FloatType"}, "result"],
                    ["raw_type", "extra_res"],
                ],
            },
            "regions": {
                "kind": "dag",
                "args": [[{"def": "Region"}, "body"]],
            },
            "traits": [{"def": "Commutative"}],
        },
        "Arith_ListOp": {
            "!name": "Arith_ListOp",
            "!superclasses": ["Op"],
            "arguments": [
                {"name": "in", "type": "AnyType"},
                {"name": "attr_flag", "type": "BoolAttr"},
                "bare_string_arg",
            ],
            "results": [
                {"name": "out", "type": "AnyType"},
                "bare_string_res",
            ],
            "regions": ["region0"],
            "traits": ["Elementwise"],
        },
        "Arith_RawCustom": {
            "!name": "Arith_RawCustom",
            "!superclasses": ["_Op"],
            "arguments": [],
            "results": [],
            "summary": "Summary without description",
        },
        "NonOpRecord": {
            "!name": "NonOpRecord",
            "!superclasses": ["Dialect"],
        },
    }

    # Test dict input
    ops = scrape_mlir.parse_llvm_tblgen_json(mock_json, "arith")
    assert len(ops) == 3

    addf = next(o for o in ops if o["class_name"] == "AddFOp")
    assert addf["api_path"] == "arith.addf"
    assert len(addf["operands"]) == 3
    assert len(addf["attributes"]) == 1
    assert addf["attributes"][0]["name"] == "fastmath"
    assert len(addf["results"]) == 2
    assert addf["results"][0]["name"] == "result"
    assert addf["regions"] == ["body"]
    assert "Commutative" in addf["traits"]

    list_op = next(o for o in ops if o["class_name"] == "ListOp")
    assert list_op["api_path"] == "arith.list"
    assert len(list_op["operands"]) == 2
    assert len(list_op["attributes"]) == 1
    assert list_op["regions"] == ["region0"]
    assert "Elementwise" in list_op["traits"]

    raw_custom = next(o for o in ops if o["class_name"] == "RawCustomOp")
    assert raw_custom["api_path"] == "arith.rawcustom"
    assert raw_custom["description"] == "Summary without description"

    # Test string input
    import json

    ops_str = scrape_mlir.parse_llvm_tblgen_json(json.dumps(mock_json), "arith")
    assert len(ops_str) == 3


def test_parse_llvm_tblgen_json_edge_branches() -> None:
    """Test parse_llvm_tblgen_json with malformed dag entries, empty traits, non-dict records, and None values."""
    mock_json = {
        "non_dict_entry": "just_a_string",
        "NonOpRecord": {
            "!name": "NonOpRecord",
            "!superclasses": ["Dialect"],
        },
        "Arith_MalformedOp": {
            "!name": "Arith_MalformedOp",
            "!superclasses": ["Arith_Op", "Op"],
            "mnemonic": "malformed",
            "arguments": {
                "kind": "dag",
                "args": [
                    ["only_one"],
                    "not_a_list",
                ],
            },
            "results": {
                "kind": "dag",
                "args": [
                    ["only_one"],
                    "not_a_list",
                ],
            },
            "regions": {
                "kind": "dag",
                "args": [
                    ["only_one"],
                    123,
                ],
            },
            "traits": ["", {"def": ""}],
        },
        "Arith_NoneOp": {
            "!name": "Arith_NoneOp",
            "!superclasses": ["Arith_Op", "Op"],
            "mnemonic": "none_op",
            "arguments": None,
            "results": None,
            "regions": None,
            "traits": None,
        },
    }

    ops = scrape_mlir.parse_llvm_tblgen_json(mock_json, "arith")
    assert len(ops) == 2
    malformed = next(o for o in ops if o["class_name"] == "MalformedOp")
    assert len(malformed["operands"]) == 0
    assert len(malformed["results"]) == 0
    assert len(malformed["traits"]) == 0
    none_op = next(o for o in ops if o["class_name"] == "NoneOp")
    assert len(none_op["operands"]) == 0
    assert len(none_op["results"]) == 0


def test_validate_dialect_ops_ods() -> None:
    """Test validate_dialect_ops_ods validating compliant and invalid operations."""
    valid_ops = [
        {
            "api_path": "arith.addf",
            "class_name": "AddFOp",
            "operands": [{"name": "lhs", "type": "FloatType"}],
            "attributes": [],
            "results": [{"name": "res", "type": "FloatType"}],
        }
    ]
    assert scrape_mlir.validate_dialect_ops_ods(valid_ops, "arith") == []

    invalid_ops = [
        {
            "api_path": "math.sin",
            "operands": "not_a_list",
            "attributes": "not_a_list",
            "results": "not_a_list",
        }
    ]
    errs = scrape_mlir.validate_dialect_ops_ods(invalid_ops, "arith")
    assert any("does not match dialect prefix" in e for e in errs)
    assert any("missing class_name" in e for e in errs)
    assert any("missing valid operands list" in e for e in errs)
    assert any("missing valid attributes list" in e for e in errs)
    assert any("missing valid results list" in e for e in errs)


def test_dump_ast_via_llvm_tooling() -> None:
    """Test dump_ast_via_llvm_tooling with mocked llvm-tblgen executions."""
    # 1. Binary not found
    with mock.patch("shutil.which", return_value=None):
        res_none = scrape_mlir.dump_ast_via_llvm_tooling("test.td", "arith")
        assert res_none is None

    mock_json_stdout = json.dumps(
        {
            "!instanceof": {"Op": ["Arith_AddFOp"]},
            "Arith_AddFOp": {
                "!name": "Arith_AddFOp",
                "!superclasses": ["Arith_Op", "Op"],
                "mnemonic": "addf",
                "summary": "Floating point addition",
                "arguments": {"kind": "dag", "args": []},
                "results": {"kind": "dag", "args": []},
            },
        }
    )

    # 2. Binary found and succeeds
    mock_proc_success = mock.MagicMock()
    mock_proc_success.returncode = 0
    mock_proc_success.stdout = mock_json_stdout
    with mock.patch("subprocess.run", return_value=mock_proc_success):
        res_succ = scrape_mlir.dump_ast_via_llvm_tooling(
            "test.td", "arith", tool_binary="/usr/bin/llvm-tblgen"
        )
        assert res_succ is not None
        assert len(res_succ) == 1
        assert res_succ[0]["api_path"] == "arith.addf"

    # 3. Binary fails with returncode != 0
    mock_proc_fail = mock.MagicMock()
    mock_proc_fail.returncode = 1
    mock_proc_fail.stdout = ""
    with mock.patch("subprocess.run", return_value=mock_proc_fail):
        res_fail = scrape_mlir.dump_ast_via_llvm_tooling(
            "test.td", "arith", tool_binary="/usr/bin/llvm-tblgen"
        )
        assert res_fail is None

    # 4. Binary raises exception
    with mock.patch("subprocess.run", side_effect=OSError("Exec format error")):
        res_err = scrape_mlir.dump_ast_via_llvm_tooling(
            "test.td", "arith", tool_binary="/usr/bin/llvm-tblgen"
        )
        assert res_err is None
