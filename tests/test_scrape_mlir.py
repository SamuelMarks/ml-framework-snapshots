"""Tests for the MLIR documentation scraper."""

from typing import Any
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
    """Test parsing a mock MLIR dialect page."""
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
    """Test scraping a mock StableHLO markdown doc."""
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
    """Test the main entrypoint via HTML scraping fallback."""
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
    """Test the main entrypoint when live inspection discovers operations."""
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
    """Test inspect_mlir_python_module with mocked modules."""

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
    """Test main() with empty TableGen responses, duplicate ops, live ops, and stablehlo duplicates."""
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
