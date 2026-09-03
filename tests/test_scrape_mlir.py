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
    """
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.fetch_html", return_value=html
    )
    ops = scrape_mlir.parse_dialect_page("http://fake", "arith")
    assert len(ops) == 1
    assert ops[0]["api_path"] == "arith.addf"
    assert ops[0]["class_name"] == "AddFOp"
    assert ops[0]["operands"] == ["lhs"]
    assert ops[0]["attributes"] == ["fastmath"]
    assert ops[0]["description"] == "Floating point addition operation"


def test_scrape_stablehlo(mocker: Any) -> None:
    """Test scraping a mock StableHLO markdown doc."""
    md = """
# Spec
### abs
#### Semantics

Does an abs operation.

#### Inputs

| Label | Name      | Type |
|-------|-----------|------|
| (I1)  | `operand` | tensor |

#### Outputs
"""
    mocker.patch("ml_framework_snapshots.tools.scrape_mlir.fetch_html", return_value=md)
    ops = scrape_mlir.scrape_stablehlo()
    assert len(ops) == 1
    assert ops[0]["api_path"] == "stablehlo.abs"
    assert ops[0]["class_name"] == "AbsOp"
    assert ops[0]["operands"] == ["operand"]
    assert ops[0]["description"] == "Does an abs operation."


def test_main(mocker: Any) -> None:
    """Test the main entrypoint."""
    mocker.patch(
        "ml_framework_snapshots.tools.scrape_mlir.fetch_html",
        return_value="""<a href=/docs/Dialects/ArithOps/>'arith' Dialect</a>""",
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
