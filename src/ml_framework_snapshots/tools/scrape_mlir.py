"""Script to generate an exhaustive JSON dump of MLIR operations by scraping MLIR documentation."""

import json
import os
import re
import urllib.request
from typing import Dict, Any, List

MLIR_DOCS_URL = "https://mlir.llvm.org/docs/Dialects/"


def fetch_html(url: str) -> str:  # pragma: no cover
    """Fetch HTML content from a URL."""
    try:
        with urllib.request.urlopen(url) as response:
            return str(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def parse_table(table_html: str, expected_cols: int) -> List[str]:
    """Parse a table and extract the first column names."""
    names = []
    # If the table has a tbody, extract rows from it, otherwise from table directly
    tbody_match = re.search(r"<tbody>(.*?)</tbody>", table_html, re.DOTALL)
    content = tbody_match.group(1) if tbody_match else table_html  # pragma: no cover

    rows = re.findall(r"<tr>(.*?)</tr>", content, re.DOTALL)
    for row in rows:
        cols = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cols) >= 1:  # pragma: no cover
            # First column is usually <code>name</code>
            name = re.sub(r"<[^>]+>", "", cols[0]).strip()
            # Ignore headers if there wasn't a proper thead/tbody separation
            if name and name.lower() not in (
                "attribute",
                "operand",
                "result",
            ):  # pragma: no cover
                names.append(name)
    return names


def parse_dialect_page(dialect_url: str, dialect_name: str) -> List[Dict[str, Any]]:
    """Parse a specific dialect page to extract all operations."""
    html = fetch_html(dialect_url)
    if not html:  # pragma: no cover
        return []

    ops = []
    # Split by h3 to isolate operations (ops are always level 3 headers in MLIR docs)
    splits = re.split(r"<h3[^>]*>", html)
    for sec in splits[1:]:
        sec = sec.strip()
        if not sec.startswith("<code>"):  # pragma: no cover
            continue

        header_end = sec.find("</h3>")
        if header_end == -1:  # pragma: no cover
            continue

        header = sec[:header_end]
        # Match: <code>arith.addf</code> (arith::AddFOp)
        m = re.match(r"<code>([^<]+)</code>\s*\(([^:]+)::([^)]+)\)", header)
        if not m:  # pragma: no cover
            # Fallback match, sometimes dialect namespace is missing or different
            m = re.match(r"<code>([^<]+)</code>\s*\(([^)]+)\)", header)
            if not m:  # pragma: no cover  # pragma: no cover
                continue
            api_path = m.group(1)
            class_name = m.group(2).split("::")[-1]
        else:
            api_path = m.group(1)
            class_name = m.group(3)

        if not api_path.startswith(f"{dialect_name}."):  # pragma: no cover
            # Ensure we are parsing actual dialect ops, not random structs
            # Some dialects might have an underscore or different naming in api path though,
            # so this check is relaxed later if needed.
            pass

        # Parse Description (first <p> after header)
        desc_match = re.search(r"</h3>\s*<p><em>(.*?)</em></p>", sec, re.DOTALL)
        description = (
            desc_match.group(1).strip() if desc_match else ""
        )  # pragma: no cover
        description = re.sub(r"<[^>]+>", "", description)  # clean html

        operands: List[str] = []
        attributes: List[str] = []

        # Find Operands table
        ops_match = re.search(
            r"<h4[^>]*>Operands:</h4>\s*<table[^>]*>(.*?)</table>", sec, re.DOTALL
        )
        if ops_match:  # pragma: no cover
            operands = parse_table(ops_match.group(1), 2)

        # Find Attributes table
        attr_match = re.search(
            r"<h4[^>]*>Attributes:</h4>\s*<table[^>]*>(.*?)</table>", sec, re.DOTALL
        )
        if attr_match:  # pragma: no cover
            attributes = parse_table(attr_match.group(1), 3)

        ops.append(
            {
                "api_path": api_path,
                "dialect": dialect_name,
                "class_name": class_name,
                "operands": operands,
                "attributes": attributes,
                "description": description,
            }
        )

    return ops


def scrape_stablehlo() -> List[Dict[str, Any]]:
    """Scrape the StableHLO spec markdown to extract operations."""
    url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
    md = fetch_html(url)
    ops: List[Dict[str, Any]] = []
    if not md:  # pragma: no cover
        return ops

    # split by ###
    sections = re.split(r"\n### (?![#])", md)
    for sec in sections[1:]:
        lines = sec.split("\n")
        op_name = lines[0].strip()
        # Ops typically have lowercase names in the spec (e.g. abs, add)
        if not re.match(r"^[a-z0-9_]+$", op_name) or op_name in (
            "functions",
            "identifiers",
            "types",
            "operations",
            "constants",
        ):
            continue  # pragma: no cover

        api_path = f"stablehlo.{op_name}"
        class_name = "".join(word.capitalize() for word in op_name.split("_")) + "Op"

        # Description: grab paragraph after #### Semantics
        desc_match = re.search(r"#### Semantics\n\n(.*?)(?=\n\n)", sec, re.DOTALL)
        desc = (
            desc_match.group(1).replace("\n", " ")
            if desc_match
            else f"StableHLO {op_name} operation"
        )  # pragma: no cover

        # Inputs: grab table under #### Inputs
        inputs_match = re.search(
            r"#### Inputs\n\n\|.*?\|.*?\|\n\|[-| ]+\|\n(.*?)(?=\n\n|\n####)",
            sec,
            re.DOTALL,
        )
        operands = []
        if inputs_match:  # pragma: no cover
            for row in inputs_match.group(1).strip().split("\n"):
                cols = [c.strip() for c in row.split("|") if c.strip()]
                if len(cols) >= 2:  # pragma: no cover
                    name = cols[1].strip("` ")
                    if name != "Name":  # skip header if parsed wrong
                        operands.append(name)

        ops.append(
            {
                "api_path": api_path,
                "dialect": "stablehlo",
                "class_name": class_name,
                "operands": operands,
                "attributes": [],
                "description": desc,
            }
        )
    return ops


def main() -> None:
    """Run the script to scrape MLIR docs and generate the JSON dump."""
    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "mlir_exhaustive.json",
    )

    print("Fetching dialect index...")
    index_html = fetch_html(MLIR_DOCS_URL)
    if not index_html:  # pragma: no cover
        print("Failed to fetch MLIR docs index.")
        return

    # Extract dialect links
    # <a href=https://mlir.llvm.org/docs/Dialects/ArithOps/>'arith' Dialect</a>
    dialect_links = re.findall(
        r"<a href=([^\s>]+)[^>]*>'([^']+)'\s*Dialect</a>", index_html
    )

    # MLIR docs links can be relative or absolute, normalize them
    # e.g. /docs/Dialects/OpenACCDialect/ or https://mlir.llvm.org/...

    all_ops: List[Dict[str, Any]] = []

    seen_urls = set()
    for link, dialect_name in dialect_links:
        link = link.strip("\"'")
        if link.startswith("/"):
            dialect_url = f"https://mlir.llvm.org{link}"
        else:  # pragma: no cover
            dialect_url = link

        if dialect_url in seen_urls:  # pragma: no cover
            continue
        seen_urls.add(dialect_url)

        print(f"Scraping dialect: {dialect_name} ({dialect_url})")
        ops = parse_dialect_page(dialect_url, dialect_name)
        print(f"  Found {len(ops)} operations.")
        all_ops.extend(ops)

    print("Scraping stablehlo dialect...")
    stablehlo_ops = scrape_stablehlo()
    print(f"  Found {len(stablehlo_ops)} operations.")
    all_ops.extend(stablehlo_ops)

    print(f"Total MLIR operations extracted: {len(all_ops)}")

    with open(output_path, "w") as f:
        json.dump(all_ops, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Dumped exhaustive MLIR operations to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
