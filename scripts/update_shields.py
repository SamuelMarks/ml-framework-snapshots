"""Module docstring."""

import os
import sys
from typing import Any

import re
import subprocess


def get_test_coverage() -> str:
    """Get the current test coverage percentage.

    Returns:
        The string representation of the test coverage percentage.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src/ml_framework_snapshots",
            "--cov-branch",
        ],
        capture_output=True,
        text=True,
    )
    match = re.search(r"TOTAL\s+.*\s+(\d+)%", result.stdout)
    if match:
        return match.group(1)
    return "unknown"


def get_doc_coverage() -> str:
    """Get the current documentation coverage percentage.

    Returns:
        The string representation of the documentation coverage percentage.
    """
    # If interrogate is not available, try parsing another way, but we know it's there
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "interrogate",
                os.path.join("src", "ml_framework_snapshots"),
            ],
            capture_output=True,
            text=True,
        )
        match = re.search(r"actual:\s*([\d\.]+)%", result.stdout)
        if match:
            val = float(match.group(1))
            return f"{val:g}"
    except FileNotFoundError:
        return "unknown"
    return "unknown"


def get_color(coverage: Any) -> str:
    """Determine the badge color based on coverage.

    Args:
        coverage: The coverage percentage (can be a string or float).

    Returns:
        The color name for the shield badge.
    """
    try:
        val = float(coverage)
        if val >= 90:
            return "brightgreen"
        if val >= 80:
            return "yellow"
        if val >= 70:
            return "orange"
        return "red"
    except (ValueError, TypeError):
        return "lightgrey"


def main() -> None:
    """Update the coverage shields in README.md."""
    test_cov = get_test_coverage()
    doc_cov = get_doc_coverage()

    with open("README.md", "r") as f:
        content = f.read()

    test_badge = f"[![Tests](https://img.shields.io/badge/tests-{test_cov}%25-{get_color(test_cov)}.svg)]()"
    doc_badge = f"[![Docs](https://img.shields.io/badge/docs-{doc_cov}%25-{get_color(doc_cov)}.svg)]()"

    if "![Tests]" in content:
        content = re.sub(r"\[\!\[Tests\].*?\]\([^\)]*\)", test_badge, content)
    else:
        content = re.sub(
            r"(\[\!\[License\][^\n]+\n)", r"\1" + test_badge + "\n", content, count=1
        )

    if "![Docs]" in content:
        content = re.sub(r"\[\!\[Docs\].*?\]\([^\)]*\)", doc_badge, content)
    else:
        content = re.sub(
            r"(\[\!\[Tests\][^\n]+\n)", r"\1" + doc_badge + "\n", content, count=1
        )

    with open("README.md", "w") as f:
        f.write(content)


if __name__ == "__main__":  # pragma: no cover
    main()
