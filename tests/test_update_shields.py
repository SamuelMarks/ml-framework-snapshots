"""Tests for scripts/update_shields.py."""

from typing import Any
import pytest
import os

from scripts.update_shields import get_test_coverage, get_doc_coverage, get_color, main


def test_get_test_coverage_success(mocker: Any) -> None:
    """Test get_test_coverage when pytest returns output with TOTAL.

    Args:
        mocker: Parameter.
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "TOTAL  100 0 100%\n"
    assert get_test_coverage() == "100"


def test_get_test_coverage_fail(mocker: Any) -> None:
    """Test get_test_coverage when pytest output fails.

    Args:
        mocker: Parameter.
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "No total here"
    assert get_test_coverage() == "unknown"


def test_get_doc_coverage_success(mocker: Any) -> None:
    """Test get_doc_coverage when interrogate returns success.

    Args:
        mocker: Parameter.
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "RESULT: PASSED (minimum: 80.0%, actual: 95.5%)"
    assert get_doc_coverage() == "95.5"


def test_get_doc_coverage_fail(mocker: Any) -> None:
    """Test get_doc_coverage when interrogate fails to match.

    Args:
        mocker: Parameter.
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "RESULT: PASSED"
    assert get_doc_coverage() == "unknown"


def test_get_doc_coverage_file_not_found(mocker: Any) -> None:
    """Test get_doc_coverage when interrogate is missing.

    Args:
        mocker: Parameter.
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.side_effect = FileNotFoundError()
    assert get_doc_coverage() == "unknown"


@pytest.mark.parametrize(
    "cov, expected",
    [
        ("95", "brightgreen"),
        (90, "brightgreen"),
        ("85", "yellow"),
        (80, "yellow"),
        ("75", "orange"),
        (70, "orange"),
        ("65", "red"),
        (0, "red"),
        ("unknown", "lightgrey"),
        (None, "lightgrey"),
    ],
)
def test_get_color(cov: Any, expected: str) -> None:
    """Test get_color with various inputs.

    Args:
        cov: Parameter.
        expected: Parameter.
    """
    assert get_color(cov) == expected


def test_main(mocker: Any, tmp_path: Any) -> None:
    """Test main function updates README.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mocker.patch("scripts.update_shields.get_test_coverage", return_value="100")
    mocker.patch("scripts.update_shields.get_doc_coverage", return_value="100")

    # Change working directory so it writes to tmp_path/README.md
    readme_path = tmp_path / "README.md"
    readme_path.write_text("[![License](some_license)]()\n")

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        main()
        content = readme_path.read_text()
        assert "tests-100%25-brightgreen.svg" in content
        assert "docs-100%25-brightgreen.svg" in content
    finally:
        os.chdir(orig_cwd)


def test_main_replace_existing(mocker: Any, tmp_path: Any) -> None:
    """Test main function replaces existing badges.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mocker.patch("scripts.update_shields.get_test_coverage", return_value="90")
    mocker.patch("scripts.update_shields.get_doc_coverage", return_value="85")

    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "[![License](license)]()\n[![Tests](oldtests)]()\n[![Docs](olddocs)]()\n"
    )

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        main()
        content = readme_path.read_text()
        assert "tests-90%25-brightgreen.svg" in content
        assert "docs-85%25-yellow.svg" in content
        assert "oldtests" not in content
        assert "olddocs" not in content
    finally:
        os.chdir(orig_cwd)
