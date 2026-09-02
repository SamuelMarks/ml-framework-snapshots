"""Tests for scripts/matrix_runner.py."""

import json
from pathlib import Path
from typing import Any

from ml_framework_snapshots.tools.matrix_runner import (
    build_and_run,
    upload_to_s3,
    main,
    DEFAULT_MATRIX,
)


def test_build_and_run(mocker: Any, tmp_path: Path) -> None:
    """Test build_and_run function.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mock_run = mocker.patch("subprocess.run")

    out_dir = tmp_path / "out"
    res = build_and_run("torch", "1.13.1", out_dir)

    assert res == out_dir / "torch" / "1.13.1"
    assert res.exists()
    assert mock_run.call_count == 3


def test_upload_to_s3_success(mocker: Any, tmp_path: Path) -> None:
    """Test upload_to_s3 function with boto3.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mock_boto3 = mocker.MagicMock()
    mock_s3 = mocker.MagicMock()
    mock_boto3.client.return_value = mock_s3
    mocker.patch.dict("sys.modules", {"boto3": mock_boto3})

    # Create dummy files
    f1 = tmp_path / "test1.json"
    f1.write_text("{}")

    d1 = tmp_path / "sub"
    d1.mkdir()
    f2 = d1 / "test2.json"
    f2.write_text("{}")

    upload_to_s3(tmp_path, "my-bucket")

    assert mock_s3.upload_file.call_count == 2
    mock_s3.upload_file.assert_any_call(str(f1), "my-bucket", "snapshots/test1.json")
    mock_s3.upload_file.assert_any_call(
        str(f2), "my-bucket", "snapshots/sub/test2.json"
    )


def test_upload_to_s3_empty_dir(mocker: Any, tmp_path: Path) -> None:
    """Test upload_to_s3 with empty directory.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mock_boto3 = mocker.MagicMock()
    mock_s3 = mocker.MagicMock()
    mock_boto3.client.return_value = mock_s3
    mocker.patch.dict("sys.modules", {"boto3": mock_boto3})

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    upload_to_s3(empty_dir, "my-bucket")

    assert mock_s3.upload_file.call_count == 0


def test_upload_to_s3_import_error(mocker: Any, tmp_path: Path) -> None:
    """Test upload_to_s3 function without boto3.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mocker.patch.dict("sys.modules", {"boto3": None})
    mock_print = mocker.patch("builtins.print")

    upload_to_s3(tmp_path, "my-bucket")
    mock_print.assert_called_with("boto3 not installed. Skipping S3 upload.")


def test_upload_to_s3_other_error(mocker: Any, tmp_path: Path) -> None:
    """Test upload_to_s3 function handling generic exceptions.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mock_boto3 = mocker.MagicMock()
    mock_boto3.client.side_effect = Exception("Test error")
    mocker.patch.dict("sys.modules", {"boto3": mock_boto3})

    mock_print = mocker.patch("builtins.print")

    upload_to_s3(tmp_path, "my-bucket")
    mock_print.assert_called_with("Failed to upload to S3: Test error")


def test_main_default(mocker: Any, tmp_path: Path) -> None:
    """Test main function with default args.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    mocker.patch("sys.argv", ["matrix_runner.py", "--output-dir", str(tmp_path)])
    mock_build = mocker.patch(
        "ml_framework_snapshots.tools.matrix_runner.build_and_run"
    )

    main()

    # Check if build_and_run was called for every framework/version in DEFAULT_MATRIX
    expected_calls = sum(len(v) for v in DEFAULT_MATRIX.values())
    assert mock_build.call_count == expected_calls


def test_main_with_matrix_and_s3(mocker: Any, tmp_path: Path) -> None:
    """Test main function with custom matrix and s3.

    Args:
        mocker: Parameter.
        tmp_path: Parameter.
    """
    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(json.dumps({"test_fw": ["1.0"]}))

    mocker.patch(
        "sys.argv",
        [
            "matrix_runner.py",
            "--output-dir",
            str(tmp_path),
            "--matrix",
            str(matrix_file),
            "--s3-bucket",
            "test-bucket",
        ],
    )
    mock_build = mocker.patch(
        "ml_framework_snapshots.tools.matrix_runner.build_and_run"
    )
    mock_upload = mocker.patch(
        "ml_framework_snapshots.tools.matrix_runner.upload_to_s3"
    )

    main()

    mock_build.assert_called_once_with("test_fw", "1.0", tmp_path)
    mock_upload.assert_called_once_with(tmp_path, "test-bucket")
