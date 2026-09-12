"""Tests for the generate_all_snapshots.py script."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the root directory to sys.path so we can import scripts.generate_all_snapshots
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.generate_all_snapshots as generate_all_snapshots


@patch("sys.argv", ["generate_all_snapshots.py"])
@patch("scripts.generate_all_snapshots.os.makedirs")
@patch("scripts.generate_all_snapshots.extract_snapshot")
@patch("scripts.generate_all_snapshots.write_snapshot")
@patch("builtins.print")
def test_main_success(
    mock_print: MagicMock,
    mock_write: MagicMock,
    mock_extract: MagicMock,
    mock_makedirs: MagicMock,
) -> None:
    """Test the main function for successful snapshot generation.

    Args:
        mock_print: Mock for builtins.print.
        mock_write: Mock for write_snapshot.
        mock_extract: Mock for extract_snapshot.
        mock_makedirs: Mock for os.makedirs.
    """
    mock_extract.return_value = {"mock": "data"}

    generate_all_snapshots.main()

    mock_makedirs.assert_called_once_with(
        os.path.join("src", "ml_framework_snapshots", "snapshots"), exist_ok=True
    )

    frameworks = [
        "torch",
        "jax",
        "tensorflow",
        "keras",
        "mlx",
        "numpy",
        "cupy",
        "dask",
        "flax_nnx",
        "deepspeed",
        "optax_shim",
        "orbax_checkpoint",
        "huggingface",
        "diffusers",
        "tokenizers",
        "triton",
        "sklearn",
        "onnxruntime",
        "pax",
        "maxtext",
        "mlir",
        "stablehlo",
        "html_dsl",
        "latex_dsl",
        "tikz",
        "nvidia_sass",
        "amd_rdna",
    ]

    assert mock_extract.call_count == len(frameworks)
    assert mock_write.call_count == len(frameworks)

    for fw in frameworks:
        mock_extract.assert_any_call(fw)
        mock_write.assert_any_call(
            fw,
            {"mock": "data"},
            os.path.join("src", "ml_framework_snapshots", "snapshots"),
        )
        mock_print.assert_any_call(f"Building snapshot for {fw}...")
        mock_print.assert_any_call("  -> Saved")


@patch("sys.argv", ["generate_all_snapshots.py"])
@patch("scripts.generate_all_snapshots.os.makedirs")
@patch("scripts.generate_all_snapshots.extract_snapshot")
@patch("scripts.generate_all_snapshots.write_snapshot")
@patch("builtins.print")
def test_main_failure(
    mock_print: MagicMock,
    mock_write: MagicMock,
    mock_extract: MagicMock,
    mock_makedirs: MagicMock,
) -> None:
    """Test the main function when snapshot extraction fails.

    Args:
        mock_print: Mock for builtins.print.
        mock_write: Mock for write_snapshot.
        mock_extract: Mock for extract_snapshot.
        mock_makedirs: Mock for os.makedirs.
    """
    mock_extract.side_effect = Exception("Mock exception")

    with pytest.raises(SystemExit):
        generate_all_snapshots.main()

    frameworks = [
        "torch",
        "jax",
        "tensorflow",
        "keras",
        "mlx",
        "numpy",
        "cupy",
        "dask",
        "flax_nnx",
        "deepspeed",
        "optax_shim",
        "orbax_checkpoint",
        "huggingface",
        "diffusers",
        "tokenizers",
        "triton",
        "sklearn",
        "onnxruntime",
        "pax",
        "maxtext",
        "mlir",
        "stablehlo",
        "html_dsl",
        "latex_dsl",
        "tikz",
        "nvidia_sass",
        "amd_rdna",
    ]

    assert mock_extract.call_count == len(frameworks)
    assert mock_write.call_count == 0

    for fw in frameworks:
        mock_print.assert_any_call(f"Building snapshot for {fw}...")
        mock_print.assert_any_call("  -> Failed: Mock exception")


def test_sys_path_insert() -> None:
    """Test module reload when _src_dir is not in sys.path to cover line 8."""
    import importlib

    _src_dir = os.path.abspath(
        os.path.join(os.path.dirname(generate_all_snapshots.__file__), "..", "src")
    )
    orig_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p != _src_dir]
        importlib.reload(generate_all_snapshots)
        assert _src_dir in sys.path
    finally:
        sys.path = orig_path


@patch("sys.argv", ["generate_all_snapshots.py", "torch", "jax"])
@patch.dict(os.environ, {"ISOLATE_EXTRACTION": "1"})
@patch("scripts.generate_all_snapshots.os.makedirs")
@patch("scripts.generate_all_snapshots.extract_snapshot_isolated")
@patch("scripts.generate_all_snapshots.write_snapshot")
@patch("builtins.print")
def test_main_custom_args_and_isolated_success(
    mock_print: MagicMock,
    mock_write: MagicMock,
    mock_extract_iso: MagicMock,
    mock_makedirs: MagicMock,
) -> None:
    """Test main function with CLI arguments and successful isolated extraction.

    Args:
        mock_print: Mock for builtins.print.
        mock_write: Mock for write_snapshot.
        mock_extract_iso: Mock for extract_snapshot_isolated.
        mock_makedirs: Mock for os.makedirs.
    """
    mock_extract_iso.return_value = {"custom": "snap"}

    generate_all_snapshots.main()

    assert mock_extract_iso.call_count == 2
    mock_extract_iso.assert_any_call("torch")
    mock_extract_iso.assert_any_call("jax")
    assert mock_write.call_count == 2


@patch("sys.argv", ["generate_all_snapshots.py", "torch"])
@patch.dict(os.environ, {"ISOLATE_EXTRACTION": "1"})
@patch("scripts.generate_all_snapshots.os.makedirs")
@patch("scripts.generate_all_snapshots.extract_snapshot_isolated")
@patch("scripts.generate_all_snapshots.extract_snapshot")
@patch("scripts.generate_all_snapshots.write_snapshot")
@patch("builtins.print")
def test_main_isolated_fallback(
    mock_print: MagicMock,
    mock_write: MagicMock,
    mock_extract: MagicMock,
    mock_extract_iso: MagicMock,
    mock_makedirs: MagicMock,
) -> None:
    """Test main function when isolated extraction returns empty snapshot and falls back to in-process.

    Args:
        mock_print: Mock for builtins.print.
        mock_write: Mock for write_snapshot.
        mock_extract: Mock for extract_snapshot.
        mock_extract_iso: Mock for extract_snapshot_isolated.
        mock_makedirs: Mock for os.makedirs.
    """
    mock_extract_iso.return_value = {}
    mock_extract.return_value = {"fallback": "data"}

    generate_all_snapshots.main()

    mock_extract_iso.assert_called_once_with("torch")
    mock_extract.assert_called_once_with("torch")
    assert mock_write.call_count == 1
