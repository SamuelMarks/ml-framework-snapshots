"""Script to extract and save API snapshots for all supported frameworks."""

import os
import sys
from ml_framework_snapshots.api import extract_snapshot, write_snapshot


def main() -> None:
    """Generate and save API snapshots for supported ML frameworks."""
    os.makedirs(
        os.path.join("src", "ml_framework_snapshots", "snapshots"), exist_ok=True
    )
    all_frameworks = [
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
        "triton",
        "sklearn",
        "onnxruntime",
        "pax",
        "maxtext",
        "mlir",
        "html_dsl",
        "latex_dsl",
        "tikz",
        "sass",
    ]
    frameworks = sys.argv[1:] if len(sys.argv) > 1 else all_frameworks
    has_error = False
    for fw in frameworks:
        print(f"Building snapshot for {fw}...")
        try:
            snapshot = extract_snapshot(fw)
            write_snapshot(
                fw, snapshot, os.path.join("src", "ml_framework_snapshots", "snapshots")
            )
            print("  -> Saved")
        except Exception as e:
            print(f"  -> Failed: {e}")
            has_error = True

    if has_error:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
