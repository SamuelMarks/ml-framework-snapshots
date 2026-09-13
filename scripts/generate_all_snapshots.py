"""Script to extract and save API snapshots for all supported frameworks."""

import os
import sys

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ml_framework_snapshots.api import (  # noqa: E402
    extract_snapshot,
    extract_snapshot_isolated,
    write_snapshot,
)


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
        "nvidia_ptx",
        "amd_rdna",
    ]
    frameworks = sys.argv[1:] if len(sys.argv) > 1 else all_frameworks
    use_isolated = os.environ.get("ISOLATE_EXTRACTION", "0") == "1"
    has_error = False
    for fw in frameworks:
        print(f"Building snapshot for {fw}...")
        try:
            if use_isolated:
                snapshot = extract_snapshot_isolated(fw)
                if not snapshot:
                    # Fallback to in-process extraction if subprocess failed
                    snapshot = extract_snapshot(fw)
            else:
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


if __name__ == "__main__":
    main()
