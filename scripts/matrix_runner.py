import argparse
import subprocess
import json
import tempfile
import os
from pathlib import Path

DEFAULT_MATRIX = {
    "torch": ["1.13.1", "2.0.1", "2.3.0"],
    "tensorflow": ["2.13.0", "2.16.1"],
    "transformers": ["4.30.0", "4.40.0"],
}


def build_and_run(framework: str, version: str, output_dir: Path) -> Path:
    print(f"Running matrix for {framework}=={version}")

    # We create a simple Dockerfile
    dockerfile_content = f"""
    FROM python:3.9-slim

    RUN pip install {framework}=={version}
    # Install ml_framework_snapshots (assuming it's installed or we copy the current dir)
    # For a real run, we would pip install the SDK. Here we copy the source.
    COPY . /app
    WORKDIR /app
    RUN pip install -e .

    # Run the snapshot generation
    CMD ["ml-snapshots", "capture", "--framework", "{framework}", "--output-dir", "/out"]
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "Dockerfile").write_text(dockerfile_content)

        image_tag = f"snapshot-runner-{framework}-{version}"

        # Build image
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                image_tag,
                "-f",
                str(tmp_path / "Dockerfile"),
                ".",
            ],
            check=False,
        )

        # Run container and mount output directory
        out_path = output_dir / framework / version
        out_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{out_path.absolute()}:/out", image_tag],
            check=False,
        )

        # Clean up image to save space
        subprocess.run(
            ["docker", "rmi", image_tag],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return out_path


def upload_to_s3(directory: Path, bucket: str) -> None:
    try:
        import boto3

        s3 = boto3.client("s3")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, directory)
                    s3_key = f"snapshots/{rel_path}"
                    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                    s3.upload_file(local_path, bucket, s3_key)
    except ImportError:
        print("boto3 not installed. Skipping S3 upload.")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Version Matrix Runner")
    parser.add_argument(
        "--matrix", type=str, help="Path to JSON matrix config", default=None
    )
    parser.add_argument("--output-dir", type=str, default="./matrix_snapshots")
    parser.add_argument(
        "--s3-bucket", type=str, help="S3 bucket for upload", default=None
    )

    args = parser.parse_args()

    matrix = DEFAULT_MATRIX
    if args.matrix:
        with open(args.matrix, "r") as f:
            matrix = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for framework, versions in matrix.items():
        for version in versions:
            build_and_run(framework, version, out_dir)

    if args.s3_bucket:
        upload_to_s3(out_dir, args.s3_bucket)


if __name__ == "__main__":
    main()
