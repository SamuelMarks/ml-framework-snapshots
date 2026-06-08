"""Command Line Interface for ML Framework Snapshots.

Provides the entrypoint for generating framework snapshots from the terminal.
"""

import argparse
import json
import os
from typing import List

from ml_framework_snapshots.api import (
    extract_snapshot,
    write_snapshot,
)
from ml_framework_snapshots.diff import diff_snapshots, generate_changelog
from ml_framework_snapshots.stubs import generate_stubs
from ml_switcheroo_ir.schema.ghost import GhostRef


def cmd_capture(args: argparse.Namespace) -> None:
    """Handle the capture command.

    Args:
        args: Parsed arguments
    """
    import logging
    from rich.progress import Progress

    logging.getLogger("griffe").setLevel(logging.ERROR)

    from ml_framework_snapshots.api import get_available_frameworks

    available = get_available_frameworks()
    target_fws = []

    if "all" in args.frameworks or "*" in args.frameworks or not args.frameworks:
        target_fws = list(available.keys())
    else:
        target_fws = [fw for fw in args.frameworks if fw in available]
        unsupported = [fw for fw in args.frameworks if fw not in available]
        if unsupported:
            print(
                f"Warning: The following frameworks are unsupported and will be skipped: {', '.join(unsupported)}"
            )

    with Progress() as progress:
        task = progress.add_task("[cyan]Scanning frameworks...", total=len(target_fws))
        for fw in target_fws:
            progress.update(task, description=f"[cyan]Scanning {fw}...")
            snapshot_data = extract_snapshot(
                fw, include_nonpublic=args.include_nonpublic
            )
            if snapshot_data:
                path = write_snapshot(fw, snapshot_data, args.out_dir)
                progress.console.print(f"[green]Saved snapshot to {path}[/green]")
            else:
                progress.console.print(
                    f"[yellow]Skipping {fw}, not installed or no components found.[/yellow]"
                )
            progress.advance(task)


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the diff command.

    Args:
        args: description

    """
    with open(args.json1, "r", encoding="utf-8") as f:
        snap1 = json.load(f)
    with open(args.json2, "r", encoding="utf-8") as f:
        snap2 = json.load(f)

    result = diff_snapshots(snap1, snap2)

    if args.changelog:
        print(generate_changelog(result))
    else:
        print(f"ADDED: {len(result.added)}")
        for p in result.added:
            print(f"  + {p}")

        print(f"REMOVED: {len(result.removed)}")
        for p in result.removed:
            print(f"  - {p}")

        print(f"SIGNATURE CHANGED: {len(result.signature_changed)}")
        for p in result.signature_changed:
            print(f"  ~ {p}")


def cmd_stubs(args: argparse.Namespace) -> None:
    """Handle the generate-stubs command.

    Args:
        args: description

    """
    with open(args.input, "r", encoding="utf-8") as f:
        snap = json.load(f)
    generate_stubs(snap, args.out_dir, include_nonpublic=args.include_nonpublic)
    print(f"Stubs generated in {args.out_dir}")


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the export command.

    Args:
        args: description

    Raises:
        ValueError: if invalid.
    """
    with open(args.input, "r", encoding="utf-8") as f:
        snap = json.load(f)

    refs: List[GhostRef] = []
    for cat, items in snap.get("categories", {}).items():
        for item in items:
            refs.append(GhostRef.model_validate(item))

    os.makedirs(args.out_dir, exist_ok=True)

    if args.format == "openapi":
        from ml_framework_snapshots.export import to_openapi

        openapi_spec = to_openapi(refs)
        out_path = os.path.join(args.out_dir, "openapi.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2)
        print(f"Exported OpenAPI spec to {out_path}")
    elif args.format == "json_schema":
        from ml_framework_snapshots.export import to_json_schema

        for ref in refs:
            schema = to_json_schema(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name}.schema.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2)
        print(f"Exported {len(refs)} JSON schemas to {args.out_dir}")
    elif args.format == "pydantic":
        from ml_framework_snapshots.export import to_pydantic

        for ref in refs:
            code = to_pydantic(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name.lower()}.py")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} Pydantic models to {args.out_dir}")
    elif args.format == "protobuf":
        from ml_framework_snapshots.export import to_protobuf

        for ref in refs:
            code = to_protobuf(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name.lower()}.proto")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} Protobuf definitions to {args.out_dir}")
    else:
        raise ValueError(f"Unknown format: {args.format}")


def cmd_check(args: argparse.Namespace) -> None:
    """Handle the check command for compliance checking.

    Args:
        args: Parsed command line arguments containing snapshot_json, target_path,
              target_prefix, and reference_prefix.
    """
    import collections
    from ml_framework_snapshots.compliance import extract_target_refs, score_compliance

    with open(args.snapshot_json, "r", encoding="utf-8") as f:
        reference_snapshot = json.load(f)

    print(f"Extracting target APIs from {args.target_path}...")
    target_refs = extract_target_refs(
        args.target_path, args.target_prefix, args.reference_prefix
    )

    print("Scoring compliance...")
    results = score_compliance(reference_snapshot, target_refs)

    score = results.get("score_percentage", 0.0)
    print("\n--- Compliance Report ---")
    print(f"Overall Compliance: {score}%\n")

    # Break down by submodules
    print("Breakdown by Module:")
    matched_by_mod = collections.defaultdict(list)
    missing_by_mod = collections.defaultdict(list)

    for api_path in results.get("matched", []):
        mod = ".".join(api_path.split(".")[:-1])
        matched_by_mod[mod].append(api_path)

    for api_path in results.get("missing", []):
        mod = ".".join(api_path.split(".")[:-1])
        missing_by_mod[mod].append(api_path)

    all_mods = set(list(matched_by_mod.keys()) + list(missing_by_mod.keys()))
    for mod in sorted(all_mods):
        m = len(matched_by_mod[mod])
        total = m + len(missing_by_mod[mod])
        mod_score = (m / total) * 100 if total > 0 else 0
        print(f"  - {mod}: {mod_score:.1f}% ({m}/{total})")

    missing = results.get("missing", [])
    if missing:
        print(f"\nMissing APIs ({len(missing)}):")
        # Just show top 10 to not spam, or maybe all
        for api_path in sorted(missing)[:20]:
            print(f"  - {api_path}")
        if len(missing) > 20:
            print(f"  - ... and {len(missing) - 20} more")

    mismatched = results.get("mismatched", [])
    if mismatched:
        print(f"\nMismatched APIs ({len(mismatched)}):")
        for item in sorted(mismatched, key=lambda x: x["api_path"])[:20]:
            print(f"  ~ {item['api_path']}")
        if len(mismatched) > 20:
            print(f"  ~ ... and {len(mismatched) - 20} more")


def main() -> None:
    """Parse arguments and route to subcommands."""
    parser = argparse.ArgumentParser(description="ML Framework Snapshots CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # capture
    parser_capture = subparsers.add_parser("capture", help="Capture snapshots")
    parser_capture.add_argument(
        "frameworks",
        nargs="*",
        default=["all"],
        help="Frameworks to capture, or 'all' for all supported frameworks",
    )
    parser_capture.add_argument(
        "--out-dir", type=str, default="snapshots", help="Output directory"
    )
    parser_capture.add_argument(
        "--include-nonpublic", action="store_true", help="Include non-public APIs"
    )
    parser_capture.set_defaults(func=cmd_capture)

    # diff
    parser_diff = subparsers.add_parser("diff", help="Diff two snapshots")
    parser_diff.add_argument("json1", type=str, help="First snapshot JSON")
    parser_diff.add_argument("json2", type=str, help="Second snapshot JSON")
    parser_diff.add_argument(
        "--changelog", action="store_true", help="Generate a markdown changelog"
    )
    parser_diff.set_defaults(func=cmd_diff)

    # generate-stubs
    parser_stubs = subparsers.add_parser("generate-stubs", help="Generate .pyi stubs")
    parser_stubs.add_argument(
        "--input", type=str, required=True, help="Input JSON snapshot"
    )
    parser_stubs.add_argument(
        "--out-dir", type=str, required=True, help="Output directory for stubs"
    )
    parser_stubs.add_argument(
        "--include-nonpublic",
        action="store_true",
        help="Include non-public APIs in stubs",
    )
    parser_stubs.set_defaults(func=cmd_stubs)

    # export
    parser_export = subparsers.add_parser(
        "export", help="Export to JSON Schema, OpenAPI, Pydantic, or Protobuf"
    )
    parser_export.add_argument(
        "--input", type=str, required=True, help="Input JSON snapshot"
    )
    parser_export.add_argument(
        "--out-dir", type=str, required=True, help="Output directory"
    )
    parser_export.add_argument(
        "--format",
        choices=["openapi", "json_schema", "pydantic", "protobuf"],
        required=True,
        help="Format to export",
    )
    parser_export.set_defaults(func=cmd_export)

    # check
    parser_check = subparsers.add_parser(
        "check", help="Check compliance of a target implementation against a snapshot"
    )
    parser_check.add_argument(
        "snapshot_json", type=str, help="Reference snapshot JSON file"
    )
    parser_check.add_argument(
        "target_path", type=str, help="Target implementation file or directory"
    )
    parser_check.add_argument(
        "--target-prefix",
        type=str,
        required=True,
        help="Module prefix in target (e.g., ml_switcheroo.jax)",
    )
    parser_check.add_argument(
        "--reference-prefix",
        type=str,
        required=True,
        help="Module prefix in reference (e.g., jax)",
    )
    parser_check.set_defaults(func=cmd_check)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
