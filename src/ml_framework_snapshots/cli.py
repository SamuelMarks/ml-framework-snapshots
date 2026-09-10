"""Command Line Interface for ML Framework Snapshots.

Provides the entrypoint for generating framework snapshots from the terminal.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from ml_framework_snapshots.api import (
    extract_snapshot,
    write_snapshot,
)
from ml_framework_snapshots.diff import diff_snapshots, generate_changelog
from ml_framework_snapshots.index import get_cache_dir
from ml_framework_snapshots.models import GhostInspector
from ml_framework_snapshots.stubs import generate_stubs
from ml_framework_snapshots.utils import get_custom_snapshots_paths, is_offline_mode
from ml_switcheroo_ir.schema.ghost import GhostRef


def resolve_snapshot_path(path: str) -> str:
    """Resolve a snapshot path, looking in the local snapshots directory if needed.

    Args:
        path: The path or name to resolve.

    Returns:
        The resolved full path if found, or the original path otherwise.
    """
    if os.path.exists(path):
        return path

    if os.path.exists(path + ".json"):
        return path + ".json"

    # Try looking in project or package-bundled snapshots/frameworks directories
    basename = os.path.basename(path)
    pkg_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))

    search_dirs: List[str] = get_custom_snapshots_paths() + [
        os.path.join(get_cache_dir(), "snapshots"),
        os.path.join(repo_root, "snapshots"),
        os.path.join(pkg_dir, "snapshots"),
        os.path.join(pkg_dir, "frameworks"),
        os.path.join(os.getcwd(), "snapshots"),
    ]

    seen_dirs = set()
    candidate_dirs: List[str] = []
    for d in search_dirs:
        norm_d = os.path.normpath(d)
        if norm_d not in seen_dirs and os.path.isdir(norm_d):
            seen_dirs.add(norm_d)
            candidate_dirs.append(norm_d)

    for d in candidate_dirs:
        exact_candidate = os.path.join(d, basename)
        if os.path.exists(exact_candidate):
            return exact_candidate
        json_candidate = os.path.join(d, basename + ".json")
        if os.path.exists(json_candidate):
            return json_candidate

    import glob

    all_matches: List[str] = []
    for d in candidate_dirs:
        matches = glob.glob(os.path.join(d, f"{basename}*.json"))
        if matches:
            all_matches.extend(matches)

    if len(all_matches) == 1:
        return all_matches[0]
    elif len(all_matches) > 1:
        return sorted(all_matches)[-1]

    return path


def cmd_capture(args: argparse.Namespace) -> None:
    """Handle the capture command.

    Args:
        args: Parsed arguments
    """
    import logging
    from rich.progress import Progress

    logging.getLogger("griffe").setLevel(logging.CRITICAL)

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
    path1 = resolve_snapshot_path(args.json1)
    path2 = resolve_snapshot_path(args.json2)

    with open(path1, "r", encoding="utf-8") as f:
        snap1 = json.load(f)
    with open(path2, "r", encoding="utf-8") as f:
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
    path = resolve_snapshot_path(args.input)
    with open(path, "r", encoding="utf-8") as f:
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
    path = resolve_snapshot_path(args.input)
    with open(path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    refs: List[GhostRef] = []
    if isinstance(snap, list):
        for item in snap:
            refs.append(GhostInspector.hydrate(item))
    elif isinstance(snap, dict):
        for cat, items in snap.get("categories", {}).items():
            for item in items:
                refs.append(GhostInspector.hydrate(item))

    os.makedirs(args.out_dir, exist_ok=True)

    if args.format == "openapi":
        from ml_framework_snapshots.export import to_openapi

        openapi_spec = to_openapi(refs)
        out_path = os.path.join(args.out_dir, "openapi.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Exported OpenAPI spec to {out_path}")
    elif args.format == "json_schema":
        from ml_framework_snapshots.export import to_json_schema

        for ref in refs:
            schema = to_json_schema(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name}.schema.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2, sort_keys=True)
                f.write("\n")
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
    elif args.format == "llm_prompt":
        from ml_framework_snapshots.export import export_llm_prompt_context

        content = export_llm_prompt_context(refs)
        out_path = os.path.join(args.out_dir, "llm_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported LLM prompt context to {out_path}")
    else:
        raise ValueError(f"Unknown format: {args.format}")


def cmd_mcp(args: argparse.Namespace) -> None:
    """Handle the mcp command to start Model Context Protocol JSON-RPC server.

    Args:
        args: Parsed command line arguments.
    """
    from ml_framework_snapshots.mcp_server import run_mcp_server

    run_mcp_server()


def cmd_check(args: argparse.Namespace) -> None:
    """Handle the check command for compliance checking.

    Args:
        args: Parsed command line arguments containing snapshot_json, target_path,
              target_prefix, and reference_prefix.
    """
    import collections
    from ml_framework_snapshots.compliance import extract_target_refs, score_compliance

    path = resolve_snapshot_path(args.snapshot_json)
    with open(path, "r", encoding="utf-8") as f:
        reference_snapshot = json.load(f)

    target_path = getattr(args, "target_path", "")
    target_ext = os.path.splitext(target_path)[1].lower() if target_path else ""

    if target_ext in (".sass", ".s"):
        from ml_framework_snapshots.compliance import (
            check_sass_assembly_compliance,
            check_rdna_assembly_compliance,
        )

        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        target = reference_snapshot.get("target", "").lower()
        if "rdna" in target or "amd" in target:
            res = check_rdna_assembly_compliance(content)
            print(
                f"AMD RDNA snippet compliance: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
            )
        else:
            res = check_sass_assembly_compliance(content)
            print(
                f"NVIDIA SASS snippet compliance: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
            )
        if not res.get("is_compliant"):
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        return

    if target_ext == ".mlir":
        from ml_framework_snapshots.compliance import check_mlir_text_compliance

        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        print(
            f"MLIR snippet compliance: {res.get('verified_ops')}/{res.get('total_ops')} operations valid."
        )
        if not res.get("is_compliant"):
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        return

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
        print(f"\nMissing APIs ({len(missing)}):\n")

        # Build reference map to get docstrings and signatures
        from ml_switcheroo_ir.schema.ghost import GhostRef

        ref_map = {}
        for cat, items in reference_snapshot.get("categories", {}).items():
            for item in items:
                ref = GhostRef.model_validate(item)
                ref_map[ref.api_path] = ref
                for alias in ref.aliases:
                    ref_map[alias] = ref

        print("|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |")
        print("|---|---|---|---|---|---|---|")

        for fqn in sorted(missing):
            ref = ref_map.get(fqn)
            if not ref:
                continue

            parts = fqn.split(".")
            framework = parts[0]
            symbol = parts[-1]
            namespace = ".".join(parts[:-1])

            # Signature
            if ref.kind == "MODULE":
                sig = "module"
            else:
                sig_parts = []
                for p in ref.params:
                    p_str = p.name
                    if p.annotation:
                        p_str += f": {p.annotation}"
                    if p.default:
                        p_str += f"={p.default}"
                    sig_parts.append(p_str)
                sig = f"({', '.join(sig_parts)})"
                if ref.returns_type:
                    sig += f" -> {ref.returns_type}"

            # Docstring
            doc = ref.docstring or ""
            if doc:
                doc = (
                    doc.strip().split("\n\n")[0].replace("\n", " ").replace("|", "\\|")
                )
                if len(doc) > 100:
                    doc = doc[:97] + "..."
            else:
                doc = "No docstring available."

            print(
                f"| [ ] | {framework} | {namespace} | {symbol} | {fqn} | `{sig}` | {doc} |"
            )

    mismatched = results.get("mismatched", [])
    if mismatched:
        print(f"\nMismatched APIs ({len(mismatched)}):")
        for item in sorted(mismatched, key=lambda x: x["api_path"])[:20]:
            print(
                f"  ~ {item['api_path']}\n    Expected: {item.get('expected')}\n    Actual:   {item.get('actual')}"
            )
        if len(mismatched) > 20:
            print(f"  ~ ... and {len(mismatched) - 20} more")


def cmd_list_snapshots(args: argparse.Namespace) -> None:
    """List pre-bundled or locally available snapshot files.

    Args:
        args: Parsed command line arguments.
    """
    snapshots_dirs = get_custom_snapshots_paths() + [
        os.path.join(get_cache_dir(), "snapshots"),
        os.path.join(os.path.dirname(__file__), "snapshots"),
        os.path.join(os.path.dirname(__file__), "frameworks"),
    ]
    found_snapshots: List[Dict[str, Any]] = []
    seen_files = set()
    for d in snapshots_dirs:
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".json"):
                    if fname in seen_files:
                        continue
                    seen_files.add(fname)
                    full_path = os.path.join(d, fname)
                    size = os.path.getsize(full_path)
                    found_snapshots.append(
                        {"file": fname, "path": full_path, "size": size}
                    )

    if not found_snapshots:
        print("No snapshot files found.")
        return

    print("Available Snapshots:")
    for s in found_snapshots:
        print(f"  - {s['file']} ({s['size']} bytes)")


def cmd_pull(args: argparse.Namespace) -> None:
    """Download a versioned snapshot from GitHub Releases into local cache.

    Args:
        args: Parsed command line arguments containing target and optional out_dir.
    """
    if getattr(args, "offline", False) or is_offline_mode():
        print(
            "Network access disabled: pull command cannot be executed in offline mode."
        )
        sys.exit(1)

    import urllib.request
    from .index import get_cache_dir

    target = args.target.strip()
    if "@" in target:
        framework, version = target.split("@", 1)
    else:
        framework, version = target, "latest"

    out_dir = args.out_dir or os.path.join(get_cache_dir(), "snapshots")
    os.makedirs(out_dir, exist_ok=True)

    filename = (
        f"{framework}_v{version}.json" if version != "latest" else f"{framework}.json"
    )
    dest_path = os.path.join(out_dir, filename)

    tag = f"v{version}" if version != "latest" else "latest"
    url = f"https://github.com/SamuelMarks/ml-framework-snapshots/releases/download/{tag}/{filename}"

    print(f"Pulling snapshot for {framework} ({version}) from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved snapshot to {dest_path}")
    except Exception as e:
        print(f"Failed to download snapshot: {e}")


def cmd_index(args: argparse.Namespace) -> None:
    """Manage ephemeral local SQLite search index.

    Args:
        args: Parsed command line arguments containing index action flags.
    """
    import sqlite3
    from .index import clear_index, ensure_index, get_index_db_path

    db_path = get_index_db_path()

    if getattr(args, "clear", False) or getattr(args, "clean", False):
        if clear_index():
            print(f"Cleared index database at {db_path}")
        else:
            print(f"Failed to clear index database at {db_path}")
        return

    if args.rebuild:
        clear_index()
        conn = ensure_index()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM symbols")
        count = cur.fetchone()[0]
        print(f"Rebuilt index with {count} symbols at {db_path}")
        return

    # Default / status
    if not os.path.exists(db_path):
        print(f"Index database does not exist yet at {db_path}.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM symbols")
    symbol_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM indexed_files")
    file_count = cur.fetchone()[0]
    size = os.path.getsize(db_path)
    print(f"Index database: {db_path}")
    print(f"  Indexed snapshots: {file_count}")
    print(f"  Total symbols:     {symbol_count}")
    print(f"  Database size:     {size} bytes")


def cmd_check_sass(args: argparse.Namespace) -> None:
    """Validate NVIDIA SASS assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, modifiers, sm_arch, or file.
    """
    from .compliance import check_sass_assembly_compliance
    from .mcp_server import check_sass_instruction

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: SASS assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_sass_assembly_compliance(content, sm_arch=args.sm_arch)
        if not res.get("is_compliant"):
            print(
                f"SASS Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"SASS snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
        )
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = (
        [op.strip() for op in args.operands.split(",") if op.strip()]
        if args.operands
        else None
    )
    modifiers = (
        [mod.strip() for mod in args.modifiers.split(",") if mod.strip()]
        if args.modifiers
        else None
    )

    res = check_sass_instruction(
        mnemonic=args.mnemonic,
        operands=operands,
        modifiers=modifiers,
        sm_arch=args.sm_arch,
    )

    if not res.get("is_valid"):
        print(f"SASS Instruction '{args.mnemonic}' Invalid:")
        for err in res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"SASS Instruction '{args.mnemonic}' is valid.")
        print(
            f"  Supported Architectures: {', '.join(res.get('supported_architectures', []))}"
        )
        if res.get("valid_modifiers"):
            print(
                f"  Recognized Modifiers:    {', '.join(res.get('valid_modifiers', []))}"
            )


def cmd_check_rdna(args: argparse.Namespace) -> None:
    """Validate AMD RDNA assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, encoding, gfx_arch, modifiers, wave_size, or file.
    """
    from .compliance import check_rdna_assembly_compliance
    from .mcp_server import check_rdna_instruction

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: RDNA assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_rdna_assembly_compliance(content, gfx_arch=args.gfx_arch)
        if not res.get("is_compliant"):
            print(
                f"RDNA Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"RDNA snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
        )
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = (
        [op.strip() for op in args.operands.split(",") if op.strip()]
        if args.operands
        else None
    )
    modifiers = (
        [mod.strip() for mod in args.modifiers.split(",") if mod.strip()]
        if args.modifiers
        else None
    )

    res = check_rdna_instruction(
        mnemonic=args.mnemonic,
        operands=operands,
        encoding=args.encoding,
        gfx_arch=args.gfx_arch,
        modifiers=modifiers,
        wave_size=args.wave_size,
    )

    if not res.get("is_valid"):
        print(f"RDNA Instruction '{args.mnemonic}' Invalid:")
        for err in res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"RDNA Instruction '{args.mnemonic}' is valid.")
        print(
            f"  Supported Architectures: {', '.join(res.get('supported_architectures', []))}"
        )
        if res.get("encoding"):
            print(f"  Encoding:                {res.get('encoding')}")


def cmd_check_mlir(args: argparse.Namespace) -> None:
    """Validate MLIR text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.
    """
    from .compliance import check_mlir_text_compliance
    from .mcp_server import check_mlir_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: MLIR file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(
                f"MLIR Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"MLIR snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid."
        )
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = (
        [a.strip() for a in args.attributes.split(",") if a.strip()]
        if args.attributes
        else None
    )

    res = check_mlir_op(
        op_name=args.op_name,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not res.get("is_valid"):
        print(f"MLIR Operation '{args.op_name}' Invalid:")
        for err in res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    print(f"MLIR Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(res.get('expected_operands', []))}")
    print(f"  Attributes: {', '.join(res.get('expected_attributes', []))}")


def cmd_check_stablehlo(args: argparse.Namespace) -> None:
    """Validate StableHLO text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.
    """
    from .compliance import check_mlir_text_compliance
    from .mcp_server import check_stablehlo_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: StableHLO file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(
                f"StableHLO Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"StableHLO snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid."
        )
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = (
        [a.strip() for a in args.attributes.split(",") if a.strip()]
        if args.attributes
        else None
    )

    res = check_stablehlo_op(
        op_name=args.op_name,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not res.get("is_valid"):
        print(f"StableHLO Operation '{args.op_name}' Invalid:")
        for err in res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    print(f"StableHLO Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(res.get('expected_operands', []))}")
    print(f"  Attributes: {', '.join(res.get('expected_attributes', []))}")


def main() -> None:
    """Parse arguments and route to subcommands."""
    parser = argparse.ArgumentParser(description="ML Framework Snapshots CLI")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enforce offline mode (disables network operations and live framework imports)",
    )
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
        choices=["openapi", "json_schema", "pydantic", "protobuf", "llm_prompt"],
        required=True,
        help="Format to export",
    )
    parser_export.set_defaults(func=cmd_export)

    # mcp
    parser_mcp = subparsers.add_parser(
        "mcp", help="Start Model Context Protocol (MCP) JSON-RPC tool server"
    )
    parser_mcp.set_defaults(func=cmd_mcp)

    # check
    parser_check = subparsers.add_parser(
        "check", help="Check compliance of a target implementation against a snapshot"
    )
    parser_check.add_argument(
        "snapshot_json", type=str, help="Reference snapshot JSON file"
    )
    parser_check.add_argument(
        "target_path",
        type=str,
        nargs="+",
        help="Target implementation file or directory",
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

    # list-snapshots
    parser_list = subparsers.add_parser(
        "list-snapshots", help="List available pre-bundled snapshots"
    )
    parser_list.set_defaults(func=cmd_list_snapshots)

    # pull
    parser_pull = subparsers.add_parser(
        "pull",
        help="Download a framework snapshot from GitHub Releases into local cache",
    )
    parser_pull.add_argument(
        "target",
        type=str,
        help="Target framework and version (e.g. torch@2.4.0 or jax@0.4.30)",
    )
    parser_pull.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Custom output directory for downloaded snapshot",
    )
    parser_pull.set_defaults(func=cmd_pull)

    # index
    parser_index = subparsers.add_parser(
        "index", help="Manage ephemeral local SQLite search index"
    )
    parser_index.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from all available snapshots",
    )
    parser_index.add_argument(
        "--clean",
        "--clear",
        dest="clean",
        action="store_true",
        help="Clear the local index database",
    )
    parser_index.add_argument(
        "--status", action="store_true", help="Display index statistics"
    )
    parser_index.set_defaults(func=cmd_index)

    # index-cache
    parser_index_cache = subparsers.add_parser(
        "index-cache", help="Manage and clean ephemeral local SQLite search index"
    )
    parser_index_cache.add_argument(
        "--clean",
        "--clear",
        dest="clean",
        action="store_true",
        help="Clear the local index cache",
    )
    parser_index_cache.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from all available snapshots",
    )
    parser_index_cache.add_argument(
        "--status", action="store_true", help="Display index statistics"
    )
    parser_index_cache.set_defaults(func=cmd_index)

    # check-sass
    parser_check_sass = subparsers.add_parser(
        "check-sass",
        help="Validate NVIDIA SASS assembly instruction or snippet",
    )
    parser_check_sass.add_argument(
        "mnemonic",
        type=str,
        nargs="?",
        default=None,
        help="SASS mnemonic to validate (e.g. FADD, WGMMA)",
    )
    parser_check_sass.add_argument(
        "--operands",
        type=str,
        default=None,
        help="Comma-separated operands (e.g. 'R0,R1,R2')",
    )
    parser_check_sass.add_argument(
        "--modifiers",
        type=str,
        default=None,
        help="Comma-separated modifiers (e.g. '.SAT,.FTZ')",
    )
    parser_check_sass.add_argument(
        "--sm-arch",
        type=str,
        default=None,
        help="Target SM architecture (e.g. 'sm_80', 'sm_90')",
    )
    parser_check_sass.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to SASS assembly file to validate",
    )
    parser_check_sass.set_defaults(func=cmd_check_sass)

    # check-rdna
    parser_check_rdna = subparsers.add_parser(
        "check-rdna",
        help="Validate AMD RDNA assembly instruction or snippet",
    )
    parser_check_rdna.add_argument(
        "mnemonic",
        type=str,
        nargs="?",
        default=None,
        help="RDNA mnemonic to validate (e.g. v_add_f32, v_dual_fma_f32)",
    )
    parser_check_rdna.add_argument(
        "--operands",
        type=str,
        default=None,
        help="Comma-separated operands (e.g. 'v0,v1,v2')",
    )
    parser_check_rdna.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Expected encoding profile (e.g. VOP2, VOP3)",
    )
    parser_check_rdna.add_argument(
        "--modifiers",
        type=str,
        default=None,
        help="Comma-separated modifiers (e.g. '-src,clamp')",
    )
    parser_check_rdna.add_argument(
        "--gfx-arch",
        type=str,
        default=None,
        help="Target GFX architecture (e.g. 'GFX11/RDNA3', 'GFX9/CDNA')",
    )
    parser_check_rdna.add_argument(
        "--wave-size",
        type=int,
        default=None,
        help="Wavefront execution size (32 or 64)",
    )
    parser_check_rdna.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to RDNA assembly file to validate",
    )
    parser_check_rdna.set_defaults(func=cmd_check_rdna)

    # check-mlir
    parser_check_mlir = subparsers.add_parser(
        "check-mlir",
        help="Validate MLIR text or operation",
    )
    parser_check_mlir.add_argument(
        "op_name",
        type=str,
        nargs="?",
        default=None,
        help="Qualified MLIR op name (e.g. arith.addf)",
    )
    parser_check_mlir.add_argument(
        "--operands-count",
        type=int,
        default=None,
        help="Expected number of SSA operands",
    )
    parser_check_mlir.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Comma-separated attribute names",
    )
    parser_check_mlir.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to MLIR text file to validate",
    )
    parser_check_mlir.set_defaults(func=cmd_check_mlir)

    # check-stablehlo
    parser_check_stablehlo = subparsers.add_parser(
        "check-stablehlo",
        help="Validate StableHLO text or operation",
    )
    parser_check_stablehlo.add_argument(
        "op_name",
        type=str,
        nargs="?",
        default=None,
        help="Qualified StableHLO op name (e.g. stablehlo.dot_general)",
    )
    parser_check_stablehlo.add_argument(
        "--operands-count",
        type=int,
        default=None,
        help="Expected number of SSA operands",
    )
    parser_check_stablehlo.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Comma-separated attribute names",
    )
    parser_check_stablehlo.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to StableHLO text file to validate",
    )
    parser_check_stablehlo.set_defaults(func=cmd_check_stablehlo)

    args = parser.parse_args()
    if getattr(args, "offline", False):
        os.environ["ML_SNAPSHOTS_OFFLINE"] = "1"

    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
