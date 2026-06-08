"""Snapshot Diffing Module.

Provides functions to diff two snapshots and detect additions, removals,
and signature changes.
"""

from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field


class DiffResult(BaseModel):
    """Result of diffing two snapshots."""

    added: List[str] = Field(
        default_factory=list, description="List of added api_paths."
    )
    removed: List[str] = Field(
        default_factory=list, description="List of removed api_paths."
    )
    signature_changed: List[str] = Field(
        default_factory=list, description="List of api_paths with changed signatures."
    )
    breaking_signature_changed: List[str] = Field(
        default_factory=list,
        description="List of api_paths with breaking signature changes.",
    )
    non_breaking_signature_changed: List[str] = Field(
        default_factory=list,
        description="List of api_paths with non-breaking signature changes.",
    )


def _is_breaking_change(
    p1_list: List[Dict[str, Any]], p2_list: List[Dict[str, Any]]
) -> bool:
    """Determine if a signature change is backwards-incompatible.

    Args:
        p1_list: Old parameter list.
        p2_list: New parameter list.

    Returns:
        True if breaking, False otherwise.

    """
    p1_map = {p.get("name"): p for p in p1_list}
    p2_map = {p.get("name"): p for p in p2_list}

    # Parameter removed?
    for name in p1_map:
        if name not in p2_map:
            return True

    for name, p2 in p2_map.items():
        if name not in p1_map:
            # Added parameter without a default is breaking
            if p2.get("default") is None and p2.get("kind") not in (
                "VAR_POSITIONAL",
                "VAR_KEYWORD",
            ):
                return True
        else:
            p1 = p1_map[name]
            k1 = p1.get("kind")
            k2 = p2.get("kind")

            # Kind changed?
            if k1 != k2:
                # If k2 is more restrictive, it's breaking.
                if (
                    k2 in ("POSITIONAL_ONLY", "KEYWORD_ONLY")
                    and k1 == "POSITIONAL_OR_KEYWORD"
                ):
                    return True
                # Changing between pos-only and kw-only is breaking
                if k1 in ("POSITIONAL_ONLY", "KEYWORD_ONLY") and k2 in (
                    "POSITIONAL_ONLY",
                    "KEYWORD_ONLY",
                ):
                    return True
                # Changing to/from VAR_*
                if ("VAR" in str(k1) and "VAR" not in str(k2)) or (
                    "VAR" not in str(k1) and "VAR" in str(k2)
                ):
                    return True

            # Default removed?
            if p1.get("default") is not None and p2.get("default") is None:
                return True

            # Default changed? Subtle breakage.
            if p1.get("default") != p2.get("default"):
                return True

    return False


def diff_snapshots(snap1: Dict[str, Any], snap2: Dict[str, Any]) -> DiffResult:
    """Diffs two snapshots to find changes.

    Args:
        snap1: The older snapshot data.
        snap2: The newer snapshot data.

    Returns:
        A DiffResult containing the differences.

    """

    def _flatten(snap: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Flatten a snapshot dictionary into a mapping of API paths to items.

        Args:
            snap: The snapshot.

        Returns:
            The flattened dict.
        """
        flattened = {}
        for cat, items in snap.get("categories", {}).items():
            for item in items:
                flattened[item["api_path"]] = item
        return flattened

    flat1 = _flatten(snap1)
    flat2 = _flatten(snap2)

    added = []
    removed = []
    signature_changed = []
    breaking_signature_changed = []
    non_breaking_signature_changed = []

    for path, item2 in flat2.items():
        is_public = item2.get("is_public", True)
        display_path = path if is_public else f"{path} (non-public)"
        if path not in flat1:
            added.append(display_path)
        else:
            item1 = flat1[path]
            # check if signature changed by comparing parameters
            p1 = item1.get("params", [])
            p2 = item2.get("params", [])

            def sig_tuple(p: Dict[str, Any]) -> Tuple[Any, ...]:
                """Convert a parameter dictionary to a tuple for easy comparison.

                Args:
                    p: The parameter dictionary.

                Returns:
                    A tuple representation.
                """
                return (
                    p.get("name"),
                    p.get("kind"),
                    p.get("default"),
                    p.get("annotation"),
                )

            sig1 = [sig_tuple(p) for p in p1]
            sig2 = [sig_tuple(p) for p in p2]

            # evaluate if non-public changes are breaking
            if sig1 != sig2:
                signature_changed.append(display_path)
                if _is_breaking_change(p1, p2):
                    breaking_signature_changed.append(display_path)
                else:
                    non_breaking_signature_changed.append(display_path)

    for path, item1 in flat1.items():
        if path not in flat2:
            is_public = item1.get("is_public", True)
            display_path = path if is_public else f"{path} (non-public)"
            removed.append(display_path)

    return DiffResult(
        added=sorted(added),
        removed=sorted(removed),
        signature_changed=sorted(signature_changed),
        breaking_signature_changed=sorted(breaking_signature_changed),
        non_breaking_signature_changed=sorted(non_breaking_signature_changed),
    )


def generate_changelog(diff: DiffResult) -> str:
    """Generate a markdown changelog from a DiffResult.

    Args:
        diff: The DiffResult to generate a changelog for.

    Returns:
        A markdown formatted changelog string.

    """
    lines = ["# Changelog Report", ""]

    if not diff.added and not diff.removed and not diff.signature_changed:
        lines.append("No changes detected.")
        return "\n".join(lines)

    if diff.added:
        lines.append("## Added")
        for path in diff.added:
            lines.append(f"- `{path}`")
        lines.append("")

    if diff.removed:
        lines.append("## Removed (Breaking)")
        for path in diff.removed:
            lines.append(f"- `{path}`")
        lines.append("")

    if diff.breaking_signature_changed:
        lines.append("## Breaking Signature Changes")
        for path in diff.breaking_signature_changed:
            lines.append(f"- `{path}`")
        lines.append("")

    if diff.non_breaking_signature_changed:
        lines.append("## Non-Breaking Signature Changes")
        for path in diff.non_breaking_signature_changed:
            lines.append(f"- `{path}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
