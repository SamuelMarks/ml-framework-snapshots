"""High-performance in-memory grounding engine for anti-hallucination verification."""

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Set

from ml_framework_snapshots.models import ExtendedGhostRef, GhostInspector


def compute_levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Integer edit distance.
    """
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class GroundingEngine:
    """In-memory cache and symbol resolution engine for anti-hallucination verification."""

    def __init__(self, base_dirs: Optional[List[str]] = None) -> None:
        """Initialize the GroundingEngine.

        Args:
            base_dirs: Optional explicit directory paths to search for snapshots.
        """
        default_frameworks = os.path.join(os.path.dirname(__file__), "..", "frameworks")
        default_snapshots = os.path.join(os.path.dirname(__file__), "..", "snapshots")

        self.search_dirs: List[str] = base_dirs or [
            os.path.abspath(default_frameworks),
            os.path.abspath(default_snapshots),
        ]
        # In-memory target registry: target -> {symbol_key: ExtendedGhostRef}
        self._target_cache: Dict[str, Dict[str, ExtendedGhostRef]] = {}
        # Mnemonic/short-name map: target -> {short_name: ExtendedGhostRef}
        self._short_name_cache: Dict[str, Dict[str, ExtendedGhostRef]] = {}

    def get_loaded_targets(self) -> List[str]:
        """Retrieve the names of all currently loaded targets.

        Returns:
            List of target names.
        """
        return sorted(list(self._target_cache.keys()))

    def _discover_target_files(self, target: str) -> List[str]:
        """Locate snapshot or exhaustive files matching a target name.

        Args:
            target: The framework, dialect, or architecture name.

        Returns:
            List of matching file paths.
        """
        norm_target = target.lower().replace("-", "_")
        matched_files: List[str] = []

        for d in self.search_dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                clean_fname = fname.lower()
                is_json = clean_fname.endswith(".json") or clean_fname.endswith(
                    ".json.gz"
                )
                if not is_json:
                    continue

                if clean_fname.startswith(f"{norm_target}_exhaustive."):
                    matched_files.append(os.path.join(d, fname))
                elif (
                    clean_fname.startswith(f"{norm_target}_v")
                    or clean_fname == f"{norm_target}.json"
                ):
                    matched_files.append(os.path.join(d, fname))

        return matched_files

    def load_target(self, target: str) -> Dict[str, ExtendedGhostRef]:
        """Lazily load and index all symbols for a target framework or ISA.

        Args:
            target: The target framework, dialect, or hardware architecture.

        Returns:
            Dictionary mapping symbol keys to their hydrated ExtendedGhostRef objects.
        """
        norm_target = target.lower().replace("-", "_")
        if norm_target in self._target_cache:
            return self._target_cache[norm_target]

        files = self._discover_target_files(norm_target)
        symbol_index: Dict[str, ExtendedGhostRef] = {}
        short_index: Dict[str, ExtendedGhostRef] = {}

        for file_path in files:
            raw_data: Any = None
            if file_path.endswith(".gz"):
                with gzip.open(file_path, "rt", encoding="utf-8") as gf:
                    raw_data = json.load(gf)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

            raw_items: List[Dict[str, Any]] = []
            if isinstance(raw_data, list):
                raw_items = raw_data
            elif isinstance(raw_data, dict):
                if "categories" in raw_data and isinstance(
                    raw_data["categories"], dict
                ):
                    for cat_items in raw_data["categories"].values():
                        if isinstance(cat_items, list):
                            raw_items.extend(cat_items)
                elif "items" in raw_data and isinstance(raw_data["items"], list):
                    raw_items = raw_data["items"]
                else:
                    raw_items = [raw_data]

            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                try:
                    ref = GhostInspector.hydrate(item)
                    if ref.api_path:
                        symbol_index[ref.api_path] = ref
                    if ref.name:
                        symbol_index[ref.name] = ref
                        short_index[ref.name.lower()] = ref
                    # For hardware ISAs, map mnemonic as well
                    mnemonic = item.get("mnemonic")
                    if mnemonic:
                        symbol_index[str(mnemonic)] = ref
                        short_index[str(mnemonic).lower()] = ref
                except Exception:
                    continue

        self._target_cache[norm_target] = symbol_index
        self._short_name_cache[norm_target] = short_index
        return symbol_index

    def has_symbol(self, target: str, symbol: str) -> bool:
        """Check if a symbol exists within the target ground-truth dataset.

        Args:
            target: The framework, dialect, or ISA.
            symbol: The fully-qualified path, mnemonic, or short name.

        Returns:
            True if symbol is present, False otherwise.
        """
        index = self.load_target(target)
        if symbol in index:
            return True
        norm_symbol = symbol.lower()
        short_map = self._short_name_cache.get(target.lower().replace("-", "_"), {})
        return norm_symbol in short_map

    def get_symbol(self, target: str, symbol: str) -> Optional[ExtendedGhostRef]:
        """Retrieve the canonical ExtendedGhostRef for a symbol.

        Args:
            target: The framework, dialect, or ISA.
            symbol: The fully-qualified path, mnemonic, or short name.

        Returns:
            The matched ExtendedGhostRef if found, or None.
        """
        index = self.load_target(target)
        if symbol in index:
            return index[symbol]
        norm_symbol = symbol.lower()
        short_map = self._short_name_cache.get(target.lower().replace("-", "_"), {})
        return short_map.get(norm_symbol)

    def suggest_closest_symbol(
        self, target: str, typo_candidate: str, max_distance: int = 4
    ) -> Optional[str]:
        """Suggest the closest valid symbol using Levenshtein distance matching.

        Args:
            target: The framework, dialect, or ISA.
            typo_candidate: The ungrounded candidate string.
            max_distance: Maximum allowable edit distance for a suggestion.

        Returns:
            Suggested symbol name, or None if no close candidate exists.
        """
        index = self.load_target(target)
        if not index:
            return None

        clean_candidate = typo_candidate.lower()
        best_candidate: Optional[str] = None
        best_distance = max_distance + 1

        # Check against both full paths and short names
        all_candidates: Set[str] = set()
        for ref in index.values():
            if ref.api_path:
                all_candidates.add(ref.api_path)
            if ref.name:
                all_candidates.add(ref.name)
            mnemonic = getattr(ref, "mnemonic", None)
            if mnemonic:
                all_candidates.add(str(mnemonic))

        for cand in sorted(all_candidates):
            dist = compute_levenshtein(clean_candidate, cand.lower())
            if dist < best_distance:
                best_distance = dist
                best_candidate = cand
            elif dist == best_distance and best_candidate is not None:
                # Prefer shorter, more direct candidates on tie
                if len(cand) < len(best_candidate):
                    best_candidate = cand

        return best_candidate if best_distance <= max_distance else None

    def fuzzy_search_symbol(self, target: str, query: str, top_k: int = 5) -> List[str]:
        """Search symbols in target using fuzzy substring and Levenshtein ranking.

        Args:
            target: The framework, dialect, or ISA.
            query: Substring or partial token to search for.
            top_k: Maximum number of ranked suggestions to return.

        Returns:
            Ranked list of matching symbol identifiers.
        """
        index = self.load_target(target)
        if not index:
            return []

        clean_query = query.lower()
        matches: List[tuple[int, str]] = []

        seen: Set[str] = set()
        for ref in index.values():
            for sym in [ref.api_path, ref.name]:
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                sym_lower = sym.lower()

                if clean_query == sym_lower:
                    matches.append((0, sym))
                elif sym_lower.startswith(clean_query):
                    matches.append((1, sym))
                elif clean_query in sym_lower:
                    matches.append((2, sym))
                else:
                    dist = compute_levenshtein(clean_query, sym_lower)
                    if dist <= 3:
                        matches.append((3 + dist, sym))

        matches.sort(key=lambda x: (x[0], len(x[1]), x[1]))
        return [m[1] for m in matches[:top_k]]
