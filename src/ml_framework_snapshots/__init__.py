"""ML Framework Snapshots.

Provides an SDK and CLI to dynamically introspect ML frameworks and
capture their API signatures into JSON snapshots.

Attributes:
    extract_snapshot: Extracts a snapshot for a specific framework.
    extract_all_snapshots: Extracts snapshots for all installed frameworks.
    write_snapshot: Saves a snapshot dictionary to a JSON file.

"""

from ml_framework_snapshots.api import (
    extract_snapshot,
    extract_all_snapshots,
    write_snapshot,
)

__all__ = ["extract_snapshot", "extract_all_snapshots", "write_snapshot"]
