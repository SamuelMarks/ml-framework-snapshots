"""Ephemeral local SQLite search index for fast symbol lookups.

Provides on-demand SQLite FTS5 indexing over bundled and locally cached
snapshot JSON files, enabling sub-millisecond lookups and fuzzy searches
without loading multi-megabyte JSON trees into memory.
"""

import hashlib
import json
import os
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple, cast


def get_cache_dir() -> str:
    """Retrieve the platform-specific cache directory for ml-framework-snapshots.

    Returns:
        Absolute path to the user's local cache directory.
    """
    base = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_CACHE_DIR")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            base = os.path.join(xdg, "ml_framework_snapshots")
        else:
            base = os.path.join(
                os.path.expanduser("~"), ".cache", "ml_framework_snapshots"
            )
    try:
        os.makedirs(base, exist_ok=True)
    except FileNotFoundError:
        base = os.path.join(tempfile.gettempdir(), "ml_framework_cache")
        try:
            os.makedirs(base, exist_ok=True)
        except (FileExistsError, OSError):
            pass
    except FileExistsError:
        pass
    return base


def get_index_db_path() -> str:
    """Retrieve the path to the ephemeral SQLite index database.

    Returns:
        Absolute path to the index database file.
    """
    return os.path.join(get_cache_dir(), "index.db")


def compute_file_sha256(file_path: str) -> str:
    """Compute the SHA256 checksum of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA256 checksum string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Initialize the SQLite index database connection and schema.

    Args:
        db_path: Optional path to the database file (defaults to get_index_db_path()).

    Returns:
        Open sqlite3 connection with schema initialized.
    """
    path = db_path or get_index_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError:
        pass

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                framework TEXT NOT NULL,
                version TEXT NOT NULL,
                symbol_count INTEGER NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                framework TEXT NOT NULL,
                version TEXT NOT NULL,
                api_path TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                docstring TEXT,
                json_data TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbols_lookup
            ON symbols(framework, api_path)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbols_name
            ON symbols(framework, name)
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                api_path,
                name,
                docstring,
                content='symbols',
                content_rowid='id'
            )
            """
        )

    return conn


def get_readonly_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a read-only SQLite connection to the index database with concurrency timeout.

    Args:
        db_path: Optional path to SQLite database.

    Returns:
        Configured read-only sqlite3.Connection.
    """
    path = db_path or get_index_db_path()
    if not os.path.exists(path):
        return init_db(path)
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn
    except sqlite3.OperationalError:
        return init_db(path)


def clean_index_cache(db_path: Optional[str] = None) -> int:
    """Remove local ephemeral index database and auxiliary journal/WAL files.

    Args:
        db_path: Optional path to SQLite database file.

    Returns:
        Number of removed cache files.
    """
    path = db_path or get_index_db_path()
    count = 0
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = f"{path}{suffix}"
        if os.path.exists(p):
            try:
                os.remove(p)
                count += 1
            except OSError:
                pass
    return count


def extract_framework_and_version(file_name: str) -> Tuple[str, str]:
    """Extract framework identifier and version string from snapshot filename.

    Args:
        file_name: Filename of the snapshot (e.g. 'torch_v2.2.0.json').

    Returns:
        Tuple of (framework, version).
    """
    base = file_name
    if base.endswith(".json"):
        base = base[:-5]

    if "_v" in base:
        parts = base.split("_v", 1)
        return parts[0], parts[1]
    if "_exhaustive" in base:
        fw = base.replace("_exhaustive", "")
        return fw, "latest"
    if "_" in base:
        parts = base.split("_", 1)
        return parts[0], parts[1]

    return base, "latest"


def index_snapshot_file(json_path: str, conn: sqlite3.Connection) -> int:
    """Index a single snapshot JSON file into the database.

    Args:
        json_path: Path to the snapshot JSON file.
        conn: Open SQLite connection.

    Returns:
        Number of symbols indexed.
    """
    if not os.path.isfile(json_path):
        return 0

    file_hash = compute_file_sha256(json_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT file_hash FROM indexed_files WHERE file_path = ?",
        (json_path,),
    )
    row = cur.fetchone()
    if row and row["file_hash"] == file_hash:
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = os.path.basename(json_path)
    framework, version = extract_framework_and_version(file_name)

    items: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if "categories" in data:
            for _cat, cat_items in data["categories"].items():
                if isinstance(cat_items, list):
                    items.extend(cat_items)
        elif "items" in data and isinstance(data["items"], list):
            items.extend(data["items"])
    elif isinstance(data, list):
        items.extend(data)

    with conn:
        # Clear previous records for this file
        cur.execute("SELECT id FROM symbols WHERE file_path = ?", (json_path,))
        old_ids = [r["id"] for r in cur.fetchall()]
        if old_ids:
            for oid in old_ids:
                cur.execute(
                    "INSERT INTO symbols_fts(symbols_fts, rowid, api_path, name, docstring) VALUES('delete', ?, '', '', '')",
                    (oid,),
                )
            cur.execute("DELETE FROM symbols WHERE file_path = ?", (json_path,))

        inserted = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            api_path = (
                item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
            )
            name = item.get("name") or item.get("mnemonic") or api_path.split(".")[-1]
            kind = item.get("kind", "function")
            doc = item.get("docstring", "")
            raw_json = json.dumps(item)

            cur.execute(
                """
                INSERT INTO symbols (file_path, framework, version, api_path, name, kind, docstring, json_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (json_path, framework, version, api_path, name, kind, doc, raw_json),
            )
            sym_id = cur.lastrowid
            cur.execute(
                """
                INSERT INTO symbols_fts (rowid, api_path, name, docstring)
                VALUES (?, ?, ?, ?)
                """,
                (sym_id, api_path, name, doc),
            )
            inserted += 1

        cur.execute(
            """
            INSERT OR REPLACE INTO indexed_files (file_path, file_hash, framework, version, symbol_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (json_path, file_hash, framework, version, inserted),
        )

    return inserted


def get_available_snapshot_files() -> List[str]:
    """Discover all bundled and locally cached snapshot JSON files.

    Returns:
        List of absolute file paths to discovered JSON files.
    """
    found: List[str] = []
    base_dir = os.path.dirname(__file__)
    search_dirs = [
        os.path.join(base_dir, "snapshots"),
        os.path.join(base_dir, "frameworks"),
        os.path.join(get_cache_dir(), "snapshots"),
    ]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for fname in sorted(os.listdir(sdir)):
                if fname.endswith(".json"):
                    found.append(os.path.join(sdir, fname))
    return found


def ensure_index(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Ensure the local SQLite index is initialized and up to date.

    Args:
        db_path: Optional path to the database file.

    Returns:
        Active sqlite3 connection ready for querying.
    """
    conn = init_db(db_path)
    snapshot_files = get_available_snapshot_files()
    for sfile in snapshot_files:
        try:
            index_snapshot_file(sfile, conn)
        except Exception:
            pass
    return conn


def search_index(
    query: str,
    framework: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search for operations using the SQLite FTS5 index.

    Args:
        query: Full-text search term or mnemonic.
        framework: Optional framework filter.
        version: Optional version filter.
        limit: Maximum results to return.
        db_path: Optional custom database path.

    Returns:
        List of serialized symbol dictionaries.
    """
    conn = ensure_index(db_path)
    cur = conn.cursor()

    clean_q = query.replace("'", " ").replace('"', " ").strip()
    if not clean_q:
        return []

    # FTS query with prefix wildcard
    fts_term = f'"{clean_q}"*'

    conditions = ["symbols_fts MATCH ?"]
    params: List[Any] = [fts_term]

    if framework:
        conditions.append("s.framework = ?")
        params.append(framework.lower().strip())
    if version:
        conditions.append("s.version = ?")
        params.append(version.strip())

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT s.json_data
        FROM symbols_fts f
        JOIN symbols s ON f.rowid = s.id
        WHERE {where_clause}
        ORDER BY rank
        LIMIT ?
    """
    params.append(limit)

    cur.execute(sql, params)
    rows = cur.fetchall()
    results = []
    for r in rows:
        try:
            results.append(json.loads(r["json_data"]))
        except Exception:
            pass
    return results


def lookup_symbol(
    framework: str,
    api_path: str,
    version: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Look up exact symbol definition by framework and API path.

    Args:
        framework: Target framework name.
        api_path: Canonical API path or mnemonic.
        version: Optional framework version.
        db_path: Optional database path.

    Returns:
        Deserialized GhostRef dict, or None if not found.
    """
    conn = ensure_index(db_path)
    cur = conn.cursor()

    clean_fw = framework.lower().strip()
    conditions = ["framework = ?", "(api_path = ? OR name = ?)"]
    params: List[Any] = [clean_fw, api_path, api_path]

    if version:
        conditions.append("version = ?")
        params.append(version.strip())

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT json_data
        FROM symbols
        WHERE {where_clause}
        ORDER BY id DESC
        LIMIT 1
    """
    cur.execute(sql, params)
    row = cur.fetchone()
    if row:
        return cast(Dict[str, Any], json.loads(row["json_data"]))
    return None


def clear_index(db_path: Optional[str] = None) -> bool:
    """Clear and delete the local ephemeral index database file.

    Args:
        db_path: Optional database path.

    Returns:
        True if cleared or absent, False if removal failed.
    """
    path = db_path or get_index_db_path()
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except OSError:
            return False
    return True
