"""Shared SQLite hardening, used by every repository in this app (auth,
track_record, monitoring) so connection tuning is defined in exactly one
place instead of copy-pasted into three `__init__` methods.

The tuning here targets what a small, always-on, multi-user Streamlit
deployment actually needs from SQLite in production:

- WAL journaling so readers are never blocked behind a writer (the default
  rollback journal takes a lock that stalls concurrent reads).
- A busy-timeout so momentary write contention retries internally instead
  of raising `sqlite3.OperationalError: database is locked` straight to
  the caller.
- `synchronous=NORMAL`, which is the documented safe pairing with WAL
  (durable against application crashes; the tiny extra risk is only an
  OS-level power-loss at the exact moment of a checkpoint) and meaningfully
  cheaper than `FULL` on every write.

None of this makes the on-disk file itself durable across a host restart —
that's a deployment concern (a real volume, or Litestream replication),
not something a PRAGMA can fix. See `litestream.yml` and
`scripts/backup_sqlite.py` for that half of the story.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: str, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open a SQLite connection tuned for production use. Safe to call with
    `check_same_thread=False` callers already guard with their own lock —
    this only changes how the connection behaves, not who may use it."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    if db_path != ":memory:":
        # WAL needs a real file on disk; in-memory DBs (occasionally used
        # in tests) silently keep SQLite's default journal mode instead.
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def integrity_check(conn: sqlite3.Connection) -> bool:
    """SQLite's own consistency check. Cheap enough to run before a backup
    to avoid faithfully replicating a corrupted file."""
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return row is not None and row[0] == "ok"


def backup_to(conn: sqlite3.Connection, dest_path: str) -> None:
    """Hot backup via SQLite's online backup API. Safe to call while `conn`
    is in active use — unlike copying the file on disk, this can't grab a
    torn/mid-write WAL frame."""
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    dest = sqlite3.connect(dest_path)
    try:
        conn.backup(dest)
    finally:
        dest.close()
