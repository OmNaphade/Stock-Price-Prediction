"""Manual/scheduled hot backup of the app's three SQLite stores.

Run this from cron, a CI job, or by hand:

    python scripts/backup_sqlite.py --out-dir backups/

Each source DB is integrity-checked first — a failing check is reported
and that file is skipped rather than faithfully backing up corruption.
This is the local, dependency-free fallback; `litestream.yml` (continuous
replication to S3-compatible object storage) is the primary defense
against losing data to a host restart in production.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings, log  # noqa: E402
from infra.db import backup_to, connect, integrity_check  # noqa: E402

_DB_PATHS = {
    "users": settings.db_path,
    "track_record": settings.track_record_db_path,
    "monitoring": settings.monitoring_db_path,
}


def backup_all(out_dir: Path) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failures = 0
    for name, db_path in _DB_PATHS.items():
        if not Path(db_path).exists():
            log.info("Skipping %s: %s does not exist yet", name, db_path)
            continue
        conn = connect(db_path)
        try:
            if not integrity_check(conn):
                log.error("Integrity check FAILED for %s (%s) — not backing up", name, db_path)
                failures += 1
                continue
            dest = out_dir / f"{name}-{timestamp}.db"
            backup_to(conn, str(dest))
            log.info("Backed up %s -> %s", db_path, dest)
        finally:
            conn.close()
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="backups", help="Directory to write timestamped .db snapshots into")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = backup_all(out_dir)
    sys.exit(1 if failed else 0)
