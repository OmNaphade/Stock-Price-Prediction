from __future__ import annotations

from infra.db import backup_to, connect, integrity_check


def test_file_backed_connection_uses_wal(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_in_memory_connection_does_not_force_wal():
    # WAL requires a real file; connect() must not blow up on ":memory:".
    conn = connect(":memory:")
    conn.execute("SELECT 1")


def test_busy_timeout_is_applied(tmp_path):
    conn = connect(str(tmp_path / "test.db"), busy_timeout_ms=1234)
    timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout == 1234


def test_integrity_check_passes_on_fresh_db(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    conn.commit()
    assert integrity_check(conn) is True


def test_backup_to_produces_a_readable_copy(tmp_path):
    src_path = tmp_path / "source.db"
    conn = connect(str(src_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t (val) VALUES ('hello')")
    conn.commit()

    dest_path = tmp_path / "backup" / "source-copy.db"
    backup_to(conn, str(dest_path))

    assert dest_path.exists()
    dest_conn = connect(str(dest_path))
    row = dest_conn.execute("SELECT val FROM t").fetchone()
    assert row == ("hello",)
