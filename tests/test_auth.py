from __future__ import annotations

from datetime import datetime, timedelta, timezone

from auth.repository import SqliteUserRepository
from auth.service import AuthService


def _service(tmp_path) -> AuthService:
    db_path = str(tmp_path / "test_users.db")
    return AuthService(SqliteUserRepository(db_path))


def _service_with_repo(tmp_path) -> tuple[AuthService, SqliteUserRepository]:
    repo = SqliteUserRepository(str(tmp_path / "test_users.db"))
    return AuthService(repo), repo


def test_register_then_login_succeeds(tmp_path):
    service = _service(tmp_path)
    assert service.register("alice", "correct-horse").success

    result = service.login("alice", "correct-horse")
    assert result.success


def test_duplicate_registration_fails(tmp_path):
    service = _service(tmp_path)
    service.register("alice", "pw1")
    result = service.register("alice", "pw2")
    assert not result.success


def test_wrong_password_fails_without_locking_out_immediately(tmp_path):
    service = _service(tmp_path)
    service.register("bob", "correct-horse")

    result = service.login("bob", "wrong-password")
    assert not result.success
    # One bad attempt shouldn't lock the account.
    assert service.login("bob", "correct-horse").success


def test_lockout_after_repeated_failures(tmp_path):
    from config import settings

    service = _service(tmp_path)
    service.register("carol", "correct-horse")

    for _ in range(settings.max_login_attempts):
        service.login("carol", "wrong-password")

    result = service.login("carol", "correct-horse")
    assert not result.success
    assert "locked" in result.message.lower() or "attempts" in result.message.lower()


def test_reset_password_allows_login_with_new_password(tmp_path):
    service = _service(tmp_path)
    service.register("dave", "old-password")

    assert service.reset_password("dave", "new-password").success
    assert service.login("dave", "new-password").success
    assert not service.login("dave", "old-password").success


def test_reset_password_unknown_user_fails(tmp_path):
    service = _service(tmp_path)
    result = service.reset_password("ghost", "whatever")
    assert not result.success


def test_expired_lockout_grants_fresh_attempts_not_an_immediate_relock(tmp_path):
    """Regression test: failed_attempts must actually be cleared in storage
    once a lockout window has passed — not just treated as 0 for a single
    response — otherwise record_failed_login's relative '+1' keeps climbing
    on top of the stale stored count and every subsequent miss re-locks the
    account immediately, forever, since only a successful login clears it."""
    service, repo = _service_with_repo(tmp_path)
    service.register("erin", "correct-horse")

    # Simulate a lockout that has already expired (set directly via the
    # repository rather than sleeping in the test).
    already_expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    repo.record_failed_login("erin", already_expired)
    for _ in range(4):
        repo.record_failed_login("erin", already_expired)
    assert repo.get_user("erin").failed_attempts == 5

    # One wrong attempt after expiry should NOT immediately re-lock —
    # it should be treated as the first of a fresh set of attempts.
    result = service.login("erin", "still-wrong")
    assert not result.success
    assert "locked" not in result.message.lower()
    assert repo.get_user("erin").failed_attempts == 1

    # And the correct password still works right after.
    assert service.login("erin", "correct-horse").success
