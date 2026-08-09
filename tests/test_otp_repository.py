from __future__ import annotations

from auth.otp_repository import SqliteOtpRepository

_ISSUED_AT = "2098-06-01T00:00:00+00:00"


def _repo(tmp_path) -> SqliteOtpRepository:
    return SqliteOtpRepository(str(tmp_path / "test_users.db"))


def test_issue_then_get_round_trips(tmp_path):
    repo = _repo(tmp_path)
    repo.issue("alice@example.com", "verify_email", "hash1", "2099-01-01T00:00:00+00:00", _ISSUED_AT)

    record = repo.get("alice@example.com", "verify_email")
    assert record is not None
    assert record.code_hash == "hash1"
    assert record.expires_at == "2099-01-01T00:00:00+00:00"
    assert record.issued_at == _ISSUED_AT
    assert record.attempts == 0
    assert record.used_at is None


def test_get_returns_none_for_unknown_pair(tmp_path):
    repo = _repo(tmp_path)
    assert repo.get("nobody@example.com", "verify_email") is None


def test_reissuing_overwrites_the_previous_code_and_resets_state(tmp_path):
    repo = _repo(tmp_path)
    repo.issue("alice@example.com", "verify_email", "hash1", "2099-01-01T00:00:00+00:00", _ISSUED_AT)
    repo.record_failed_attempt("alice@example.com", "verify_email")
    repo.record_failed_attempt("alice@example.com", "verify_email")

    later_issued_at = "2098-07-01T00:00:00+00:00"
    repo.issue("alice@example.com", "verify_email", "hash2", "2099-02-01T00:00:00+00:00", later_issued_at)
    record = repo.get("alice@example.com", "verify_email")
    assert record.code_hash == "hash2"
    assert record.issued_at == later_issued_at
    assert record.attempts == 0
    assert record.used_at is None


def test_different_purposes_for_same_email_are_independent(tmp_path):
    repo = _repo(tmp_path)
    repo.issue("alice@example.com", "verify_email", "hash-verify", "2099-01-01T00:00:00+00:00", _ISSUED_AT)
    repo.issue("alice@example.com", "reset_password", "hash-reset", "2099-01-01T00:00:00+00:00", _ISSUED_AT)

    assert repo.get("alice@example.com", "verify_email").code_hash == "hash-verify"
    assert repo.get("alice@example.com", "reset_password").code_hash == "hash-reset"


def test_record_failed_attempt_increments_counter(tmp_path):
    repo = _repo(tmp_path)
    repo.issue("alice@example.com", "verify_email", "hash1", "2099-01-01T00:00:00+00:00", _ISSUED_AT)
    repo.record_failed_attempt("alice@example.com", "verify_email")
    repo.record_failed_attempt("alice@example.com", "verify_email")
    assert repo.get("alice@example.com", "verify_email").attempts == 2


def test_mark_used_sets_used_at(tmp_path):
    repo = _repo(tmp_path)
    repo.issue("alice@example.com", "verify_email", "hash1", "2099-01-01T00:00:00+00:00", _ISSUED_AT)
    repo.mark_used("alice@example.com", "verify_email")
    assert repo.get("alice@example.com", "verify_email").used_at is not None
