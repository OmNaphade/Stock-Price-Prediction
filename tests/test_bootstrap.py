from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from auth.bootstrap import ensure_admin_account
from auth.password_hashing import check_password
from auth.repository import SqliteUserRepository


def _repo(tmp_path) -> SqliteUserRepository:
    return SqliteUserRepository(str(tmp_path / "test_users.db"))


def test_noop_when_admin_email_or_password_unset(tmp_path):
    import config

    repo = _repo(tmp_path)
    with patch("auth.bootstrap.settings", replace(config.settings, admin_email="", admin_password="secret")):
        ensure_admin_account(repo)
    with patch("auth.bootstrap.settings", replace(config.settings, admin_email="admin@example.com", admin_password="")):
        ensure_admin_account(repo)
    assert repo.get_user("admin@example.com") is None


def test_creates_a_verified_admin_account_when_configured(tmp_path):
    import config

    repo = _repo(tmp_path)
    fake_settings = replace(config.settings, admin_email="Admin@Example.com", admin_password="myAdmin@15#13")
    with patch("auth.bootstrap.settings", fake_settings):
        ensure_admin_account(repo)

    user = repo.get_user("admin@example.com")  # normalized to lowercase
    assert user is not None
    assert user.email_verified is True
    assert check_password("myAdmin@15#13", user.password_hash)


def test_never_overwrites_an_existing_account(tmp_path):
    """The core safety property: if the admin already changed their
    password via the normal forgot-password flow, a redeploy must not
    silently reset it back to whatever ADMIN_PASSWORD still says."""
    import config

    repo = _repo(tmp_path)
    fake_settings = replace(config.settings, admin_email="admin@example.com", admin_password="original-pw-123")
    with patch("auth.bootstrap.settings", fake_settings):
        ensure_admin_account(repo)

    # Simulate the admin having since changed their password.
    from auth.password_hashing import hash_password

    repo.update_password("admin@example.com", hash_password("a-new-rotated-password"))

    # Bootstrap runs again (e.g. app restart) with the *same* env var still set.
    with patch("auth.bootstrap.settings", fake_settings):
        ensure_admin_account(repo)

    user = repo.get_user("admin@example.com")
    assert check_password("a-new-rotated-password", user.password_hash)
    assert not check_password("original-pw-123", user.password_hash)
