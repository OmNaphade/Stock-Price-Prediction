"""Seeds an admin account from ADMIN_EMAIL/ADMIN_PASSWORD (config.py) if
one doesn't already exist — so the operator can log in as admin without
going through the normal register + email-verification flow for their
own account. A no-op if either setting is unset, or if the account
already exists: this only ever *creates*, never overwrites, so a
password changed later via the normal forgot-password flow survives the
next redeploy instead of being silently reset back to whatever these env
vars still say.

Kept separate from AuthService on purpose — seeding an account at
startup is a different concern from the runtime register/login/reset
policy AuthService owns, even though both end up calling the same
UserRepository and password_hashing helpers."""

from __future__ import annotations

from config import log, settings

from .password_hashing import hash_password
from .repository import UserRepository


def ensure_admin_account(repository: UserRepository) -> None:
    if not settings.admin_email or not settings.admin_password:
        return
    email = settings.admin_email.strip().lower()
    if repository.get_user(email) is not None:
        return
    created = repository.create_user(email, hash_password(settings.admin_password), language="en")
    if created:
        repository.mark_email_verified(email)
        log.info("Bootstrapped admin account for %s", email)
