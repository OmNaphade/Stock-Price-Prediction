"""Auth policy: hashing, verification, and login-attempt lockout. The UI
calls only this — it never touches bcrypt or the repository directly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import bcrypt

from config import settings

from .repository import UserRepository, minutes_from_now_iso


@dataclass
class AuthResult:
    success: bool
    message: str


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


class AuthService:
    def __init__(self, repository: UserRepository):
        self._repo = repository

    def register(self, username: str, password: str) -> AuthResult:
        if not username or not password:
            return AuthResult(False, "Enter both username and password.")
        created = self._repo.create_user(username, _hash_password(password))
        if created:
            return AuthResult(True, "Registered! Please log in.")
        return AuthResult(False, "Username already exists.")

    def login(self, username: str, password: str) -> AuthResult:
        user = self._repo.get_user(username)
        if user is None:
            return AuthResult(False, "Invalid username or password.")

        # An expired lockout must actually clear failed_attempts in storage,
        # not just be treated as 0 for this one response: record_failed_login
        # increments failed_attempts *relative to what's already stored*, so
        # without this reset, a stale count of e.g. 5 would climb to 6, 7...
        # on every retry and re-lock the account on the very next miss,
        # forever, since only a successful login otherwise clears it.
        effective_failed_attempts = user.failed_attempts
        if user.locked_until:
            locked_until = datetime.fromisoformat(user.locked_until)
            now = datetime.now(timezone.utc)
            if locked_until > now:
                minutes_left = max(1, int((locked_until - now).total_seconds() / 60))
                return AuthResult(
                    False, f"Too many failed attempts. Try again in {minutes_left} min."
                )
            self._repo.reset_login_attempts(username)
            effective_failed_attempts = 0

        if _check_password(password, user.password_hash):
            self._repo.record_successful_login(username)
            return AuthResult(True, "Logged in!")

        attempts = effective_failed_attempts + 1
        locked_until = None
        if attempts >= settings.max_login_attempts:
            locked_until = minutes_from_now_iso(settings.lockout_minutes)
        self._repo.record_failed_login(username, locked_until)

        if locked_until:
            return AuthResult(
                False,
                f"Too many failed attempts. Locked for {settings.lockout_minutes} min.",
            )
        remaining = settings.max_login_attempts - attempts
        return AuthResult(False, f"Invalid username or password. {remaining} attempt(s) left.")

    def reset_password(self, username: str, new_password: str) -> AuthResult:
        if not new_password:
            return AuthResult(False, "Enter a new password.")
        updated = self._repo.update_password(username, _hash_password(new_password))
        if updated:
            return AuthResult(True, "Password reset successful.")
        return AuthResult(False, "Username not found.")
