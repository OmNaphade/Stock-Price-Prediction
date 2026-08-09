"""Auth policy: identity, verification, password reset, and login-attempt
lockout. The UI calls only this — it never touches bcrypt, smtplib, or a
repository directly.

Identity is an email address, not an arbitrary username — every account
is created unverified and can't log in until its owner proves they
control that inbox via an emailed OTP code (register -> verify_email).
Password reset follows the same shape (request_password_reset ->
reset_password_with_code) instead of the old "type any new password and
we'll believe you" flow: that flow had no identity check at all, so
anyone who could guess an account's identifier could take it over.
request_password_reset always returns the same message regardless of
whether the email is registered, so the response itself can't be used to
enumerate accounts.

AuthResult carries a message *key* (+ params to interpolate), not a
rendered string — this service computes what happened, never how to say
it in a given language. auth_ui.py is the only place that turns a key
into displayed text for the *in-app UI* (`i18n.t(result.message_key,
**result.message_params)`), same split as everywhere else in this app
where a service computes meaning and a page only renders it.

The one exception is outbound *email* content: nobody is browsing a
session when a resend or a reset email goes out, so there's no
`st.session_state` to read a language from. Instead, each account stores
the language it was registered under (`UserRecord.language`, set by
whatever the UI's sidebar selector said at that moment — see
auth_ui.py), and this service passes that plain string into `i18n.t()`
explicitly (`lang=...`). That's still not a Streamlit dependency: `t()`
takes a language *value*, not a session to inspect, the same as it takes
`code`/`minutes` as plain values — nothing here imports `streamlit`.

`_send_otp` enforces a cooldown (`settings.otp_resend_cooldown_seconds`)
between two emails for the same (email, purpose) — verified live that
without one, nothing stopped 50 password-reset requests for the same
account firing in under a second, each one a real email. This throttles
that *account-targeted* abuse (bombing one inbox via resend/reset).
It deliberately does not throttle *how many distinct accounts* can be
registered — that's IP-level/CAPTCHA territory this service has no
request context for (it's transport-agnostic on purpose, like every
other service in this app); a reverse proxy or hosting platform is the
right layer for that half of the problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import bcrypt

from config import settings
from i18n import t

from .email_sender import EmailSender
from .otp import generate_code, hash_code, verify_code
from .otp_repository import OtpRepository
from .repository import UserRepository, minutes_from_now_iso, utcnow_iso

_VERIFY_EMAIL = "verify_email"
_RESET_PASSWORD = "reset_password"
_INVALID_OR_EXPIRED_CODE = "auth.invalid_or_expired_code"


@dataclass
class AuthResult:
    success: bool
    message_key: str
    message_params: dict = field(default_factory=dict)


def _looks_like_email(value: str) -> bool:
    if not value or " " in value or value.count("@") != 1:
        return False
    if len(value) > settings.max_email_length:
        return False
    local, _, domain = value.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")


def _validate_password(password: str) -> Optional[tuple[str, dict]]:
    """None if `password` satisfies policy, otherwise (message_key,
    params) explaining why not. Length only, deliberately — composition
    rules (forced uppercase/digit/symbol) are out of favor per NIST
    800-63B and mostly just push people toward predictable substitutions.
    The max is bcrypt's real limit: verified live that this bcrypt build
    doesn't error past 72 bytes, it silently truncates, so two different
    100-character passwords sharing the same first 72 bytes would
    otherwise both be accepted as if genuinely different credentials."""
    if not password or not password.strip():
        return ("auth.enter_valid_email_password", {})
    if len(password) < settings.min_password_length:
        return ("auth.password_too_short", {"min_length": settings.min_password_length})
    if len(password.encode("utf-8")) > settings.max_password_length_bytes:
        return ("auth.password_too_long", {})
    return None


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


class AuthService:
    def __init__(
        self,
        repository: UserRepository,
        otp_repository: OtpRepository,
        email_sender: EmailSender,
    ):
        self._repo = repository
        self._otp_repo = otp_repository
        self._email_sender = email_sender

    def register(self, email: str, password: str, language: str = "en") -> AuthResult:
        email = email.strip().lower()
        if not _looks_like_email(email):
            return AuthResult(False, "auth.enter_valid_email_password")
        password_error = _validate_password(password)
        if password_error:
            return AuthResult(False, *password_error)
        created = self._repo.create_user(email, _hash_password(password), language)
        if not created:
            return AuthResult(False, "auth.email_exists")
        self._send_otp(
            email, _VERIFY_EMAIL, language, "auth.email_verify_subject", "auth.email_verify_body"
        )
        return AuthResult(True, "auth.registered_check_email")

    def verify_email(self, email: str, code: str) -> AuthResult:
        email = email.strip().lower()
        check = self._check_otp(email, _VERIFY_EMAIL, code)
        if not check.success:
            return check
        self._repo.mark_email_verified(email)
        return AuthResult(True, "auth.email_verified_please_login")

    def resend_verification(self, email: str) -> AuthResult:
        email = email.strip().lower()
        user = self._repo.get_user(email)
        if user is None:
            return AuthResult(False, "auth.email_not_found")
        if user.email_verified:
            return AuthResult(False, "auth.already_verified")
        sent = self._send_otp(
            email, _VERIFY_EMAIL, user.language, "auth.email_verify_subject", "auth.email_verify_body"
        )
        if not sent:
            return AuthResult(
                False, "auth.please_wait_before_resend", {"seconds": settings.otp_resend_cooldown_seconds}
            )
        return AuthResult(True, "auth.registered_check_email")

    def login(self, email: str, password: str) -> AuthResult:
        email = email.strip().lower()
        user = self._repo.get_user(email)
        if user is None:
            return AuthResult(False, "auth.invalid_credentials")
        if not user.email_verified:
            return AuthResult(False, "auth.email_not_verified")

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
                    False, "auth.locked_try_again_in", {"minutes_left": minutes_left}
                )
            self._repo.reset_login_attempts(email)
            effective_failed_attempts = 0

        if _check_password(password, user.password_hash):
            self._repo.record_successful_login(email)
            return AuthResult(True, "auth.logged_in")

        attempts = effective_failed_attempts + 1
        locked_until = None
        if attempts >= settings.max_login_attempts:
            locked_until = minutes_from_now_iso(settings.lockout_minutes)
        self._repo.record_failed_login(email, locked_until)

        if locked_until:
            return AuthResult(
                False,
                "auth.locked_for_minutes",
                {"lockout_minutes": settings.lockout_minutes},
            )
        remaining = settings.max_login_attempts - attempts
        return AuthResult(
            False, "auth.invalid_credentials_attempts_left", {"remaining": remaining}
        )

    def request_password_reset(self, email: str) -> AuthResult:
        email = email.strip().lower()
        user = self._repo.get_user(email)
        if user is not None:
            # Return value intentionally ignored: whether this actually
            # sent a new email or silently suppressed a duplicate one
            # (cooldown) must not change the response below — otherwise a
            # "please wait" vs "sent" distinction would itself leak
            # whether the email is registered.
            self._send_otp(
                email, _RESET_PASSWORD, user.language, "auth.email_reset_subject", "auth.email_reset_body"
            )
        # Deliberately the same result whether or not the email is
        # registered — the response itself must not reveal which emails
        # have accounts.
        return AuthResult(True, "auth.reset_code_sent_if_exists")

    def reset_password_with_code(self, email: str, code: str, new_password: str) -> AuthResult:
        email = email.strip().lower()
        password_error = _validate_password(new_password)
        if password_error:
            return AuthResult(False, *password_error)
        check = self._check_otp(email, _RESET_PASSWORD, code)
        if not check.success:
            return check
        updated = self._repo.update_password(email, _hash_password(new_password))
        if not updated:
            return AuthResult(False, "auth.email_not_found")
        return AuthResult(True, "auth.password_reset_successful")

    def _send_otp(self, email: str, purpose: str, language: str, subject_key: str, body_key: str) -> bool:
        """True if an email was actually sent. False means the cooldown
        suppressed it — a real code was already issued too recently for
        this (email, purpose) to send another one."""
        if self._recently_issued(email, purpose):
            return False
        code = generate_code(settings.otp_code_length)
        expires_at = minutes_from_now_iso(settings.otp_expiry_minutes)
        self._otp_repo.issue(email, purpose, hash_code(code), expires_at, utcnow_iso())
        subject = t(subject_key, lang=language)
        body = t(body_key, lang=language, code=code, minutes=settings.otp_expiry_minutes)
        self._email_sender.send(email, subject, body)
        return True

    def _recently_issued(self, email: str, purpose: str) -> bool:
        record = self._otp_repo.get(email, purpose)
        if record is None:
            return False
        issued_at = datetime.fromisoformat(record.issued_at)
        elapsed = (datetime.now(timezone.utc) - issued_at).total_seconds()
        return elapsed < settings.otp_resend_cooldown_seconds

    def _check_otp(self, email: str, purpose: str, code: str) -> AuthResult:
        record = self._otp_repo.get(email, purpose)
        if record is None or record.used_at is not None:
            return AuthResult(False, _INVALID_OR_EXPIRED_CODE)
        if datetime.fromisoformat(record.expires_at) < datetime.now(timezone.utc):
            return AuthResult(False, _INVALID_OR_EXPIRED_CODE)
        if record.attempts >= settings.otp_max_attempts:
            return AuthResult(False, "auth.too_many_code_attempts")
        if not verify_code(code, record.code_hash):
            self._otp_repo.record_failed_attempt(email, purpose)
            return AuthResult(False, _INVALID_OR_EXPIRED_CODE)
        self._otp_repo.mark_used(email, purpose)
        return AuthResult(True, "")
