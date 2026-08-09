from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from auth.otp_repository import SqliteOtpRepository
from auth.repository import SqliteUserRepository
from auth.service import AuthService

# No \b word-boundary anchors: Unicode-aware \b treats CJK ideographs as
# word characters too, so a code adjoining Chinese/Japanese/Korean text
# with no separating punctuation wouldn't have a boundary on that side.
_CODE_RE = re.compile(r"(\d{6})")


class FakeEmailSender:
    """Captures every send() call instead of hitting a real mailbox, so
    tests can pull the OTP code straight out of the message body."""

    def __init__(self):
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to_address: str, subject: str, body: str) -> bool:
        self.sent.append((to_address, subject, body))
        return True

    def last_code(self) -> str:
        match = _CODE_RE.search(self.sent[-1][2])
        assert match, f"no 6-digit code found in last email body: {self.sent[-1][2]!r}"
        return match.group(1)


def _service(tmp_path) -> tuple[AuthService, FakeEmailSender]:
    db_path = str(tmp_path / "test_users.db")
    sender = FakeEmailSender()
    service = AuthService(SqliteUserRepository(db_path), SqliteOtpRepository(db_path), sender)
    return service, sender


def _service_with_repos(tmp_path):
    db_path = str(tmp_path / "test_users.db")
    user_repo = SqliteUserRepository(db_path)
    otp_repo = SqliteOtpRepository(db_path)
    sender = FakeEmailSender()
    service = AuthService(user_repo, otp_repo, sender)
    return service, user_repo, otp_repo, sender


def _register_and_verify(service, sender, email, password) -> None:
    assert service.register(email, password).success
    code = sender.last_code()
    assert service.verify_email(email, code).success


def test_register_sends_a_verification_code_and_blocks_login_until_verified(tmp_path):
    service, sender = _service(tmp_path)
    result = service.register("alice@example.com", "correct-horse")
    assert result.success
    assert result.message_key == "auth.registered_check_email"
    assert len(sender.sent) == 1
    assert sender.sent[0][0] == "alice@example.com"

    blocked = service.login("alice@example.com", "correct-horse")
    assert not blocked.success
    assert blocked.message_key == "auth.email_not_verified"


def test_verify_then_login_succeeds(tmp_path):
    service, sender = _service(tmp_path)
    _register_and_verify(service, sender, "alice@example.com", "correct-horse")

    result = service.login("alice@example.com", "correct-horse")
    assert result.success


def test_wrong_verification_code_fails(tmp_path):
    service, sender = _service(tmp_path)
    service.register("alice@example.com", "correct-horse")

    result = service.verify_email("alice@example.com", "000000")
    assert not result.success
    assert result.message_key == "auth.invalid_or_expired_code"


def test_verification_code_locks_out_after_too_many_wrong_attempts(tmp_path):
    from config import settings

    service, sender = _service(tmp_path)
    service.register("alice@example.com", "correct-horse")
    real_code = sender.last_code()

    for _ in range(settings.otp_max_attempts):
        service.verify_email("alice@example.com", "000000")

    result = service.verify_email("alice@example.com", real_code)
    assert not result.success
    assert result.message_key == "auth.too_many_code_attempts"


def test_resend_verification_issues_a_new_code(tmp_path, monkeypatch):
    from dataclasses import replace

    import config

    service, sender = _service(tmp_path)
    service.register("alice@example.com", "correct-horse")
    first_code = sender.last_code()

    # Bypass the resend cooldown (covered on its own in
    # TestOtpResendRateLimiting) — this test is about a resend issuing a
    # genuinely new, independently-valid code, not about throttling.
    monkeypatch.setattr("auth.service.settings", replace(config.settings, otp_resend_cooldown_seconds=0))
    result = service.resend_verification("alice@example.com")
    assert result.success
    second_code = sender.last_code()

    # The old code must no longer work once a new one has been issued.
    assert not service.verify_email("alice@example.com", first_code).success
    assert service.verify_email("alice@example.com", second_code).success


def test_resend_verification_for_already_verified_email_fails(tmp_path):
    service, sender = _service(tmp_path)
    _register_and_verify(service, sender, "alice@example.com", "correct-horse")

    result = service.resend_verification("alice@example.com")
    assert not result.success
    assert result.message_key == "auth.already_verified"


def test_duplicate_registration_fails(tmp_path):
    service, sender = _service(tmp_path)
    service.register("alice@example.com", "password1")
    result = service.register("alice@example.com", "password2")
    assert not result.success
    assert result.message_key == "auth.email_exists"


def test_registration_rejects_invalid_email(tmp_path):
    service, sender = _service(tmp_path)
    result = service.register("not-an-email", "correct-horse")
    assert not result.success
    assert result.message_key == "auth.enter_valid_email_password"
    assert sender.sent == []


def test_registration_rejects_email_over_max_length(tmp_path):
    from config import settings

    service, sender = _service(tmp_path)
    huge_email = ("a" * (settings.max_email_length + 1)) + "@example.com"
    result = service.register(huge_email, "correct-horse")
    assert not result.success
    assert result.message_key == "auth.enter_valid_email_password"
    assert sender.sent == []


class TestPasswordPolicy:
    def test_password_below_minimum_length_is_rejected(self, tmp_path):
        from config import settings

        service, sender = _service(tmp_path)
        short = "a" * (settings.min_password_length - 1)
        result = service.register("alice@example.com", short)
        assert not result.success
        assert result.message_key == "auth.password_too_short"
        assert result.message_params == {"min_length": settings.min_password_length}
        assert sender.sent == []

    def test_password_at_exactly_minimum_length_is_accepted(self, tmp_path):
        from config import settings

        service, sender = _service(tmp_path)
        exact = "a" * settings.min_password_length
        result = service.register("alice@example.com", exact)
        assert result.success

    def test_whitespace_only_password_is_rejected(self, tmp_path):
        service, sender = _service(tmp_path)
        result = service.register("alice@example.com", "        ")
        assert not result.success
        assert sender.sent == []

    def test_password_over_bcrypt_byte_limit_is_rejected(self, tmp_path):
        from config import settings

        service, sender = _service(tmp_path)
        too_long = "a" * (settings.max_password_length_bytes + 1)
        result = service.register("alice@example.com", too_long)
        assert not result.success
        assert result.message_key == "auth.password_too_long"
        assert sender.sent == []

    def test_password_at_exactly_bcrypt_byte_limit_is_accepted(self, tmp_path):
        from config import settings

        service, sender = _service(tmp_path)
        exact = "a" * settings.max_password_length_bytes
        result = service.register("alice@example.com", exact)
        assert result.success

    def test_two_long_passwords_sharing_a_72_byte_prefix_are_no_longer_silently_equivalent(self, tmp_path):
        """The bug the length cap exists to prevent: bcrypt truncates at
        72 bytes rather than erroring, so without this validation two
        genuinely different long passwords sharing a prefix would both
        hash identically and be accepted interchangeably."""
        from config import settings

        service, sender = _service(tmp_path)
        base = "a" * settings.max_password_length_bytes
        result = service.register("alice@example.com", base + "SUFFIX-ONE")
        assert not result.success
        assert result.message_key == "auth.password_too_long"

    def test_reset_password_with_code_also_enforces_password_policy(self, tmp_path):
        service, sender = _service(tmp_path)
        _register_and_verify(service, sender, "dave@example.com", "old-password")
        service.request_password_reset("dave@example.com")
        code = sender.last_code()

        result = service.reset_password_with_code("dave@example.com", code, "short")
        assert not result.success
        assert result.message_key == "auth.password_too_short"
        # A rejected new password must not consume the reset code.
        assert service.reset_password_with_code("dave@example.com", code, "a-fine-password").success


def test_email_is_case_and_whitespace_normalized(tmp_path):
    service, sender = _service(tmp_path)
    service.register("  Alice@Example.com  ", "correct-horse")
    code = sender.last_code()
    assert service.verify_email("alice@example.com", code).success
    assert service.login("ALICE@EXAMPLE.COM  ", "correct-horse").success


def test_wrong_password_fails_without_locking_out_immediately(tmp_path):
    service, sender = _service(tmp_path)
    _register_and_verify(service, sender, "bob@example.com", "correct-horse")

    result = service.login("bob@example.com", "wrong-password")
    assert not result.success
    # One bad attempt shouldn't lock the account.
    assert service.login("bob@example.com", "correct-horse").success


def test_lockout_after_repeated_failures(tmp_path):
    from config import settings

    service, sender = _service(tmp_path)
    _register_and_verify(service, sender, "carol@example.com", "correct-horse")

    for _ in range(settings.max_login_attempts):
        service.login("carol@example.com", "wrong-password")

    result = service.login("carol@example.com", "correct-horse")
    assert not result.success
    assert result.message_key == "auth.locked_try_again_in"


def test_request_password_reset_sends_code_and_allows_reset(tmp_path):
    service, sender = _service(tmp_path)
    _register_and_verify(service, sender, "dave@example.com", "old-password")

    result = service.request_password_reset("dave@example.com")
    assert result.success
    assert result.message_key == "auth.reset_code_sent_if_exists"
    code = sender.last_code()

    reset_result = service.reset_password_with_code("dave@example.com", code, "new-password")
    assert reset_result.success
    assert service.login("dave@example.com", "new-password").success
    assert not service.login("dave@example.com", "old-password").success


def test_request_password_reset_for_unknown_email_gives_same_generic_response(tmp_path):
    """No account-enumeration signal: the response is identical whether or
    not the email is registered, even though no email is actually sent."""
    service, sender = _service(tmp_path)
    result = service.request_password_reset("ghost@example.com")
    assert result.success
    assert result.message_key == "auth.reset_code_sent_if_exists"
    assert sender.sent == []


def test_reset_password_with_wrong_code_fails(tmp_path):
    service, sender = _service(tmp_path)
    _register_and_verify(service, sender, "dave@example.com", "old-password")
    service.request_password_reset("dave@example.com")

    result = service.reset_password_with_code("dave@example.com", "000000", "new-password")
    assert not result.success
    assert result.message_key == "auth.invalid_or_expired_code"
    assert service.login("dave@example.com", "old-password").success


def test_reset_password_code_cannot_be_reused(tmp_path):
    service, sender = _service(tmp_path)
    _register_and_verify(service, sender, "dave@example.com", "old-password")
    service.request_password_reset("dave@example.com")
    code = sender.last_code()

    assert service.reset_password_with_code("dave@example.com", code, "new-password").success
    replay = service.reset_password_with_code("dave@example.com", code, "another-password")
    assert not replay.success
    assert replay.message_key == "auth.invalid_or_expired_code"


def test_register_sends_verification_email_in_the_chosen_language(tmp_path):
    service, sender = _service(tmp_path)
    service.register("alice@example.com", "correct-horse", language="zh")

    subject = sender.sent[0][1]
    assert subject == "验证您的邮箱"  # i18n/translations/zh.json: auth.email_verify_subject


def test_register_defaults_to_english_when_no_language_given(tmp_path):
    service, sender = _service(tmp_path)
    service.register("alice@example.com", "correct-horse")

    assert sender.sent[0][1] == "Verify your email"


def test_resend_and_reset_emails_use_the_language_stored_at_registration(tmp_path):
    """The account's stored language must be used for later emails, not
    whatever language a *different* session happens to be in when
    triggering a resend or a reset — there's no current session to read
    a language from at that point, only what was saved at signup."""
    service, sender = _service(tmp_path)
    service.register("alice@example.com", "correct-horse", language="ja")

    service.resend_verification("alice@example.com")
    assert sender.sent[-1][1] == "メールアドレスの確認"

    code = sender.last_code()
    service.verify_email("alice@example.com", code)

    service.request_password_reset("alice@example.com")
    assert sender.sent[-1][1] == "パスワードリセットコード"


class TestOtpResendRateLimiting:
    """Verified live before this existed: 50 password-reset requests and
    30 resend-verification requests for the same account fired instantly
    with zero throttling, each one a real email. These tests pin the fix:
    a cooldown between two OTP emails for the same (email, purpose)."""

    def test_resend_verification_is_throttled_within_the_cooldown_window(self, tmp_path):
        service, sender = _service(tmp_path)
        service.register("alice@example.com", "correct-horse")
        assert len(sender.sent) == 1

        result = service.resend_verification("alice@example.com")
        assert not result.success
        assert result.message_key == "auth.please_wait_before_resend"
        assert len(sender.sent) == 1  # no second email actually sent

    def test_resend_verification_succeeds_again_once_cooldown_expires(self, tmp_path, monkeypatch):
        from dataclasses import replace

        import config

        service, sender = _service(tmp_path)
        service.register("alice@example.com", "correct-horse")

        monkeypatch.setattr(
            "auth.service.settings", replace(config.settings, otp_resend_cooldown_seconds=0)
        )
        result = service.resend_verification("alice@example.com")
        assert result.success
        assert len(sender.sent) == 2

    def test_repeated_password_reset_requests_return_identical_response_regardless_of_throttling(self, tmp_path):
        """The response must never reveal whether a new email actually
        went out — see request_password_reset's docstring on why (an
        enumeration side-channel otherwise)."""
        service, sender = _service(tmp_path)
        _register_and_verify(service, sender, "dave@example.com", "old-password")
        assert len(sender.sent) == 1

        results = [service.request_password_reset("dave@example.com") for _ in range(10)]
        assert all(r.success and r.message_key == "auth.reset_code_sent_if_exists" for r in results)
        # Only the first request within the cooldown window actually sent
        # an email; the other 9 were silently throttled.
        assert len(sender.sent) == 2

    def test_password_reset_still_works_after_being_throttled(self, tmp_path):
        service, sender = _service(tmp_path)
        _register_and_verify(service, sender, "dave@example.com", "old-password")

        service.request_password_reset("dave@example.com")
        service.request_password_reset("dave@example.com")  # throttled, no new code
        code = sender.last_code()  # the one and only code actually issued

        assert service.reset_password_with_code("dave@example.com", code, "new-password").success

    def test_cooldown_is_independent_per_purpose(self, tmp_path):
        """A verify_email code issued at registration must not count
        against reset_password's own cooldown for the same email — if
        cooldown were tracked per-email instead of per-(email, purpose),
        this request_password_reset call would be wrongly throttled
        immediately after register()'s own OTP send."""
        service, sender = _service(tmp_path)
        service.register("dave@example.com", "old-password")
        assert len(sender.sent) == 1  # the verify_email code from register()

        result = service.request_password_reset("dave@example.com")
        assert result.success
        assert len(sender.sent) == 2  # reset_password code — not blocked by verify_email's recent send


def test_expired_lockout_grants_fresh_attempts_not_an_immediate_relock(tmp_path):
    """Regression test: failed_attempts must actually be cleared in storage
    once a lockout window has passed — not just treated as 0 for a single
    response — otherwise record_failed_login's relative '+1' keeps climbing
    on top of the stale stored count and every subsequent miss re-locks the
    account immediately, forever, since only a successful login clears it."""
    service, user_repo, otp_repo, sender = _service_with_repos(tmp_path)
    _register_and_verify(service, sender, "erin@example.com", "correct-horse")

    # Simulate a lockout that has already expired (set directly via the
    # repository rather than sleeping in the test).
    already_expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    for _ in range(5):
        user_repo.record_failed_login("erin@example.com", already_expired)
    assert user_repo.get_user("erin@example.com").failed_attempts == 5

    # One wrong attempt after expiry should NOT immediately re-lock —
    # it should be treated as the first of a fresh set of attempts.
    result = service.login("erin@example.com", "still-wrong")
    assert not result.success
    assert result.message_key == "auth.invalid_credentials_attempts_left"
    assert user_repo.get_user("erin@example.com").failed_attempts == 1

    # And the correct password still works right after.
    assert service.login("erin@example.com", "correct-horse").success
