"""Email delivery for OTP codes — a narrow Protocol so AuthService never
touches smtplib directly (Dependency Inversion, same pattern as
MarketDataSource/NewsSource elsewhere in this app).

SmtpEmailSender is the real implementation. NullEmailSender logs the
message server-side instead of sending it — used whenever SMTP isn't
configured, so registration/reset flows still work end-to-end in local
dev and in the test suite without a real mailbox, the same
degrade-gracefully contract every other optional integration in this app
follows (AlphaVantageSource, OpenAlgoSource, FredMacroSource, ...)."""

from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from typing import Protocol

from config import log, settings


class EmailSender(Protocol):
    def send(self, to_address: str, subject: str, body: str) -> bool:
        """True if the message was handed off successfully."""
        ...


class SmtpEmailSender:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        from_address: str,
        use_tls: bool = True,
        timeout_seconds: int = 15,
    ):
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from_address = from_address
        self._use_tls = use_tls
        self._timeout = timeout_seconds

    def send(self, to_address: str, subject: str, body: str) -> bool:
        message = MIMEText(body)
        message["Subject"] = subject
        message["From"] = self._from_address
        message["To"] = to_address
        try:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as server:
                if self._use_tls:
                    server.starttls()
                if self._username:
                    server.login(self._username, self._password)
                server.sendmail(self._from_address, [to_address], message.as_string())
            return True
        except Exception:
            log.warning("Failed to send email to %s", to_address, exc_info=True)
            return False


class NullEmailSender:
    def send(self, to_address: str, subject: str, body: str) -> bool:
        log.info("SMTP not configured — logging email instead.\nTo: %s\nSubject: %s\n%s", to_address, subject, body)
        return True


def build_default_email_sender() -> EmailSender:
    if settings.smtp_host and settings.smtp_username and settings.smtp_password:
        return SmtpEmailSender(
            settings.smtp_host,
            settings.smtp_port,
            settings.smtp_username,
            settings.smtp_password,
            settings.smtp_from_address,
            settings.smtp_use_tls,
        )
    return NullEmailSender()
