from __future__ import annotations

from unittest.mock import MagicMock, patch

from auth.email_sender import NullEmailSender, SmtpEmailSender, build_default_email_sender


class TestNullEmailSender:
    def test_send_returns_true_and_never_raises(self, caplog):
        sender = NullEmailSender()
        assert sender.send("alice@example.com", "Subject", "Body with code 123456") is True

    def test_send_logs_the_message_server_side(self, caplog):
        import logging

        # config.py's logger has propagate=False (to avoid duplicate
        # console output), which also keeps it off pytest's root-logger
        # capture handler — attach caplog's handler directly instead.
        logger = logging.getLogger("stock_prediction")
        logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="stock_prediction")

        sender = NullEmailSender()
        sender.send("alice@example.com", "Verify your email", "Your code is 654321")
        assert "alice@example.com" in caplog.text
        assert "654321" in caplog.text


class TestSmtpEmailSender:
    def test_send_uses_starttls_and_login_then_sendmail(self):
        sender = SmtpEmailSender(
            host="smtp.example.com", port=587, username="bot@example.com",
            password="secret", from_address="bot@example.com", use_tls=True,
        )
        mock_server = MagicMock()
        with patch("auth.email_sender.smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            result = sender.send("alice@example.com", "Subject", "Body")

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("bot@example.com", "secret")
        mock_server.sendmail.assert_called_once()
        args = mock_server.sendmail.call_args[0]
        assert args[0] == "bot@example.com"
        assert args[1] == ["alice@example.com"]
        assert "Body" in args[2]

    def test_send_skips_login_when_no_username(self):
        sender = SmtpEmailSender(
            host="localhost", port=25, username="", password="", from_address="app@localhost", use_tls=False,
        )
        mock_server = MagicMock()
        with patch("auth.email_sender.smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            sender.send("alice@example.com", "Subject", "Body")

        mock_server.starttls.assert_not_called()
        mock_server.login.assert_not_called()

    def test_send_returns_false_and_does_not_raise_on_smtp_failure(self):
        sender = SmtpEmailSender(
            host="smtp.example.com", port=587, username="bot@example.com",
            password="secret", from_address="bot@example.com",
        )
        with patch("auth.email_sender.smtplib.SMTP", side_effect=Exception("connection refused")):
            result = sender.send("alice@example.com", "Subject", "Body")
        assert result is False


class TestBuildDefaultEmailSender:
    def test_returns_null_sender_when_smtp_unconfigured(self):
        with patch("auth.email_sender.settings") as mock_settings:
            mock_settings.smtp_host = ""
            mock_settings.smtp_username = ""
            mock_settings.smtp_password = ""
            sender = build_default_email_sender()
        assert isinstance(sender, NullEmailSender)

    def test_returns_smtp_sender_when_fully_configured(self):
        with patch("auth.email_sender.settings") as mock_settings:
            mock_settings.smtp_host = "smtp.example.com"
            mock_settings.smtp_port = 587
            mock_settings.smtp_username = "bot@example.com"
            mock_settings.smtp_password = "secret"
            mock_settings.smtp_from_address = "bot@example.com"
            mock_settings.smtp_use_tls = True
            sender = build_default_email_sender()
        assert isinstance(sender, SmtpEmailSender)
