"""OTP (one-time passcode) generation and verification — pure functions,
no storage or transport concerns (see otp_repository.py and
email_sender.py for those).

Deliberately a short random code, not a TOTP/HOTP construct: this is a
single-use, server-generated code tied to one request (registration or a
password reset), not something a user's authenticator app needs to
reproduce independently."""

from __future__ import annotations

import hashlib
import secrets


def generate_code(length: int) -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(length))


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def verify_code(code: str, code_hash: str) -> bool:
    return secrets.compare_digest(hash_code(code), code_hash)
