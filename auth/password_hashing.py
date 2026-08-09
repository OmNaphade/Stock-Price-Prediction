"""bcrypt password hashing, split out of AuthService — anything else that
needs to hash a password (auth/bootstrap.py seeding the admin account
from ADMIN_EMAIL/ADMIN_PASSWORD) shouldn't have to reach into
AuthService's internals to get it. One narrow module, one job."""

from __future__ import annotations

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())
