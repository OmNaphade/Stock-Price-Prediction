from __future__ import annotations

from auth.password_hashing import check_password, hash_password


def test_hash_password_does_not_store_the_plaintext():
    hashed = hash_password("correct-horse")
    assert "correct-horse" not in hashed


def test_check_password_accepts_the_matching_password():
    hashed = hash_password("correct-horse")
    assert check_password("correct-horse", hashed) is True


def test_check_password_rejects_a_different_password():
    hashed = hash_password("correct-horse")
    assert check_password("wrong-password", hashed) is False


def test_hashing_the_same_password_twice_produces_different_hashes():
    # bcrypt salts each hash independently — two hashes of the same
    # password must not be byte-identical.
    assert hash_password("correct-horse") != hash_password("correct-horse")
