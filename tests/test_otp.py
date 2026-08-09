from __future__ import annotations

from auth.otp import generate_code, hash_code, verify_code


def test_generate_code_has_requested_length_and_is_numeric():
    code = generate_code(6)
    assert len(code) == 6
    assert code.isdigit()


def test_generate_code_varies_across_calls():
    codes = {generate_code(6) for _ in range(20)}
    assert len(codes) > 1  # astronomically unlikely to collide 20x by chance


def test_verify_code_accepts_the_matching_code():
    code = generate_code(6)
    assert verify_code(code, hash_code(code)) is True


def test_verify_code_rejects_a_different_code():
    code = generate_code(6)
    other = "0" * 6 if code != "0" * 6 else "1" * 6
    assert verify_code(other, hash_code(code)) is False


def test_hash_code_does_not_store_the_plaintext_code():
    code = "123456"
    assert code not in hash_code(code)
