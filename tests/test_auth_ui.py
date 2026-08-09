"""Regression test for a real bug: three pages had drifted to a Login-only
auth gate with no way to register, because each page copy-pasted its own
block instead of sharing one. This boots every page and checks Register
is actually present, not just Login."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.parametrize(
    "page",
    [
        "app.py",
        "pages/prediction.py",
        "pages/watchlist.py",
        "pages/track_record.py",
        "pages/monitoring.py",
    ],
)
def test_every_page_offers_registration_not_just_login(page):
    at = AppTest.from_file(page)
    at.run(timeout=30)
    assert not at.exception

    button_labels = [b.label for b in at.button]
    assert "Login" in button_labels
    assert "Register" in button_labels


def test_registering_transitions_to_the_verify_email_step():
    """End-to-end check that submitting Register on the real login form
    actually moves the gate into the verify-email step (not just that
    AuthService.register() returns success in isolation — see
    tests/test_auth.py for that unit-level coverage).

    Uses a fresh, random email each run: this drives the real
    web_context.get_auth_service(), backed by the app's actual (non-temp)
    users.db, so a fixed address would collide with a previous run's
    leftover registration."""
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception

    email = f"newuser-{uuid.uuid4().hex[:12]}@example.com"
    at.text_input[0].input(email)
    at.text_input[1].input("correct-horse-battery-staple")
    at.button[1].click().run(timeout=30)  # [0]=Login, [1]=Register

    assert not at.exception
    assert at.session_state["auth_flow"] == "verify_email"
    assert "Verify" in [b.label for b in at.button]


def test_language_switch_actually_changes_rendered_button_text():
    """End-to-end check that the sidebar language selector isn't just
    plumbing that never gets exercised: switching to Chinese should change
    the real rendered Login button label, not just what t() returns in
    isolation (see tests/test_i18n.py for that unit-level coverage)."""
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    assert "Login" in [b.label for b in at.button]

    at.session_state["lang"] = "zh"
    at.run(timeout=30)
    assert not at.exception
    button_labels = [b.label for b in at.button]
    assert "登录" in button_labels
    assert "Login" not in button_labels
