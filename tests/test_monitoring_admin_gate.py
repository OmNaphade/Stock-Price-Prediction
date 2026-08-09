"""End-to-end check that pages/monitoring.py's admin gate actually works
when the real page runs — not just that require_admin_user's logic is
correct in isolation (that part doesn't need Streamlit at all)."""

from __future__ import annotations

from dataclasses import replace

import config
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_monitoring_blocks_unconfigured_admin(monkeypatch):
    monkeypatch.setattr("auth_ui.settings", replace(config.settings, admin_email="", admin_password=""))

    at = AppTest.from_file("pages/monitoring.py")
    at.run(timeout=30)
    at.session_state["is_authenticated"] = True
    at.session_state["username"] = "anyone@example.com"
    at.run(timeout=30)

    assert not at.exception
    assert any("administrator" in e.value or "configured" in e.value for e in at.error)


def test_monitoring_blocks_a_logged_in_non_admin(monkeypatch):
    monkeypatch.setattr(
        "auth_ui.settings", replace(config.settings, admin_email="admin@example.com", admin_password="whatever123")
    )

    at = AppTest.from_file("pages/monitoring.py")
    at.run(timeout=30)
    at.session_state["is_authenticated"] = True
    at.session_state["username"] = "regular-user@example.com"
    at.run(timeout=30)

    assert not at.exception
    assert any("administrator" in e.value for e in at.error)
    # Never reaches the dashboard content.
    assert not any("Model Monitoring" in h.value for h in at.markdown)


def test_monitoring_allows_the_configured_admin_through(monkeypatch):
    monkeypatch.setattr(
        "auth_ui.settings", replace(config.settings, admin_email="admin@example.com", admin_password="whatever123")
    )

    at = AppTest.from_file("pages/monitoring.py")
    at.run(timeout=30)
    at.session_state["is_authenticated"] = True
    at.session_state["username"] = "admin@example.com"
    at.run(timeout=30)

    assert not at.exception
    # Passed the gate — no "admin only"/"not configured" error, whatever
    # renders next (dashboard, or "no backtests logged yet") is fine.
    assert not any("administrator" in e.value for e in at.error)
