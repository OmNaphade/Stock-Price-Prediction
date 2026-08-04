"""Regression test for a real bug: three pages had drifted to a Login-only
auth gate with no way to register, because each page copy-pasted its own
block instead of sharing one. This boots every page and checks Register
is actually present, not just Login."""

from __future__ import annotations

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
