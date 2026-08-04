"""Boots each Streamlit page headlessly via Streamlit's own AppTest harness
and checks it reaches the login gate without raising. This is what caught
a real bug during development: `pages/prediction.py` called
`st.set_page_config()` after other Streamlit calls, which Streamlit
requires to be first — a plain syntax check or unit test wouldn't have
caught it, only actually running the script does."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.parametrize("page", ["app.py", "pages/prediction.py", "pages/watchlist.py"])
def test_page_boots_to_login_gate_without_exceptions(page):
    at = AppTest.from_file(page)
    at.run(timeout=30)
    assert not at.exception
