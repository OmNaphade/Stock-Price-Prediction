from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run_apply_theme():
    from theme_ui import apply_theme

    apply_theme()


def test_apply_theme_injects_a_style_block_without_raising():
    at = AppTest.from_function(_run_apply_theme)
    at.run(timeout=30)
    assert not at.exception
    assert any("<style>" in md.value for md in at.markdown)


def test_theme_css_uses_streamlit_theme_variables_not_hardcoded_colors():
    """Regression guard for the reason this module exists: colors must come
    from Streamlit's own CSS custom properties so light/dark/custom themes
    are respected automatically, never a hardcoded hex value."""
    import re

    from theme_ui import _CSS

    hex_colors = re.findall(r"#[0-9a-fA-F]{3,8}\b", _CSS)
    assert hex_colors == []
    assert "var(--primary-color)" in _CSS
    assert "var(--background-color)" in _CSS or "var(--secondary-background-color)" in _CSS
