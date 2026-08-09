"""The language-selector sidebar widget every page renders identically —
same reasoning as auth_ui.require_authenticated_user being one shared
function instead of copy-pasted per page: one place for this to drift
would eventually mean pages disagree on how language switching works."""

from __future__ import annotations

import streamlit as st

from .translator import DEFAULT_LANGUAGE, LANGUAGES


def render_language_selector() -> None:
    """Renders the sidebar language picker and updates st.session_state
    ["lang"] on change. Call this before any t() calls on the page —
    including before the auth gate, so the login screen itself is shown
    in the user's chosen language, not just the app after logging in."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = DEFAULT_LANGUAGE

    codes = list(LANGUAGES.keys())
    selected = st.sidebar.selectbox(
        "🌐 Language",
        codes,
        index=codes.index(st.session_state["lang"]),
        format_func=lambda code: LANGUAGES[code],
        key="lang_selector",
    )
    if selected != st.session_state["lang"]:
        st.session_state["lang"] = selected
        st.rerun()
