"""The login/register gate every page renders identically.

This used to be copy-pasted into every page individually — which is
exactly how three of the five pages ended up with a Login button but no
way to Register at all: each page's copy could drift independently, and
three of them did. One shared function means there's only one place this
can be gotten wrong, and only one place to fix it.
"""

from __future__ import annotations

import streamlit as st

from auth import AuthService


def require_authenticated_user(
    auth_service: AuthService, title: str = "Login to Stock Prediction App"
) -> bool:
    """Renders the login/register/reset-password gate if the current
    session isn't authenticated yet. Returns True if the caller should go
    on to render the rest of the page; False means the gate was shown —
    the caller should `st.stop()`."""
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
        st.session_state.username = ""

    if st.session_state.is_authenticated:
        return True

    st.title(title)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", use_container_width=True):
            result = auth_service.login(username, password)
            if result.success:
                st.session_state.is_authenticated = True
                st.session_state.username = username
                st.success(result.message)
                st.rerun()
            else:
                st.error(result.message)
    with col2:
        if st.button("Register", use_container_width=True):
            result = auth_service.register(username, password)
            if result.success:
                st.success(result.message)
            else:
                st.error(result.message)

    with st.expander("Forgot Password?"):
        reset_username = st.text_input("Username:", key="reset_user")
        reset_new_password = st.text_input("New Password:", type="password", key="reset_pw")
        if st.button("Reset Password"):
            result = auth_service.reset_password(reset_username, reset_new_password)
            if result.success:
                st.success(result.message)
            else:
                st.error(result.message)

    return False
