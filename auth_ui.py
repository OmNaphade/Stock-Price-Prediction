"""The login/register/verify/reset gate every page renders identically.

This used to be copy-pasted into every page individually — which is
exactly how three of the five pages ended up with a Login button but no
way to Register at all: each page's copy could drift independently, and
three of them did. One shared function means there's only one place this
can be gotten wrong, and only one place to fix it.

Identity is an email address, and registration/reset both route through
an emailed OTP code (see auth/service.py for why) — so this gate is a
small state machine, not a single form: `auth_flow` in session_state is
one of "login" (default), "verify_email", or "forgot_reset", and
`auth_pending_email` carries the email that state is acting on.
"""

from __future__ import annotations

import streamlit as st

from auth import AuthService
from config import settings
from i18n import current_language, t


def require_authenticated_user(auth_service: AuthService, title: str | None = None) -> bool:
    """Renders the auth gate if the current session isn't authenticated
    yet. Returns True if the caller should go on to render the rest of
    the page; False means the gate was shown — the caller should
    `st.stop()`."""
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
        st.session_state.username = ""
    if "auth_flow" not in st.session_state:
        st.session_state.auth_flow = "login"
        st.session_state.auth_pending_email = ""

    if st.session_state.is_authenticated:
        return True

    st.title(title or t("auth.default_title"))

    flow = st.session_state.auth_flow
    if flow == "verify_email":
        _render_verify_view(auth_service)
    elif flow == "forgot_reset":
        _render_forgot_reset_view(auth_service)
    else:
        _render_login_view(auth_service)

    return False


def require_admin_user(auth_service: AuthService, title: str | None = None) -> bool:
    """Same gate as require_authenticated_user, plus a second, independent
    check: the logged-in account must be the configured admin
    (settings.admin_email — see auth/bootstrap.py). Used by
    pages/monitoring.py. Kept as two separate checks rather than one
    combined one — "are you logged in" vs. "are you allowed to see this"
    — so a page that only needs the first doesn't have to reason about
    the second, same Interface Segregation reasoning the rest of this app
    already follows for its other Protocols."""
    if not require_authenticated_user(auth_service, title):
        return False
    if not settings.admin_email:
        st.error(t("auth.admin_not_configured"))
        return False
    if st.session_state.username != settings.admin_email.strip().lower():
        st.error(t("auth.admin_only"))
        return False
    return True


def _goto(flow: str, email: str = "") -> None:
    st.session_state.auth_flow = flow
    st.session_state.auth_pending_email = email
    st.rerun()


def _render_login_view(auth_service: AuthService) -> None:
    email = st.text_input(t("auth.email_label"), max_chars=settings.max_email_length)
    password = st.text_input(
        t("auth.password_label"), type="password", max_chars=settings.max_password_length_bytes
    )
    col1, col2 = st.columns(2)

    with col1:
        if st.button(t("auth.login_button"), use_container_width=True):
            result = auth_service.login(email, password)
            if result.success:
                st.session_state.is_authenticated = True
                st.session_state.username = email.strip().lower()
                st.success(t(result.message_key, **result.message_params))
                st.rerun()
            elif result.message_key == "auth.email_not_verified":
                st.warning(t(result.message_key, **result.message_params))
                _goto("verify_email", email.strip().lower())
            else:
                st.error(t(result.message_key, **result.message_params))
    with col2:
        if st.button(t("auth.register_button"), use_container_width=True):
            result = auth_service.register(email, password, language=current_language())
            if result.success:
                st.success(t(result.message_key, email=email.strip().lower()))
                _goto("verify_email", email.strip().lower())
            else:
                st.error(t(result.message_key, **result.message_params))

    with st.expander(t("auth.forgot_password")):
        reset_email = st.text_input(
            t("auth.email_label"), key="forgot_email", max_chars=settings.max_email_length
        )
        if st.button(t("auth.send_reset_code_button")):
            result = auth_service.request_password_reset(reset_email)
            st.success(t(result.message_key, **result.message_params))
            _goto("forgot_reset", reset_email.strip().lower())


def _render_verify_view(auth_service: AuthService) -> None:
    email = st.session_state.auth_pending_email
    st.subheader(t("auth.verify_title"))
    st.caption(t("auth.verify_instructions", email=email))

    code = st.text_input(t("auth.code_label"), key="verify_code", max_chars=settings.otp_code_length)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("auth.verify_button"), use_container_width=True):
            result = auth_service.verify_email(email, code)
            if result.success:
                st.success(t(result.message_key, **result.message_params))
                _goto("login")
            else:
                st.error(t(result.message_key, **result.message_params))
    with col2:
        if st.button(t("auth.resend_code_button"), use_container_width=True):
            result = auth_service.resend_verification(email)
            if result.success:
                st.success(t(result.message_key, email=email))
            else:
                st.error(t(result.message_key, **result.message_params))

    if st.button(t("auth.back_to_login")):
        _goto("login")


def _render_forgot_reset_view(auth_service: AuthService) -> None:
    email = st.session_state.auth_pending_email
    st.subheader(t("auth.reset_title"))
    st.caption(t("auth.reset_instructions", email=email))

    code = st.text_input(t("auth.code_label"), key="reset_code", max_chars=settings.otp_code_length)
    new_password = st.text_input(
        t("auth.reset_new_password_label"),
        type="password",
        key="reset_pw",
        max_chars=settings.max_password_length_bytes,
    )
    if st.button(t("auth.reset_button")):
        result = auth_service.reset_password_with_code(email, code, new_password)
        if result.success:
            st.success(t(result.message_key, **result.message_params))
            _goto("login")
        else:
            st.error(t(result.message_key, **result.message_params))

    if st.button(t("auth.back_to_login")):
        _goto("login")
