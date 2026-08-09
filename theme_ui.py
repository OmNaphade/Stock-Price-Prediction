"""Shared visual-polish layer every page opts into once — same pattern as
i18n.render_language_selector(): one call, near the top of the page, before
any other rendering. Built entirely on Streamlit's own CSS custom
properties (--primary-color, --background-color, --secondary-background-
color, --text-color) instead of hardcoded hex values, so it automatically
matches whichever theme (light, dark, or a custom .streamlit/config.toml)
the user has selected — including switching live from Streamlit's own
settings menu — with no separate dark-mode branch needed here."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ── Buttons ─────────────────────────────────────────────────────────── */
div.stButton > button, div.stDownloadButton > button {
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--primary-color) 40%, transparent);
    transition: transform 0.08s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: var(--primary-color);
    box-shadow: 0 2px 8px color-mix(in srgb, var(--primary-color) 25%, transparent);
    transform: translateY(-1px);
}
div.stButton > button:active, div.stDownloadButton > button:active {
    transform: translateY(0);
}

/* ── Metric cards ────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
    border-radius: 10px;
    padding: 0.9rem 1rem 0.7rem;
}

/* ── Headings ────────────────────────────────────────────────────────── */
h1 {
    border-bottom: 2px solid color-mix(in srgb, var(--primary-color) 55%, transparent);
    padding-bottom: 0.35rem;
}
h2, h3 {
    border-bottom: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
    padding-bottom: 0.3rem;
}

/* ── Alerts (info / warning / error / success) ──────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Dataframes / tables ─────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}
table.dataframe {
    border-collapse: collapse;
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
}
table.dataframe th {
    background: var(--secondary-background-color);
    text-align: right;
    padding: 0.4rem 0.6rem;
}
table.dataframe td {
    padding: 0.35rem 0.6rem;
    border-top: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
}
table.dataframe tr:hover td {
    background: color-mix(in srgb, var(--primary-color) 8%, transparent);
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    border-right: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div, .stTextInput > div > div {
    border-radius: 8px !important;
}
</style>
"""


def apply_theme() -> None:
    """Injects the shared CSS once. Idempotent and safe to call on every
    page — st.markdown with the same content just re-renders the same
    <style> tag on rerun, same as any other Streamlit element."""
    st.markdown(_CSS, unsafe_allow_html=True)
