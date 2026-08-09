"""Translation lookup — the only piece of i18n that knows how catalogs are
stored (JSON files in translations/) or how the active language is tracked
(Streamlit session state). Everything else in this app calls `t(key, ...)`
and never touches either concern directly."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_TRANSLATIONS_DIR = Path(__file__).parent / "translations"
DEFAULT_LANGUAGE = "en"

LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ko": "한국어",
    "ja": "日本語",
    "tr": "Türkçe",
    "ru": "Русский",
}


@lru_cache(maxsize=None)
def _catalog(lang: str) -> dict[str, str]:
    path = _TRANSLATIONS_DIR / f"{lang}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def current_language() -> str:
    """The active language for this session, or English outside of a
    Streamlit run (e.g. a plain `python` import, or a test that doesn't
    set up session state) — same fallback shape as config.py's
    _get_secret when st.secrets isn't available."""
    try:
        import streamlit as st

        return st.session_state.get("lang", DEFAULT_LANGUAGE)
    except Exception:
        return DEFAULT_LANGUAGE


def t(key: str, lang: str | None = None, **kwargs: Any) -> str:
    """Translate `key`, falling back to English and then to the key itself
    — a missing translation degrades to readable-if-wrong text instead of
    a crash or a raw dotted key on screen. `kwargs` are interpolated with
    `str.format`, same placeholder syntax as an f-string (e.g.
    `t("app.welcome", username=name)` for a catalog entry of
    `"Welcome, {username}"`).

    `lang` defaults to the current Streamlit session's language. Pass it
    explicitly for text rendered outside of a session — e.g. AuthService
    sending an email to an address that isn't necessarily the currently
    browsing session (a resend, or a reset requested from a different
    device): the caller supplies a plain language-code string it looked
    up itself, so this module still never needs anything to import
    Streamlit or touch session state on its behalf."""
    lang = lang or current_language()
    text = _catalog(lang).get(key) or _catalog(DEFAULT_LANGUAGE).get(key) or key
    return text.format(**kwargs) if kwargs else text
