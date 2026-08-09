from __future__ import annotations

import json
from pathlib import Path

import pytest

from i18n.translator import LANGUAGES, _catalog, t

_TRANSLATIONS_DIR = Path(__file__).resolve().parent.parent / "i18n" / "translations"


def test_every_declared_language_has_a_catalog_file():
    for code in LANGUAGES:
        assert (_TRANSLATIONS_DIR / f"{code}.json").exists(), f"missing catalog for {code}"


@pytest.mark.parametrize("code", list(LANGUAGES.keys()))
def test_catalog_is_valid_json(code):
    path = _TRANSLATIONS_DIR / f"{code}.json"
    json.loads(path.read_text(encoding="utf-8"))  # raises if malformed


def test_all_catalogs_have_the_same_keys_as_english():
    _catalog.cache_clear()
    english_keys = set(_catalog("en").keys())
    assert english_keys, "English catalog should not be empty"
    for code in LANGUAGES:
        _catalog.cache_clear()
        keys = set(_catalog(code).keys())
        assert keys == english_keys, f"{code}.json keys differ from en.json"


def test_missing_key_falls_back_to_the_key_itself(monkeypatch):
    monkeypatch.setattr("i18n.translator.current_language", lambda: "en")
    assert t("this.key.does.not.exist") == "this.key.does.not.exist"


def test_missing_translation_falls_back_to_english(monkeypatch):
    # zh.json has every key english has (parity test above), so simulate a
    # language with a gap by pointing at one that genuinely lacks the key.
    monkeypatch.setattr("i18n.translator.current_language", lambda: "xx")
    assert t("auth.login_button") == "Login"


def test_interpolation_substitutes_kwargs(monkeypatch):
    monkeypatch.setattr("i18n.translator.current_language", lambda: "en")
    assert t("common.welcome", username="alice") == "👋 Welcome, alice"


@pytest.mark.parametrize("code", list(LANGUAGES.keys()))
def test_login_button_is_translated_in_every_language(monkeypatch, code):
    monkeypatch.setattr("i18n.translator.current_language", lambda: code)
    label = t("auth.login_button")
    assert label
    assert label != "auth.login_button"  # never falls through to a raw key
