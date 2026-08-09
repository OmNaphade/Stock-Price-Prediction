"""config.py loads .env at import time (module-level `load_dotenv()`), so
by the time `settings` is built, values from a .env file are already in
os.environ exactly as if they'd been exported in the shell. This test
exercises that mechanism directly with python-dotenv's own API rather
than reloading the config module (which is a singleton imported
everywhere else in the test suite) — the point under test is "does a
.env file's contents actually reach os.environ," which is precisely
what config.py relies on."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def test_load_dotenv_populates_os_environ_from_a_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_PROBE_VAR=hello-from-dotenv\n")
    monkeypatch.delenv("TEST_DOTENV_PROBE_VAR", raising=False)

    load_dotenv(dotenv_path=env_file)

    assert os.environ["TEST_DOTENV_PROBE_VAR"] == "hello-from-dotenv"


def test_real_environment_variables_take_precedence_over_dotenv_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_PROBE_VAR=from-file\n")
    monkeypatch.setenv("TEST_DOTENV_PROBE_VAR", "from-real-env")

    # Default override=False: an already-set real env var must win.
    load_dotenv(dotenv_path=env_file)

    assert os.environ["TEST_DOTENV_PROBE_VAR"] == "from-real-env"


def test_missing_dotenv_file_is_a_silent_no_op(tmp_path):
    missing_path = tmp_path / "does-not-exist" / ".env"
    result = load_dotenv(dotenv_path=missing_path)
    assert result is False  # python-dotenv's own "nothing loaded" signal, not an exception
