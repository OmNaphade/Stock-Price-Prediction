# Contributing

Thanks for taking a look at this project. It's designed so a fresh clone runs fully on free, no-key sources — you should never need anyone else's credentials to develop, test, or run it.

## Getting set up

```bash
git clone <your-fork-url>
cd Algorithmic-Stock-Price-Prediction
python -m venv .venv
.venv\Scripts\activate          # or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env            # optional — see below
streamlit run app.py
pytest tests/                   # 200+ tests, runs entirely offline, no keys required
```

That's it — no API keys, no admin account, no SMTP server. Everything in `.env.example` is optional and independently additive:

- Leave it all blank and the app runs on `yfinance` for market data, logs OTP codes to the console instead of emailing them, and skips macro/sentiment enrichment.
- Fill in only what you're working on. Touching the email flow? Set `SMTP_*` (or just read the codes from your terminal — that's what the test suite and local dev do by default). Touching the OpenAlgo integration? You'll need your own free [OpenAlgo](https://openalgo.in) instance and broker link — nobody else's works for you, by design (see below).

**Never commit a real `.env`.** It's gitignored; `.env.example` is the only version that belongs in git, and it must never contain real values — a template with blank lines only.

## What "no shared secrets" means here, concretely

This app was built from day one so that every external integration is either free/keyless (`yfinance`, the free NewsAPI/YFinance news fallback) or **account-bound to whoever configures it** (Alpha Vantage, FRED, NewsAPI, SMTP, OpenAlgo, the admin account) — nobody's fork or contribution depends on secrets only the original maintainer has. Concretely:

- **OpenAlgo** (`OPENALGO_BASE_URL`/`OPENALGO_API_KEY`) points at *your own* self-hosted instance, linked to *your own* broker account. There is no shared/public OpenAlgo endpoint this app talks to. If you don't have one, the OpenAlgo-powered features (exchange-native NSE/BSE data, market depth, market-hours status) simply don't activate — every other feature works fine without them.
- **SMTP** (`SMTP_*`) is your own mail account or transactional provider. Without it, OTP codes are logged to the console — fully functional for local development.
- **Admin account** (`ADMIN_EMAIL`/`ADMIN_PASSWORD`) is bootstrapped once from your own `.env`, gating only the Monitoring page (see the README's Authentication section). It's not required to run or contribute to anything else.
- Nothing in this codebase ever reads a secret from anywhere except `config.py`'s `Settings` (env vars / `.env` / `st.secrets`) — there's no hardcoded key, token, or password anywhere in source. If you ever see one in a PR (including in a commit message, a test fixture, or a code comment), that's a bug — flag it.

## Before opening a PR

- `pytest tests/` should pass. The suite is intentionally offline-only (mocked HTTP for every external call) so it never depends on your — or anyone's — live credentials.
- Run `git status` and `git diff --staged` before committing; double-check nothing under `.env`, `*.db`, or `.streamlit/secrets.toml` snuck into the diff (all three are gitignored, but a forced-add or a copy-paste can still slip one in).
- New optional integrations should follow the existing pattern: unconfigured means the feature no-ops (empty result, `None`, or a skipped UI section) — never an exception. Grep the codebase for `_is_configured` for examples.
- Keep the architecture's separation of concerns: `data_access/` fetches, `features/` transforms, `models/` predicts, `evaluation/` scores, `services/` orchestrates and is the only layer pages call into. See the README's "Design Principles" section for the reasoning behind this, with file-level examples.

## Reporting a security issue

If you find something that could expose another user's data or credentials (not just your own misconfiguration), please don't open a public issue — email the maintainer directly instead so it can be fixed before it's disclosed.
