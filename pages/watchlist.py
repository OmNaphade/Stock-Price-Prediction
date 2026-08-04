import streamlit as st

st.set_page_config(page_title="Watchlist Comparison", page_icon="📈")

import matplotlib.pyplot as plt
import pandas as pd

from config import settings
from services import AVAILABLE_MODELS, PredictionError
from web_context import (
    get_auth_service,
    get_monitoring_service,
    get_prediction_service,
    get_track_record_service,
)

auth_service = get_auth_service()
prediction_service = get_prediction_service()
track_record_service = get_track_record_service()
monitoring_service = get_monitoring_service()

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
    st.session_state.username = ""
if "track_record_resolved_this_session" not in st.session_state:
    st.session_state.track_record_resolved_this_session = False

# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not st.session_state.is_authenticated:
    st.title("Login to Stock Prediction App")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    if st.button("Login", use_container_width=True):
        result = auth_service.login(username, password)
        if result.success:
            st.session_state.is_authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error(result.message)
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(f"Welcome, {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.is_authenticated = False
    st.session_state.username = ""
    st.rerun()

if not st.session_state.track_record_resolved_this_session:
    try:
        track_record_service.resolve_pending()
    except Exception:
        pass
    st.session_state.track_record_resolved_this_session = True

st.markdown("# 📈 Watchlist Comparison")
st.caption(
    "Runs the same walk-forward-backtested prediction across several tickers at "
    "once, so you can see which ones actually show a model edge over the naive "
    "baseline instead of checking one at a time."
)

tickers_input = st.text_input(
    "Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, TSLA, AMZN"
)
model_name = st.selectbox("Model", list(AVAILABLE_MODELS.keys()), index=1)
compare_btn = st.button("🔍 Compare")

if not compare_btn:
    st.info("Enter a few tickers and click **🔍 Compare**.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

rows = []
failures = []
progress = st.progress(0.0)
for i, ticker in enumerate(tickers):
    try:
        report = prediction_service.analyze(
            ticker, start=settings.history_start, model_name=model_name
        )
        track_record_service.record_prediction(st.session_state.username, report)
        monitoring_service.log_from_report(st.session_state.username, report)
        pct_change = (report.predicted_next_close / report.last_close - 1) * 100
        rows.append(
            {
                "Ticker": ticker,
                "Last Close": report.last_close,
                "Predicted Next Close": report.predicted_next_close,
                "Predicted % Change": pct_change,
                "Directional Accuracy": report.model_backtest.mean_directional_accuracy,
                "Beats Baseline": report.beats_baseline_on_direction or report.beats_baseline_on_price_error,
            }
        )
    except PredictionError as e:
        failures.append(f"{ticker}: {e}")
    except Exception as e:
        failures.append(f"{ticker}: unexpected error — {e}")
    progress.progress((i + 1) / len(tickers))
progress.empty()

if failures:
    st.warning("⚠️ Couldn't analyze:\n\n" + "\n".join(f"- {f}" for f in failures))

if not rows:
    st.stop()

table = pd.DataFrame(rows).sort_values("Predicted % Change", ascending=False)

st.subheader("📋 Comparison")
st.dataframe(
    table.style.format(
        {
            "Last Close": "${:,.2f}",
            "Predicted Next Close": "${:,.2f}",
            "Predicted % Change": "{:+.2f}%",
            "Directional Accuracy": "{:.1%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("📊 Predicted % Change by Ticker")
up_color, down_color = "#2E7D5B", "#B3452C"  # muted gain/loss pair, not raw RGB red/green
colors = [up_color if v >= 0 else down_color for v in table["Predicted % Change"]]

fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 * len(table))))
bars = ax.barh(table["Ticker"], table["Predicted % Change"], color=colors)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Predicted next-close change (%)")
ax.set_title(f"{model_name} — Predicted Next-Close Change")
for bar, value in zip(bars, table["Predicted % Change"]):
    offset = 0.05 * max(abs(table["Predicted % Change"]).max(), 1)
    align = "left" if value >= 0 else "right"
    x = value + offset if value >= 0 else value - offset
    ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%", va="center", ha=align, fontsize=9)
ax.invert_yaxis()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Color marks direction (green = predicted gain, red = predicted loss) and every "
    "bar is also labeled with its value — color is never the only signal."
)
