import streamlit as st

st.set_page_config(page_title="Watchlist Comparison", page_icon="📈")

import matplotlib.pyplot as plt
import pandas as pd

from auth_ui import require_authenticated_user
from config import settings
from i18n import render_language_selector, t
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

if "track_record_resolved_this_session" not in st.session_state:
    st.session_state.track_record_resolved_this_session = False

render_language_selector()

# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not require_authenticated_user(auth_service):
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(t("common.welcome", username=st.session_state.username))
if st.sidebar.button(t("common.logout")):
    st.session_state.is_authenticated = False
    st.session_state.username = ""
    st.rerun()

if not st.session_state.track_record_resolved_this_session:
    try:
        track_record_service.resolve_pending()
    except Exception:
        pass
    st.session_state.track_record_resolved_this_session = True

st.markdown(f"# {t('watchlist.title')}")
st.caption(t("watchlist.caption"))

tickers_input = st.text_input(
    t("watchlist.tickers_label"),
    value="AAPL, MSFT, GOOGL, TSLA, AMZN",
    # A comma-separated list, not a single ticker — room for ~10 entries
    # at a single ticker's own max length, not settings.max_ticker_length
    # itself (that would barely fit two tickers with the ", " separator).
    max_chars=settings.max_ticker_length * 10,
)
model_name = st.selectbox(t("app.model_label"), list(AVAILABLE_MODELS.keys()), index=1)
compare_btn = st.button(t("watchlist.compare_button"))

if not compare_btn:
    st.info(t("watchlist.initial_help"))
    st.stop()

tickers = [t_.strip().upper() for t_ in tickers_input.split(",") if t_.strip()]
if not tickers:
    st.warning(t("watchlist.no_tickers_warning"))
    st.stop()

col_ticker = t("watchlist.col_ticker")
col_last_close = t("watchlist.col_last_close")
col_predicted_next_close = t("watchlist.col_predicted_next_close")
col_predicted_pct_change = t("watchlist.col_predicted_pct_change")
col_directional_accuracy = t("watchlist.col_directional_accuracy")
col_beats_baseline = t("watchlist.col_beats_baseline")

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
                col_ticker: ticker,
                col_last_close: report.last_close,
                col_predicted_next_close: report.predicted_next_close,
                col_predicted_pct_change: pct_change,
                col_directional_accuracy: report.model_backtest.mean_directional_accuracy,
                col_beats_baseline: report.beats_baseline_on_direction or report.beats_baseline_on_price_error,
            }
        )
    except PredictionError as e:
        failures.append(f"{ticker}: {e}")
    except Exception as e:
        failures.append(t("watchlist.unexpected_error", ticker=ticker, error=e))
    progress.progress((i + 1) / len(tickers))
progress.empty()

if failures:
    st.warning(t("watchlist.failures_warning", failures="\n".join(f"- {f}" for f in failures)))

if not rows:
    st.stop()

table = pd.DataFrame(rows).sort_values(col_predicted_pct_change, ascending=False)

st.subheader(t("watchlist.comparison_header"))
st.dataframe(
    table.style.format(
        {
            col_last_close: "${:,.2f}",
            col_predicted_next_close: "${:,.2f}",
            col_predicted_pct_change: "{:+.2f}%",
            col_directional_accuracy: "{:.1%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader(t("watchlist.chart_header"))
up_color, down_color = "#2E7D5B", "#B3452C"  # muted gain/loss pair, not raw RGB red/green
colors = [up_color if v >= 0 else down_color for v in table[col_predicted_pct_change]]

fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 * len(table))))
bars = ax.barh(table[col_ticker], table[col_predicted_pct_change], color=colors)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel(t("watchlist.chart_xlabel"))
ax.set_title(t("watchlist.chart_title", model_name=model_name))
for bar, value in zip(bars, table[col_predicted_pct_change]):
    offset = 0.05 * max(abs(table[col_predicted_pct_change]).max(), 1)
    align = "left" if value >= 0 else "right"
    x = value + offset if value >= 0 else value - offset
    ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%", va="center", ha=align, fontsize=9)
ax.invert_yaxis()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(t("watchlist.color_caption"))
