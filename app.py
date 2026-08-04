import streamlit as st

st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
)

import matplotlib.pyplot as plt

from auth_ui import require_authenticated_user
from config import settings
from services import AVAILABLE_MODELS, PredictionError
from web_context import (
    get_auth_service,
    get_monitoring_service,
    get_prediction_service,
    get_sentiment_service,
    get_track_record_service,
)

auth_service = get_auth_service()
prediction_service = get_prediction_service()
sentiment_service = get_sentiment_service()
track_record_service = get_track_record_service()
monitoring_service = get_monitoring_service()

_defaults = {
    "is_authenticated": False,
    "username": "",
    "report": None,
    "sentiment": None,
    "last_ticker": "",
    "load_attempted": False,
    "track_record_resolved_this_session": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not require_authenticated_user(auth_service, title="📈 Stock Prediction App"):
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(f"👋 Welcome, {st.session_state.username}")
if st.sidebar.button("Logout"):
    for k, v in _defaults.items():
        st.session_state[k] = v
    st.rerun()

# Checking for newly-resolvable predictions means live network calls (to
# fetch what actually happened), so this runs once per session, not on
# every rerun — same reasoning as the sentiment fetch below.
if not st.session_state.track_record_resolved_this_session:
    try:
        track_record_service.resolve_pending()
    except Exception:
        pass  # best-effort — a stale track record isn't worth failing the page over
    st.session_state.track_record_resolved_this_session = True

if not settings.av_api_key:
    st.sidebar.warning(
        "⚠️ No Alpha Vantage key set — using yfinance only.\n\n"
        "Add `AV_API_KEY` as a Streamlit secret or env var for more reliable "
        "data when running in the cloud.\n\nGet a free key at alphavantage.co"
    )

st.title('📊 Stock Price Prediction')

col_inp, col_model, col_btn = st.columns([3, 2, 1])
with col_inp:
    user_input = st.text_input(
        'Enter Stock Ticker',
        value=st.session_state.last_ticker or 'AAPL',
        placeholder='e.g. AAPL, TSLA, RELIANCE.NS',
    ).strip().upper()
with col_model:
    model_name = st.selectbox("Model", list(AVAILABLE_MODELS.keys()), index=1)
with col_btn:
    st.write("")
    load_btn = st.button("🔍 Analyze")

if not st.session_state.load_attempted and not load_btn:
    st.info(
        "👆 Enter a ticker, pick a model, and click **🔍 Analyze**.\n\n"
        "**Examples:** `AAPL` · `TSLA` · `MSFT` · `RELIANCE.NS` · `TCS.NS`\n\n"
        "Every model is benchmarked against a naive 'tomorrow = today' "
        "baseline — if it can't beat that, the app will tell you."
    )
    st.stop()

if load_btn:
    st.session_state.report = None
    st.session_state.sentiment = None
    st.session_state.last_ticker = user_input
    st.session_state.load_attempted = True

    with st.spinner(f"Fetching data and backtesting {model_name} for {user_input}…"):
        try:
            st.session_state.report = prediction_service.analyze(
                user_input, start=settings.history_start, model_name=model_name
            )
            # Recorded before anyone knows the outcome — that's what makes
            # the track record honest rather than something that could be
            # cherry-picked after the fact. Both calls carry the current
            # username since track record and monitoring are per-user.
            track_record_service.record_prediction(st.session_state.username, st.session_state.report)
            monitoring_service.log_from_report(st.session_state.username, st.session_state.report)
            # Fetched once here, alongside the report, and cached — not on
            # every rerun. Sentiment is a live network call (headlines),
            # and this page reruns on any widget interaction, not just a
            # fresh Analyze click (e.g. pressing Enter in the ticker box).
            st.session_state.sentiment = sentiment_service.snapshot(user_input)
        except PredictionError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

report = st.session_state.report
if report is None:
    st.stop()

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
st.subheader('📋 Past Data Summary')
st.markdown(
    report.ohlcv.describe().round(2).to_html(classes="dataframe", border=0),
    unsafe_allow_html=True,
)

st.subheader('📈 Stock Trends')
ma100 = report.ohlcv['Close'].rolling(100).mean()
ma200 = report.ohlcv['Close'].rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(report.ohlcv.index, report.ohlcv['Close'], label='Close Price', alpha=0.45, linewidth=1)
ax.plot(report.ohlcv.index, ma100, label='MA 100', linewidth=2)
ax.plot(report.ohlcv.index, ma200, label='MA 200', linewidth=2)
ax.set_title(f"{report.ticker} — Close Price & Moving Averages")
ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.subheader('🧪 Walk-Forward Backtest — Model vs. Naive Baseline')
st.caption(
    f"{model_name} vs. a 'tomorrow's close = today's close' baseline, averaged "
    f"across {len(report.model_backtest.folds)} expanding-window, chronological folds."
)
b1, b2 = st.columns(2)
b1.metric(
    "Directional accuracy",
    f"{report.model_backtest.mean_directional_accuracy:.1%}",
    delta=f"{report.model_backtest.mean_directional_accuracy - 0.5:+.1%} vs. coin flip",
)
b2.metric(
    "Price RMSE",
    f"${report.model_backtest.mean_rmse_price:,.2f}",
    delta=f"{report.model_backtest.mean_rmse_price - report.baseline_backtest.mean_rmse_price:+,.2f} vs naive baseline",
    delta_color="inverse",
)
st.caption(
    "Directional accuracy is judged against a 50% coin flip, not the naive baseline — "
    "the baseline always predicts 'no change,' which has no direction to be right or "
    "wrong about, so it isn't a meaningful comparison for this particular metric. It's "
    "still the right comparison for price error (above), where 'no change' is a real "
    "prediction with a real error to measure."
)

if not report.beats_baseline_on_direction and not report.beats_baseline_on_price_error:
    st.warning(
        f"⚠️ **{model_name} beat neither a coin flip on direction nor the naive "
        "baseline on price error** for this ticker/window. That's a real, useful "
        "result — it means there's no exploitable signal in these features for this "
        "stock right now, not that the app is broken. Try a different model or ticker."
    )

if report.model_backtest.last_fold_index is not None:
    st.subheader('🔬 Most Recent Held-Out Fold: Actual vs Predicted')
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(
        report.model_backtest.last_fold_index, report.model_backtest.last_fold_actual_price,
        color='steelblue', label='Actual', alpha=0.9, linewidth=1.4,
    )
    ax2.plot(
        report.model_backtest.last_fold_index, report.model_backtest.last_fold_predicted_price,
        color='tomato', label=f'{model_name} Predicted', alpha=0.9, linewidth=1.4,
    )
    ax2.plot(
        report.baseline_backtest.last_fold_index, report.baseline_backtest.last_fold_predicted_price,
        color='gray', linestyle='--', label='Naive Predicted', alpha=0.7, linewidth=1.2,
    )
    ax2.set_xlabel('Date'); ax2.set_ylabel('Price')
    ax2.set_title(f"{report.ticker} — Held-Out Fold ({report.model_backtest.folds[-1].n_test} trading days)")
    ax2.legend()
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

target_date_str = report.target_date.strftime("%A, %b %d")
st.subheader(f'🔮 Predicted Close for {target_date_str}: **${report.predicted_next_close:,.2f}**')
if report.interval_low is not None:
    st.caption(
        f"Model: {model_name} · predicted log-return: {report.predicted_log_return:+.4f} "
        f"· last close: ${report.last_close:,.2f} · "
        f"{report.interval_confidence:.0%} interval: ${report.interval_low:,.2f} – ${report.interval_high:,.2f} "
        "(from real walk-forward backtest error, not a guess)"
    )
else:
    st.caption(
        f"Model: {model_name} · predicted log-return: {report.predicted_log_return:+.4f} "
        f"· last close: ${report.last_close:,.2f}"
    )
st.caption(
    "This prediction has been recorded and will be checked against the real close once "
    f"{target_date_str} has passed — see the **Track Record** page for how {model_name} "
    f"has actually done on {report.ticker} (and other tickers) over time."
)

if report.live_quote:
    st.metric(
        "Current Market Price",
        f"${report.live_quote:,.2f}",
        delta=f"Prediction Δ {report.predicted_next_close - report.live_quote:+.2f}",
    )

if report.drift_report is not None and report.drift_report.has_drift:
    st.warning(
        f"⚠️ **Feature drift detected** in {len(report.drift_report.drifted_features)} of "
        f"{len(report.drift_report.features)} feature(s) — the last "
        f"{report.drift_report.current_window} trading days look statistically different from "
        f"the {report.drift_report.reference_window} days before them "
        f"({', '.join(report.drift_report.drifted_features)}). Backtest metrics above were "
        "computed over the full history and may not reflect current conditions as reliably."
    )

st.subheader('📰 News Sentiment')
st.caption(
    "Descriptive context only — recent headlines, not fed into the model. Free news sources "
    "don't provide a historical archive, so there's no honest way to have backtested this."
)
sentiment = st.session_state.sentiment
if sentiment is None or sentiment.label == "unavailable":
    st.caption("No recent headlines found for this ticker.")
else:
    label_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}[sentiment.label]
    st.metric(
        f"{label_emoji} Sentiment ({sentiment.headline_count} recent headlines)",
        sentiment.label.capitalize(),
        delta=f"compound score {sentiment.mean_compound:+.2f}",
        delta_color="off",
    )
