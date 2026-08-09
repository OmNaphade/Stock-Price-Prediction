import streamlit as st

st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
)

import matplotlib.pyplot as plt

from auth_ui import require_authenticated_user
from config import settings
from i18n import render_language_selector, t
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

render_language_selector()

# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not require_authenticated_user(auth_service, title=t("app.auth_title")):
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(t("common.welcome", username=st.session_state.username))
if st.sidebar.button(t("common.logout")):
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
    st.sidebar.warning(t("app.no_av_key_warning"))

st.title(t("app.main_title"))

col_inp, col_model, col_btn = st.columns([3, 2, 1])
with col_inp:
    user_input = st.text_input(
        t("app.ticker_input_label"),
        value=st.session_state.last_ticker or 'AAPL',
        placeholder=t("app.ticker_placeholder"),
        max_chars=settings.max_ticker_length,
    ).strip().upper()
with col_model:
    model_name = st.selectbox(t("app.model_label"), list(AVAILABLE_MODELS.keys()), index=1)
with col_btn:
    st.write("")
    load_btn = st.button(t("app.analyze_button"))

if not st.session_state.load_attempted and not load_btn:
    st.info(t("app.initial_help"))
    st.stop()

if load_btn:
    st.session_state.report = None
    st.session_state.sentiment = None
    st.session_state.last_ticker = user_input
    st.session_state.load_attempted = True

    with st.spinner(t("app.fetching_spinner", model_name=model_name, ticker=user_input)):
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
            st.error(t("app.prediction_error_prefix", error=e))
        except Exception as e:
            st.error(t("app.unexpected_error_prefix", error=e))

report = st.session_state.report
if report is None:
    st.stop()

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
st.subheader(t("app.past_data_summary"))
st.markdown(
    report.ohlcv.describe().round(2).to_html(classes="dataframe", border=0),
    unsafe_allow_html=True,
)

st.subheader(t("app.stock_trends"))
ma100 = report.ohlcv['Close'].rolling(100).mean()
ma200 = report.ohlcv['Close'].rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(report.ohlcv.index, report.ohlcv['Close'], label=t("app.chart_close_label"), alpha=0.45, linewidth=1)
ax.plot(report.ohlcv.index, ma100, label=t("app.chart_ma100_label"), linewidth=2)
ax.plot(report.ohlcv.index, ma200, label=t("app.chart_ma200_label"), linewidth=2)
ax.set_title(t("app.chart_title", ticker=report.ticker))
ax.set_xlabel(t("app.chart_xlabel")); ax.set_ylabel(t("app.chart_ylabel")); ax.legend()
fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.subheader(t("app.backtest_header"))
st.caption(
    t(
        "app.backtest_caption",
        model_name=model_name,
        n_folds=len(report.model_backtest.folds),
    )
)
b1, b2 = st.columns(2)
b1.metric(
    t("app.directional_accuracy_metric"),
    f"{report.model_backtest.mean_directional_accuracy:.1%}",
    delta=t(
        "app.directional_accuracy_delta",
        value=f"{report.model_backtest.mean_directional_accuracy - 0.5:+.1%}",
    ),
)
b2.metric(
    t("app.price_rmse_metric"),
    f"${report.model_backtest.mean_rmse_price:,.2f}",
    delta=t(
        "app.price_rmse_delta",
        value=f"{report.model_backtest.mean_rmse_price - report.baseline_backtest.mean_rmse_price:+,.2f}",
    ),
    delta_color="inverse",
)
st.caption(t("app.directional_accuracy_caption"))

if not report.beats_baseline_on_direction and not report.beats_baseline_on_price_error:
    st.warning(t("app.beats_neither_warning", model_name=model_name))

if report.model_backtest.last_fold_index is not None:
    st.subheader(t("app.held_out_fold_header"))
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(
        report.model_backtest.last_fold_index, report.model_backtest.last_fold_actual_price,
        color='steelblue', label=t("app.actual_label"), alpha=0.9, linewidth=1.4,
    )
    ax2.plot(
        report.model_backtest.last_fold_index, report.model_backtest.last_fold_predicted_price,
        color='tomato', label=t("app.model_predicted_label", model_name=model_name), alpha=0.9, linewidth=1.4,
    )
    ax2.plot(
        report.baseline_backtest.last_fold_index, report.baseline_backtest.last_fold_predicted_price,
        color='gray', linestyle='--', label=t("app.naive_predicted_label"), alpha=0.7, linewidth=1.2,
    )
    ax2.set_xlabel(t("app.chart_xlabel")); ax2.set_ylabel(t("app.chart_ylabel"))
    ax2.set_title(
        t(
            "app.held_out_fold_title",
            ticker=report.ticker,
            n=report.model_backtest.folds[-1].n_test,
        )
    )
    ax2.legend()
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

target_date_str = report.target_date.strftime("%A, %b %d")
st.subheader(
    t("app.predicted_close_header", date=target_date_str, price=f"{report.predicted_next_close:,.2f}")
)
if report.interval_low is not None:
    st.caption(
        t(
            "app.prediction_caption_with_interval",
            model_name=model_name,
            log_return=f"{report.predicted_log_return:+.4f}",
            last_close=f"{report.last_close:,.2f}",
            confidence=f"{report.interval_confidence:.0%}",
            low=f"{report.interval_low:,.2f}",
            high=f"{report.interval_high:,.2f}",
        )
    )
else:
    st.caption(
        t(
            "app.prediction_caption_no_interval",
            model_name=model_name,
            log_return=f"{report.predicted_log_return:+.4f}",
            last_close=f"{report.last_close:,.2f}",
        )
    )
st.caption(
    t(
        "app.track_record_pointer",
        date=target_date_str,
        model_name=model_name,
        ticker=report.ticker,
    )
)

if report.live_quote:
    st.metric(
        t("app.current_market_price_metric"),
        f"${report.live_quote:,.2f}",
        delta=t(
            "app.prediction_delta_label",
            delta=f"{report.predicted_next_close - report.live_quote:+.2f}",
        ),
    )

if report.drift_report is not None and report.drift_report.has_drift:
    st.warning(
        t(
            "app.drift_warning",
            n_drifted=len(report.drift_report.drifted_features),
            n_total=len(report.drift_report.features),
            current_window=report.drift_report.current_window,
            reference_window=report.drift_report.reference_window,
            feature_list=', '.join(report.drift_report.drifted_features),
        )
    )

st.subheader(t("app.news_sentiment_header"))
st.caption(t("app.news_sentiment_caption"))
sentiment = st.session_state.sentiment
if sentiment is None or sentiment.label == "unavailable":
    st.caption(t("app.no_headlines"))
else:
    label_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}[sentiment.label]
    st.metric(
        t("app.sentiment_metric_label", emoji=label_emoji, count=sentiment.headline_count),
        sentiment.label.capitalize(),
        delta=t("app.sentiment_delta", score=f"{sentiment.mean_compound:+.2f}"),
        delta_color="off",
    )
