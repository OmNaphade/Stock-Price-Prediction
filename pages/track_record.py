import streamlit as st

st.set_page_config(page_title="Track Record", page_icon="📈")

import matplotlib.pyplot as plt
import pandas as pd

from auth_ui import require_authenticated_user
from i18n import render_language_selector, t
from services import AVAILABLE_MODELS
from web_context import get_auth_service, get_track_record_service

auth_service = get_auth_service()
track_record_service = get_track_record_service()

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

st.markdown(f"# {t('track_record.title')}")
st.caption(t("track_record.caption"))

all_records = track_record_service.get_track_record(st.session_state.username, limit=2000).records
if not all_records:
    st.info(t("track_record.no_records_info"))
    st.stop()

all_option = t("track_record.all_option")
known_tickers = sorted({r.ticker for r in all_records})
col_ticker, col_model = st.columns(2)
with col_ticker:
    ticker_filter = st.selectbox(t("track_record.ticker_filter_label"), [all_option] + known_tickers)
with col_model:
    model_filter = st.selectbox(t("track_record.model_filter_label"), [all_option] + list(AVAILABLE_MODELS.keys()))

summary = track_record_service.get_track_record(
    st.session_state.username,
    ticker=None if ticker_filter == all_option else ticker_filter,
    model_name=None if model_filter == all_option else model_filter,
    limit=2000,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(t("track_record.resolved_metric"), summary.resolved_count)
m2.metric(t("track_record.pending_metric"), summary.pending_count)
m3.metric(
    t("track_record.directional_accuracy_metric"),
    f"{summary.directional_accuracy:.1%}" if summary.directional_accuracy is not None else "—",
    delta=(
        t("track_record.directional_accuracy_delta", value=f"{summary.directional_accuracy - 0.5:+.1%}")
        if summary.directional_accuracy is not None
        else None
    ),
)
m4.metric(
    t("track_record.mae_metric"),
    f"{summary.mean_abs_pct_error:.2f}%" if summary.mean_abs_pct_error is not None else "—",
)

resolved = [r for r in summary.records if r.is_resolved]

if resolved:
    st.subheader(t("track_record.scatter_header"))
    st.caption(t("track_record.scatter_caption"))

    pred_pct = [(r.predicted_close / r.last_close - 1) * 100 for r in resolved]
    actual_pct = [(r.actual_close / r.last_close - 1) * 100 for r in resolved]
    correct = [bool(r.direction_correct) for r in resolved]
    correct_color, wrong_color = "#2E7D5B", "#B3452C"  # same status pair as the watchlist chart
    colors = [correct_color if c else wrong_color for c in correct]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pred_pct, actual_pct, c=colors, alpha=0.75, s=40, edgecolors="none")
    limit = max(1.0, max(abs(v) for v in [*pred_pct, *actual_pct]) * 1.15)
    perfect_label = t("track_record.perfect_prediction_label")
    ax.plot([-limit, limit], [-limit, limit], color="gray", linestyle="--", linewidth=1, label=perfect_label)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(t("track_record.predicted_change_xlabel"))
    ax.set_ylabel(t("track_record.actual_change_ylabel"))
    ax.set_aspect("equal", adjustable="box")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=correct_color, markersize=8,
            label=t("track_record.correct_direction_label"),
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=wrong_color, markersize=8,
            label=t("track_record.wrong_direction_label"),
        ),
        Line2D([0], [0], color="gray", linestyle="--", label=perfect_label),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info(t("track_record.no_resolved_info"))

st.subheader(t("track_record.history_header"))
col_target_date = t("track_record.col_target_date")
col_ticker_h = t("track_record.col_ticker")
col_model_h = t("track_record.col_model")
col_last_close = t("track_record.col_last_close")
col_predicted = t("track_record.col_predicted")
col_actual = t("track_record.col_actual")
col_direction = t("track_record.col_direction")
col_abs_error_pct = t("track_record.col_abs_error_pct")

history_rows = [
    {
        col_target_date: r.target_date,
        col_ticker_h: r.ticker,
        col_model_h: r.model_name,
        col_last_close: r.last_close,
        col_predicted: r.predicted_close,
        col_actual: r.actual_close if r.is_resolved else None,
        col_direction: (
            t("track_record.direction_correct") if r.direction_correct else t("track_record.direction_wrong")
        ) if r.is_resolved else t("track_record.direction_pending"),
        col_abs_error_pct: r.abs_pct_error if r.is_resolved else None,
    }
    for r in summary.records
]
history_df = pd.DataFrame(history_rows)
st.dataframe(
    history_df.style.format(
        {
            col_last_close: "${:,.2f}",
            col_predicted: "${:,.2f}",
            col_actual: lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
            col_abs_error_pct: lambda v: f"{v:.2f}%" if pd.notna(v) else "—",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
