import streamlit as st

st.set_page_config(page_title="Track Record", page_icon="📈")

import matplotlib.pyplot as plt
import pandas as pd

from auth_ui import require_authenticated_user
from services import AVAILABLE_MODELS
from web_context import get_auth_service, get_track_record_service

auth_service = get_auth_service()
track_record_service = get_track_record_service()

if "track_record_resolved_this_session" not in st.session_state:
    st.session_state.track_record_resolved_this_session = False

# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not require_authenticated_user(auth_service):
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

st.markdown("# 📊 Track Record")
st.caption(
    "Every prediction you make anywhere in this app is recorded before its outcome is "
    "known, then checked against the real close once its target date has passed. "
    "This page is your receipts — not a claim, a record. (Only your own predictions — "
    "everyone's track record here is their own.)"
)

all_records = track_record_service.get_track_record(st.session_state.username, limit=2000).records
if not all_records:
    st.info(
        "No predictions recorded yet. Analyze a ticker on the **app** or **watchlist** "
        "page, and it'll show up here once its target date has passed."
    )
    st.stop()

known_tickers = sorted({r.ticker for r in all_records})
col_ticker, col_model = st.columns(2)
with col_ticker:
    ticker_filter = st.selectbox("Ticker", ["All"] + known_tickers)
with col_model:
    model_filter = st.selectbox("Model", ["All"] + list(AVAILABLE_MODELS.keys()))

summary = track_record_service.get_track_record(
    st.session_state.username,
    ticker=None if ticker_filter == "All" else ticker_filter,
    model_name=None if model_filter == "All" else model_filter,
    limit=2000,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Resolved predictions", summary.resolved_count)
m2.metric("Awaiting target date", summary.pending_count)
m3.metric(
    "Directional accuracy",
    f"{summary.directional_accuracy:.1%}" if summary.directional_accuracy is not None else "—",
    delta=(
        f"{summary.directional_accuracy - 0.5:+.1%} vs. coin flip"
        if summary.directional_accuracy is not None
        else None
    ),
)
m4.metric(
    "Mean absolute error",
    f"{summary.mean_abs_pct_error:.2f}%" if summary.mean_abs_pct_error is not None else "—",
)

resolved = [r for r in summary.records if r.is_resolved]

if resolved:
    st.subheader("🎯 Predicted vs. Actual Change")
    st.caption(
        "Each point is one resolved prediction. The x-axis is what the model predicted "
        "would change; the y-axis is what actually happened. Points on the diagonal "
        "line were spot-on; points in the top-right or bottom-left quadrants at least "
        "called the right direction, even if the size was off."
    )

    pred_pct = [(r.predicted_close / r.last_close - 1) * 100 for r in resolved]
    actual_pct = [(r.actual_close / r.last_close - 1) * 100 for r in resolved]
    correct = [bool(r.direction_correct) for r in resolved]
    correct_color, wrong_color = "#2E7D5B", "#B3452C"  # same status pair as the watchlist chart
    colors = [correct_color if c else wrong_color for c in correct]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pred_pct, actual_pct, c=colors, alpha=0.75, s=40, edgecolors="none")
    limit = max(1.0, max(abs(v) for v in [*pred_pct, *actual_pct]) * 1.15)
    ax.plot([-limit, limit], [-limit, limit], color="gray", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Predicted change (%)")
    ax.set_ylabel("Actual change (%)")
    ax.set_aspect("equal", adjustable="box")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=correct_color, markersize=8, label="Correct direction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=wrong_color, markersize=8, label="Wrong direction"),
        Line2D([0], [0], color="gray", linestyle="--", label="Perfect prediction"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No resolved predictions yet for this filter — check back after their target dates pass.")

st.subheader("📋 Prediction History")
history_rows = [
    {
        "Target Date": r.target_date,
        "Ticker": r.ticker,
        "Model": r.model_name,
        "Last Close": r.last_close,
        "Predicted": r.predicted_close,
        "Actual": r.actual_close if r.is_resolved else None,
        "Direction": (
            "✅ Correct" if r.direction_correct else "❌ Wrong"
        ) if r.is_resolved else "⏳ Pending",
        "Abs Error %": r.abs_pct_error if r.is_resolved else None,
    }
    for r in summary.records
]
history_df = pd.DataFrame(history_rows)
st.dataframe(
    history_df.style.format(
        {
            "Last Close": "${:,.2f}",
            "Predicted": "${:,.2f}",
            "Actual": lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
            "Abs Error %": lambda v: f"{v:.2f}%" if pd.notna(v) else "—",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
