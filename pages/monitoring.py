import streamlit as st

st.set_page_config(page_title="Model Monitoring", page_icon="📈")

import matplotlib.pyplot as plt
import pandas as pd

from services import AVAILABLE_MODELS
from web_context import get_auth_service, get_monitoring_service

auth_service = get_auth_service()
monitoring_service = get_monitoring_service()

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
    st.session_state.username = ""

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

st.markdown("# 🩺 Model Monitoring")
st.caption(
    "One entry per ticker/model/day — every backtest run updates that day's row rather "
    "than piling up duplicates, so this reflects how each model has actually looked over "
    "time, not how many times someone clicked Analyze. (Scoped to your own analyses — "
    "not a dashboard shared across everyone using this app.)"
)

known_tickers = monitoring_service.get_known_tickers(st.session_state.username)
if not known_tickers:
    st.info(
        "No backtests logged yet. Analyze a ticker on the **app** or **watchlist** page "
        "and it'll show up here."
    )
    st.stop()

col_ticker, col_model = st.columns(2)
with col_ticker:
    ticker_filter = st.selectbox("Ticker", known_tickers)
with col_model:
    model_filter = st.selectbox("Model", ["All"] + list(AVAILABLE_MODELS.keys()))

summary = monitoring_service.get_summary(
    st.session_state.username,
    ticker=ticker_filter,
    model_name=None if model_filter == "All" else model_filter,
)

if not summary.records:
    st.info("No logged runs for this ticker/model combination yet.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Logged days", summary.logged_days)
m2.metric(
    "Latest directional accuracy",
    f"{summary.latest_directional_accuracy:.1%}"
    if summary.latest_directional_accuracy is not None
    else "—",
)
m3.metric("Days with drift detected", f"{summary.drift_days} / {summary.logged_days}")

# Everything below is display shaping only (chart/table layout) — the
# numbers themselves (logged_days, drift_days, latest accuracy) already
# came fully computed from ModelMonitoringService above.
df = pd.DataFrame(
    {
        "log_date": [r.log_date for r in summary.records],
        "model_name": [r.model_name for r in summary.records],
        "model_directional_accuracy": [r.model_directional_accuracy for r in summary.records],
        "model_rmse_price": [r.model_rmse_price for r in summary.records],
        "baseline_rmse_price": [r.baseline_rmse_price for r in summary.records],
    }
)

st.subheader("📈 Directional Accuracy Over Time")
st.caption(
    "Per model, against the 50% coin-flip floor (dashed) — not the naive baseline, "
    "for the same reason the main app doesn't compare them directly there."
)
fig, ax = plt.subplots(figsize=(10, 4))
for name, group in df.groupby("model_name"):
    ax.plot(group["log_date"], group["model_directional_accuracy"], marker="o", label=name)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Coin flip")
ax.set_ylabel("Directional accuracy")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("💲 Price RMSE Over Time")
st.caption("Model vs. the naive baseline, in the same dollar units — lower is better for both.")
fig2, ax2 = plt.subplots(figsize=(10, 4))
for name, group in df.groupby("model_name"):
    ax2.plot(group["log_date"], group["model_rmse_price"], marker="o", label=f"{name} (model)")
baseline_by_date = df.drop_duplicates("log_date")
ax2.plot(
    baseline_by_date["log_date"], baseline_by_date["baseline_rmse_price"],
    color="gray", linestyle="--", marker=".", label="Naive baseline",
)
ax2.set_ylabel("RMSE ($)")
ax2.legend(fontsize=9)
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

if summary.drift_days:
    st.warning(
        f"⚠️ Feature drift was detected on {summary.drift_days} of {summary.logged_days} "
        f"logged day(s) for {ticker_filter}. Backtest metrics from drifted days may not "
        "reflect current market behavior as reliably — see the per-day breakdown below."
    )

st.subheader("📋 Logged Runs")
history_df = pd.DataFrame(
    {
        "Date": [r.log_date for r in reversed(summary.records)],
        "Model": [r.model_name for r in reversed(summary.records)],
        "Directional Accuracy": [r.model_directional_accuracy for r in reversed(summary.records)],
        "Model RMSE": [r.model_rmse_price for r in reversed(summary.records)],
        "Baseline RMSE": [r.baseline_rmse_price for r in reversed(summary.records)],
        "Drift": ["⚠️ Yes" if r.has_drift else "OK" for r in reversed(summary.records)],
        "Drifted Features": [r.drifted_feature_count for r in reversed(summary.records)],
    }
)
st.dataframe(
    history_df.style.format(
        {
            "Directional Accuracy": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
            "Model RMSE": lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
            "Baseline RMSE": lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
