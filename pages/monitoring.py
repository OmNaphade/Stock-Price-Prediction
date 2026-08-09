import streamlit as st

st.set_page_config(page_title="Model Monitoring", page_icon="📈")

import matplotlib.pyplot as plt
import pandas as pd

from auth_ui import require_admin_user
from i18n import render_language_selector, t
from services import AVAILABLE_MODELS
from theme_ui import apply_theme
from web_context import get_auth_service, get_monitoring_service

auth_service = get_auth_service()
monitoring_service = get_monitoring_service()

render_language_selector()
apply_theme()

# ── AUTH PAGE (admin only — see auth_ui.require_admin_user) ───────────────────
if not require_admin_user(auth_service):
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(t("common.welcome", username=st.session_state.username))
if st.sidebar.button(t("common.logout")):
    st.session_state.is_authenticated = False
    st.session_state.username = ""
    st.rerun()

st.markdown(f"# {t('monitoring.title')}")
st.caption(t("monitoring.admin_caption"))

known_tickers = monitoring_service.get_known_tickers_all_users()
if not known_tickers:
    st.info(t("monitoring.no_backtests_info"))
    st.stop()

all_option = t("monitoring.all_option")
col_ticker, col_model = st.columns(2)
with col_ticker:
    ticker_filter = st.selectbox(t("monitoring.ticker_filter_label"), known_tickers)
with col_model:
    model_filter = st.selectbox(t("monitoring.model_filter_label"), [all_option] + list(AVAILABLE_MODELS.keys()))

summary = monitoring_service.get_summary_all_users(
    ticker=ticker_filter,
    model_name=None if model_filter == all_option else model_filter,
)

if not summary.records:
    st.info(t("monitoring.no_runs_info"))
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric(t("monitoring.logged_days_metric"), summary.logged_days)
m2.metric(
    t("monitoring.latest_accuracy_metric"),
    f"{summary.latest_directional_accuracy:.1%}"
    if summary.latest_directional_accuracy is not None
    else "—",
)
m3.metric(t("monitoring.drift_days_metric"), f"{summary.drift_days} / {summary.logged_days}")

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

st.subheader(t("monitoring.accuracy_chart_header"))
st.caption(t("monitoring.accuracy_chart_caption"))
fig, ax = plt.subplots(figsize=(10, 4))
for name, group in df.groupby("model_name"):
    ax.plot(group["log_date"], group["model_directional_accuracy"], marker="o", label=name)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label=t("monitoring.coin_flip_label"))
ax.set_ylabel(t("monitoring.accuracy_ylabel"))
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader(t("monitoring.rmse_chart_header"))
st.caption(t("monitoring.rmse_chart_caption"))
fig2, ax2 = plt.subplots(figsize=(10, 4))
for name, group in df.groupby("model_name"):
    ax2.plot(
        group["log_date"], group["model_rmse_price"], marker="o",
        label=t("monitoring.rmse_model_label", model_name=name),
    )
baseline_by_date = df.drop_duplicates("log_date")
ax2.plot(
    baseline_by_date["log_date"], baseline_by_date["baseline_rmse_price"],
    color="gray", linestyle="--", marker=".", label=t("monitoring.rmse_baseline_label"),
)
ax2.set_ylabel(t("monitoring.rmse_ylabel"))
ax2.legend(fontsize=9)
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

if summary.drift_days:
    st.warning(
        t(
            "monitoring.drift_warning",
            drift_days=summary.drift_days,
            logged_days=summary.logged_days,
            ticker=ticker_filter,
        )
    )

st.subheader(t("monitoring.history_header"))
history_df = pd.DataFrame(
    {
        t("monitoring.col_date"): [r.log_date for r in reversed(summary.records)],
        t("monitoring.col_user"): [r.username for r in reversed(summary.records)],
        t("monitoring.col_model"): [r.model_name for r in reversed(summary.records)],
        t("monitoring.col_directional_accuracy"): [r.model_directional_accuracy for r in reversed(summary.records)],
        t("monitoring.col_model_rmse"): [r.model_rmse_price for r in reversed(summary.records)],
        t("monitoring.col_baseline_rmse"): [r.baseline_rmse_price for r in reversed(summary.records)],
        t("monitoring.col_drift"): [
            t("monitoring.drift_yes") if r.has_drift else t("monitoring.drift_ok")
            for r in reversed(summary.records)
        ],
        t("monitoring.col_drifted_features"): [r.drifted_feature_count for r in reversed(summary.records)],
    }
)
st.dataframe(
    history_df.style.format(
        {
            t("monitoring.col_directional_accuracy"): lambda v: f"{v:.1%}" if pd.notna(v) else "—",
            t("monitoring.col_model_rmse"): lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
            t("monitoring.col_baseline_rmse"): lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
