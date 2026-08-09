import streamlit as st

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈")

import plotly.graph_objects as go

from auth_ui import require_authenticated_user
from data_access.reference import load_equity_list, periods_and_intervals
from data_access.sources import YFinanceSource
from i18n import render_language_selector, t
from models import AutoRegForecaster
from theme_ui import apply_theme
from web_context import get_auth_service, get_openalgo_source

auth_service = get_auth_service()
yfinance_source = YFinanceSource()

render_language_selector()
apply_theme()

# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not require_authenticated_user(auth_service):
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(t("common.welcome", username=st.session_state.username))
if st.sidebar.button(t("common.logout")):
    st.session_state.is_authenticated = False
    st.session_state.username = ""
    st.rerun()

# ── Stock selector ────────────────────────────────────────────────────────────
stock_dict = load_equity_list()

st.sidebar.markdown(t("prediction.select_stock_header"))
stock = st.sidebar.selectbox(t("prediction.choose_stock_label"), list(stock_dict.keys()))

st.sidebar.markdown(t("prediction.select_exchange_header"))
stock_exchange = st.sidebar.radio(t("prediction.choose_exchange_label"), ("BSE", "NSE"), index=0)

# stock_dict value is the Yahoo Finance ticker symbol (e.g. "BAJFINANCE")
ticker_symbol = stock_dict[stock]
suffix = "BO" if stock_exchange == "BSE" else "NS"
stock_ticker = f"{ticker_symbol}.{suffix}"

# Live OpenAlgo symbol search — purely additive on top of the static CSV
# picker above: unconfigured or untouched, it changes nothing, so this
# never regresses the default flow. A picked match overrides stock_ticker.
openalgo_source = get_openalgo_source()
if openalgo_source.is_configured:
    st.sidebar.markdown(t("prediction.live_search_header"))
    if st.sidebar.checkbox(t("prediction.live_search_toggle_label")):
        query = st.sidebar.text_input(
            t("prediction.live_search_input_label"),
            placeholder=t("prediction.live_search_placeholder"),
        ).strip()
        if query:
            matches = openalgo_source.search_symbols(query)
            if matches:
                labels = [f"{m.name} ({m.symbol}) — {m.exchange}" for m in matches]
                choice = st.sidebar.selectbox(t("prediction.live_search_select_label"), labels)
                stock_ticker = matches[labels.index(choice)].ticker
            else:
                st.sidebar.caption(t("prediction.live_search_no_matches", query=query))

st.sidebar.markdown(t("prediction.ticker_header"))
st.sidebar.text_input(label=t("prediction.ticker_code_label"), value=stock_ticker, disabled=True)

periods = periods_and_intervals()

st.sidebar.markdown(t("prediction.select_period_header"))
period = st.sidebar.selectbox(t("prediction.choose_period_label"), list(periods.keys()))

st.sidebar.markdown(t("prediction.select_interval_header"))
interval = st.sidebar.selectbox(t("prediction.choose_interval_label"), periods[period])

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(f"# {t('prediction.title')}")

# ── Historical candlestick chart ──────────────────────────────────────────────
try:
    stock_data = yfinance_source.get_history_by_period(stock_ticker, period, interval)

    if stock_data is None or stock_data.empty:
        st.warning(t("prediction.no_data_warning", ticker=stock_ticker))
    else:
        st.markdown(t("prediction.historical_data_header"))
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(t("prediction.historical_error", error=e))

# ── Forecast section ──────────────────────────────────────────────────────────
st.markdown(t("prediction.forecast_header"))
st.caption(t("prediction.forecast_caption"))

forecaster = AutoRegForecaster()

with st.spinner(t("prediction.training_spinner")):
    try:
        history = yfinance_source.get_history_by_period(stock_ticker, "2y", "1d")
        train_df, test_df, forecast, predictions = (
            forecaster.fit_predict(history[["Close"]]) if not history.empty else (None, None, None, None)
        )
    except Exception as e:
        train_df = test_df = forecast = predictions = None
        st.error(t("prediction.forecast_error", error=e))

if train_df is not None and forecast is not None and predictions is not None:
    fig2 = go.Figure(data=[
        go.Scatter(
            x=train_df.index, y=train_df["Close"],
            name=t("prediction.train_label"), mode="lines", line={"color": "blue"},
        ),
        go.Scatter(
            x=test_df.index, y=test_df["Close"],
            name=t("prediction.test_label"), mode="lines", line={"color": "orange"},
        ),
        go.Scatter(
            x=predictions.index, y=predictions,
            name=t("prediction.test_predictions_label"), mode="lines", line={"color": "green"},
        ),
        go.Scatter(
            x=forecast.index, y=forecast,
            name=t("prediction.forecast_label", days=forecaster.forecast_days), mode="lines",
            line={"color": "red", "dash": "dot"},
        ),
    ])
    fig2.update_layout(
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning(t("prediction.forecast_failed_warning", ticker=stock_ticker))
