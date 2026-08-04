import streamlit as st

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈")

import plotly.graph_objects as go

from auth_ui import require_authenticated_user
from data_access.reference import load_equity_list, periods_and_intervals
from data_access.sources import YFinanceSource
from models import AutoRegForecaster
from web_context import get_auth_service

auth_service = get_auth_service()
yfinance_source = YFinanceSource()

# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not require_authenticated_user(auth_service):
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(f"Welcome, {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.is_authenticated = False
    st.session_state.username = ""
    st.rerun()

# ── Stock selector ────────────────────────────────────────────────────────────
stock_dict = load_equity_list()

st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

st.sidebar.markdown("### **Select stock exchange**")
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)

# stock_dict value is the Yahoo Finance ticker symbol (e.g. "BAJFINANCE")
ticker_symbol = stock_dict[stock]
suffix = "BO" if stock_exchange == "BSE" else "NS"
stock_ticker = f"{ticker_symbol}.{suffix}"

st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input(label="Stock ticker code", value=stock_ticker, disabled=True)

periods = periods_and_intervals()

st.sidebar.markdown("### **Select period**")
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))

st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("# 📈 Stock Price Prediction")

# ── Historical candlestick chart ──────────────────────────────────────────────
try:
    stock_data = yfinance_source.get_history_by_period(stock_ticker, period, interval)

    if stock_data is None or stock_data.empty:
        st.warning(
            f"⚠️ No historical data for **{stock_ticker}**. "
            "Try a different exchange (BSE ↔ NSE) or period."
        )
    else:
        st.markdown("## 📊 Historical Data")
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
    st.error(f"❌ Error loading historical data: {e}")

# ── Forecast section ──────────────────────────────────────────────────────────
st.markdown("## 🔮 Multi-Day Forecast")
st.caption(
    "AutoReg model trained on 2 years of daily close prices (90% train / 10% test). "
    "Forecasts beyond a few days compound their own error at every step — treat the "
    "far end of the dotted line as illustrative, not a real price target."
)

forecaster = AutoRegForecaster()

with st.spinner("Training AutoReg model…"):
    try:
        history = yfinance_source.get_history_by_period(stock_ticker, "2y", "1d")
        train_df, test_df, forecast, predictions = (
            forecaster.fit_predict(history[["Close"]]) if not history.empty else (None, None, None, None)
        )
    except Exception as e:
        train_df = test_df = forecast = predictions = None
        st.error(f"❌ Forecast error: {e}")

if train_df is not None and forecast is not None and predictions is not None:
    fig2 = go.Figure(data=[
        go.Scatter(
            x=train_df.index, y=train_df["Close"],
            name="Train", mode="lines", line={"color": "blue"},
        ),
        go.Scatter(
            x=test_df.index, y=test_df["Close"],
            name="Test", mode="lines", line={"color": "orange"},
        ),
        go.Scatter(
            x=predictions.index, y=predictions,
            name="Test Predictions", mode="lines", line={"color": "green"},
        ),
        go.Scatter(
            x=forecast.index, y=forecast,
            name=f"{forecaster.forecast_days}-day Forecast", mode="lines",
            line={"color": "red", "dash": "dot"},
        ),
    ])
    fig2.update_layout(
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning(
        f"⚠️ Could not generate a forecast for **{stock_ticker}**.\n\n"
        "This usually means Yahoo Finance returned insufficient history for this "
        "symbol. Try switching the exchange (BSE ↔ NSE) or selecting a different stock."
    )
