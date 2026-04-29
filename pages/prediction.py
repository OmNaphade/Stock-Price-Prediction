import plotly.graph_objects as go
import streamlit as st
import sqlite3
from helper import *
import bcrypt

# ── Database setup ────────────────────────────────────────────────────────────
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username      TEXT PRIMARY KEY,
        password_hash TEXT
    )
''')
conn.commit()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def authenticate(username: str, password: str) -> bool:
    c.execute('SELECT password_hash FROM users WHERE username=?', (username,))
    row = c.fetchone()
    return check_password(password, row[0]) if row else False


def register_user(username: str, password: str) -> bool:
    try:
        c.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def reset_password(username: str, new_password: str) -> bool:
    c.execute(
        'UPDATE users SET password_hash=? WHERE username=?',
        (hash_password(new_password), username)
    )
    conn.commit()
    return c.rowcount > 0


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
    st.session_state.username = ""

# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.is_authenticated:
    st.title("Login to Stock Prediction App")

    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", use_container_width=True):
            if authenticate(username, password):
                st.session_state.is_authenticated = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with col2:
        if st.button("Register New User", use_container_width=True):
            if username and password:
                if register_user(username, password):
                    st.success("Registration successful. Please log in.")
                else:
                    st.error("Username already exists.")
            else:
                st.error("Please enter username and password.")

    with st.expander("Forgot Password?"):
        reset_username     = st.text_input("Username:", key="reset_user")
        reset_new_password = st.text_input("New Password:", type="password", key="reset_pw")
        if st.button("Reset Password"):
            if reset_password(reset_username, reset_new_password):
                st.success("Password reset successful.")
            else:
                st.error("Username not found.")

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title(f"Welcome, {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.is_authenticated = False
    st.session_state.username = ""
    st.rerun()

# ── Stock selector ────────────────────────────────────────────────────────────
stock_dict = fetch_stocks()

st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

st.sidebar.markdown("### **Select stock exchange**")
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)

# stock_dict value is the Yahoo Finance ticker symbol (e.g. "BAJFINANCE")
ticker_symbol = stock_dict[stock]
suffix        = "BO" if stock_exchange == "BSE" else "NS"
stock_ticker  = f"{ticker_symbol}.{suffix}"

st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input(
    label="Stock ticker code",
    value=stock_ticker,
    disabled=True,
)

periods  = fetch_periods_intervals()

st.sidebar.markdown("### **Select period**")
period   = st.sidebar.selectbox("Choose a period", list(periods.keys()))

st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("# 📈 Stock Price Prediction")

# ── Historical candlestick chart ──────────────────────────────────────────────
try:
    stock_data = fetch_stock_history(stock_ticker, period, interval)

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

# ── Prediction section ────────────────────────────────────────────────────────
st.markdown("## 🔮 Stock Prediction")
st.caption("AutoReg model trained on 2 years of daily close prices (90 % train / 10 % test).")

with st.spinner("Training AutoReg model…"):
    try:
        train_df, test_df, forecast, predictions = generate_stock_prediction(stock_ticker)
    except Exception as e:
        train_df = None
        st.error(f"❌ Prediction error: {e}")

# ── FIX: removed the wrong `(forecast >= 0).all()` gate ──────────────────────
# AutoReg can produce any real-valued output; gating on positivity wrongly
# suppressed valid predictions.  We now only check that data was returned.
if train_df is not None and forecast is not None and predictions is not None:
    fig2 = go.Figure(data=[
        go.Scatter(
            x=train_df.index, y=train_df["Close"],
            name="Train", mode="lines", line=dict(color="blue"),
        ),
        go.Scatter(
            x=test_df.index, y=test_df["Close"],
            name="Test", mode="lines", line=dict(color="orange"),
        ),
        go.Scatter(
            x=predictions.index, y=predictions,
            name="Test Predictions", mode="lines", line=dict(color="green"),
        ),
        go.Scatter(
            x=forecast.index, y=forecast,
            name="90-day Forecast", mode="lines",
            line=dict(color="red", dash="dot"),
        ),
    ])
    fig2.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning(
        f"⚠️ Could not generate a prediction for **{stock_ticker}**.\n\n"
        "This usually means Yahoo Finance returned insufficient history for this symbol. "
        "Try switching the exchange (BSE ↔ NSE) or selecting a different stock."
    )
