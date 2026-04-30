import streamlit as st

st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import bcrypt
import requests

# ── Data source: yfinance with curl_cffi, fallback to requests ────────────────
try:
    from curl_cffi import requests as curl_requests
    YF_SESSION = curl_requests.Session(impersonate="chrome101")
    USE_CURL = True
except Exception:
    YF_SESSION = requests.Session()
    YF_SESSION.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    })
    USE_CURL = False

import yfinance as yf


# ── Simple Linear Regression ──────────────────────────────────────────────────
class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        denom = np.sum((X - X_mean) ** 2, axis=0)
        denom = np.where(denom == 0, 1e-12, denom)
        self.coef_ = np.sum((X - X_mean) * (y - y_mean), axis=0) / denom
        self.intercept_ = float(y_mean - np.dot(self.coef_, X_mean))

    def predict(self, X):
        return np.dot(np.array(X, dtype=np.float64), self.coef_) + self.intercept_


# ── SQLite auth ───────────────────────────────────────────────────────────────
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
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


# ── Alpha Vantage fetch (primary on cloud, no IP blocks) ─────────────────────
AV_API_KEY = st.secrets.get("AV_API_KEY", "")

def fetch_av(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Alpha Vantage. Returns empty df if key not set."""
    if not AV_API_KEY:
        return pd.DataFrame()
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={ticker}"
            f"&outputsize=full"
            f"&apikey={AV_API_KEY}"
        )
        r = requests.get(url, timeout=15)
        data = r.json().get("Time Series (Daily)", {})
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data).T
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High",
            "3. low": "Low",   "5. adjusted close": "Close",
            "6. volume": "Volume",
        })
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        df = df.loc[start:end]
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


# ── yfinance fetch (works locally, may be blocked on cloud) ──────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    def _clean(df):
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            return pd.DataFrame()
        cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[cols].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["Close"])

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False,
                              threads=False, session=YF_SESSION)
            df = _clean(raw)
            if not df.empty:
                return df
            t = yf.Ticker(ticker, session=YF_SESSION)
            df2 = _clean(t.history(start=start, end=end, auto_adjust=True))
            return df2
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
    raise last_err


@st.cache_data(ttl=600, show_spinner=False)
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Try Alpha Vantage first (reliable on cloud), fall back to yfinance."""
    df = fetch_av(ticker, start, end)
    if not df.empty:
        return df
    return fetch_yf(ticker, start, end)


# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "is_authenticated": False,
    "username": "",
    "df": None,
    "last_ticker": "",
    "load_attempted": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
if not st.session_state.is_authenticated:
    st.title("📈 Stock Prediction App")
    st.subheader("Login / Register")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", use_container_width=True):
            if authenticate(username, password):
                st.session_state.is_authenticated = True
                st.session_state.username = username
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with col2:
        if st.button("Register", use_container_width=True):
            if username and password:
                if register_user(username, password):
                    st.success("Registered! Please log in.")
                else:
                    st.error("Username already exists.")
            else:
                st.error("Enter both username and password.")
    st.stop()


# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.sidebar.title(f"👋 Welcome, {st.session_state.username}")
st.sidebar.caption(
    f"{'✅ curl_cffi' if USE_CURL else '⚠️ requests fallback'} | "
    f"{'✅ Alpha Vantage' if AV_API_KEY else '⚠️ No AV key — yfinance only'}"
)
if st.sidebar.button("Logout"):
    for k, v in _defaults.items():
        st.session_state[k] = v
    st.rerun()

if not AV_API_KEY:
    st.sidebar.warning(
        "⚠️ No Alpha Vantage key set.\n\n"
        "Add `AV_API_KEY` in Render → Environment Variables\n"
        "for reliable data on cloud.\n\n"
        "Get free key: alphavantage.co"
    )

START = '2016-01-01'
END   = '2024-12-31'

st.title('📊 Stock Price Prediction')

col_inp, col_btn = st.columns([4, 1])
with col_inp:
    user_input = st.text_input(
        'Enter Stock Ticker',
        value=st.session_state.last_ticker or 'AAPL',
        placeholder='e.g. AAPL, TSLA, RELIANCE.NS',
    ).strip().upper()
with col_btn:
    st.write("")
    load_btn = st.button("🔍 Load")

if not st.session_state.load_attempted and not load_btn:
    st.info(
        "👆 Enter a ticker and click **🔍 Load** to begin.\n\n"
        "**Examples:** `AAPL` · `TSLA` · `MSFT` · `RELIANCE.NS` · `TCS.NS`"
    )
    st.stop()

if load_btn:
    load_stock_data.clear()
    fetch_yf.clear()
    st.session_state.df = None
    st.session_state.last_ticker = user_input
    st.session_state.load_attempted = True

    bar  = st.progress(0)
    info = st.empty()

    try:
        info.info(f"📡 Fetching data for **{user_input}**...")
        bar.progress(30)

        df_loaded = load_stock_data(user_input, START, END)

        bar.progress(90)
        time.sleep(0.1)

        if df_loaded is None or df_loaded.empty:
            bar.empty()
            info.error(
                f"❌ No data returned for **{user_input}**.\n\n"
                "• Check the ticker symbol is correct.\n"
                "• Indian stocks: append `.NS` (NSE) or `.BO` (BSE).\n"
                "• If this works locally but not on Render, add an "
                "`AV_API_KEY` in Render → Environment Variables."
            )
        else:
            st.session_state.df = df_loaded
            bar.progress(100)
            time.sleep(0.1)
            bar.empty()
            info.success(f"✅ Loaded **{len(df_loaded):,} trading days** for **{user_input}**")

    except Exception as e:
        bar.empty()
        info.error(f"❌ Error: {e}")

if st.session_state.df is None:
    st.stop()

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
df = st.session_state.df.copy()
ticker_label = st.session_state.last_ticker

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].squeeze(), errors='coerce').astype(np.float64)
df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

st.subheader('📋 Past Data Summary')
st.markdown(df.describe().round(2).to_html(classes="dataframe", border=0),
            unsafe_allow_html=True)

st.subheader('📈 Stock Trends')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['Close'], label='Close Price', alpha=0.45, linewidth=1)
ax.plot(df.index, ma100, label='MA 100', linewidth=2)
ax.plot(df.index, ma200, label='MA 200', linewidth=2)
ax.set_title(f"{ticker_label} — Close Price & Moving Averages")
ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
fig.tight_layout(); st.pyplot(fig); plt.close(fig)

df['Open-Close'] = df['Close'] - df['Open']
df['High-Low']   = df['High']  - df['Low']
df['ma5']        = df['Close'].rolling(5).mean()
df['ma10']       = df['Close'].rolling(10).mean()
df['ma20']       = df['Close'].rolling(20).mean()
df = df.dropna()
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

if len(df) < 60:
    st.warning(f"⚠️ Only **{len(df)}** usable rows — not enough to train.")
    st.stop()

features = ['Open-Close', 'High-Low', 'ma5', 'ma10', 'ma20']
X = df[features].values.astype(np.float64)
Y = df['Target'].values.astype(np.float64)
split = int(0.8 * len(df))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

with st.spinner("🧠 Training model..."):
    model = SimpleLinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test).flatten()

mae  = float(np.mean(np.abs(Y_test - predictions)))
rmse = float(np.sqrt(np.mean((Y_test - predictions) ** 2)))
m1, m2 = st.columns(2)
m1.metric("MAE", f"${mae:,.2f}")
m2.metric("RMSE", f"${rmse:,.2f}")

predicted_next = float(model.predict(X[-1:]).flatten()[0])
st.subheader(f'🔮 Predicted Next Close: **${predicted_next:,.2f}**')

try:
    time.sleep(0.5)
    live = yf.Ticker(user_input, session=YF_SESSION).info.get('currentPrice')
    if live:
        st.metric("Current Market Price", f"${float(live):,.2f}",
                  delta=f"Prediction Δ {predicted_next - float(live):+.2f}")
except Exception:
    pass

st.subheader('🧪 Test Set: Actual vs Predicted')
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(Y_test,      color='steelblue', label='Actual',    alpha=0.85, linewidth=1.2)
ax2.plot(predictions, color='tomato',    label='Predicted', alpha=0.85, linewidth=1.2)
ax2.set_xlabel('Test Day Index'); ax2.set_ylabel('Price')
ax2.set_title(f"{ticker_label} — Test Set Predictions"); ax2.legend()
fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
