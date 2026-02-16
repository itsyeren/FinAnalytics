import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import sys
import os
import glob
from pathlib import Path

# =========================================================
# 1. ROOT PATH FINDER
# =========================================================
def find_project_root():
    current = Path(__file__).resolve()
    for _ in range(6):
        if (current / "src").exists() and (current / "models").exists():
            return current
        current = current.parent
    return None

ROOT_DIR = find_project_root()
if ROOT_DIR is None:
    st.error("Project root not found.")
    st.stop()

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# =========================================================
# IMPORTS
# =========================================================
try:
    from src.config import UNIVERSE
    from src.features import add_features
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

MODEL_PATH = ROOT_DIR / "models/trend_classifier_model.pkl"
DATA_PATH = ROOT_DIR / "data/raw/D1"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Trend Hunter AI",
    layout="wide",
    page_icon="⚡"
)

# =========================================================
# DARK THEME CSS
# =========================================================
st.markdown("""
<style>
.stApp { background-color: #050505; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
.big-price { font-size: 56px; font-weight: 700; color: white; }
.price-green { color: #00ff41; font-size: 20px; margin-left: 12px; }
.price-red { color: #ff3333; font-size: 20px; margin-left: 12px; }
.metric-card {
    background-color: #111;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #222;
}
.card-title { font-size: 12px; color: #666; margin-bottom: 6px; }
.card-value { font-size: 28px; font-weight: 700; }
.card-sub { font-size: 14px; color: #aaa; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA + PREDICTION
# =========================================================
@st.cache_data(ttl=3600)
def load_data_and_predict():

    if not MODEL_PATH.exists():
        return None, None, "Model not found"

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    files = glob.glob(str(DATA_PATH / "*.US_D1.csv"))
    available = {os.path.basename(f).split(".")[0] for f in files}
    tickers = [t for t in UNIVERSE if t in available]

    dfs = []
    for t in tickers:
        try:
            df = pd.read_csv(DATA_PATH / f"{t}.US_D1.csv")
            df["ticker"] = t
            dfs.append(df.tail(400))
        except:
            continue

    if not dfs:
        return None, None, "No data"

    df = pd.concat(dfs)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["ticker", "datetime"])

    df = add_features(df, normalize=True)

    last_date = df["datetime"].max()
    latest = df[df["datetime"] == last_date].copy()

    if latest.empty:
        return None, None, "Empty latest data"

    probs = model.predict_proba(latest[feature_cols])[:, 1]

    # Normalize AI score 0-100
    min_p, max_p = probs.min(), probs.max()
    latest["ai_score"] = ((probs - min_p) / (max_p - min_p + 1e-9)) * 100

    return latest.sort_values("ai_score", ascending=False), df, None


results, history_df, error = load_data_and_predict()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚡ TREND HUNTER AI")
    if error:
        st.error(error)
        sel_ticker = None
    else:
        sel_ticker = st.selectbox("Select Ticker", results["ticker"])
        st.success("System Active")
        st.caption("Daily Data • 1M Horizon")

# =========================================================
# MAIN VIEW
# =========================================================
if sel_ticker and history_df is not None:

    row = results[results["ticker"] == sel_ticker].iloc[0]
    price = row["close"]

    st.markdown(f"""
    <div style="display:flex;align-items:baseline;">
        <div class="big-price">${price:,.2f}</div>
    </div>
    <div style="color:#666;font-size:12px;margin-bottom:30px;">
        {sel_ticker}.US • Last Close
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # METRIC CARDS
    # =====================================================
    c1, c2, c3, c4 = st.columns(4)

    ai_score = row["ai_score"]
    mom = row["mom_126"]
    slope = row["sma50_slope20"]
    dist = row["dist_sma_200"]

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">AI SIGNAL</div>
            <div class="card-value">{'BUY' if ai_score>60 else 'WAIT'}</div>
            <div class="card-sub">Score: {ai_score:.0f}/100</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">MOMENTUM 6M</div>
            <div class="card-value">{mom:.2f}</div>
            <div class="card-sub">126-day return</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">TREND SLOPE</div>
            <div class="card-value">{slope:.3f}</div>
            <div class="card-sub">SMA50 slope</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">SMA200 DIST</div>
            <div class="card-value">{dist:.2f}</div>
            <div class="card-sub">Distance to SMA200</div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # ADVANCED CANDLESTICK
    # =====================================================
    st.markdown("### 3-Month Price Action")

    hist = history_df[history_df["ticker"] == sel_ticker].tail(120).copy()

    hist["ema20"] = hist["close"].ewm(span=20).mean()
    hist["ema50"] = hist["close"].ewm(span=50).mean()
    hist["sma200"] = hist["close"].rolling(200).mean()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist["datetime"],
        open=hist["open"],
        high=hist["high"],
        low=hist["low"],
        close=hist["close"],
        increasing_line_color="#00ff41",
        decreasing_line_color="#ff3333",
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=hist["datetime"],
        y=hist["ema20"],
        line=dict(width=1),
        name="EMA20"
    ))

    fig.add_trace(go.Scatter(
        x=hist["datetime"],
        y=hist["ema50"],
        line=dict(width=1),
        name="EMA50"
    ))

    fig.add_trace(go.Scatter(
        x=hist["datetime"],
        y=hist["sma200"],
        line=dict(width=2),
        name="SMA200"
    ))

    fig.add_trace(go.Bar(
        x=hist["datetime"],
        y=hist["volume"],
        yaxis="y2",
        opacity=0.25,
        name="Volume"
    ))

    fig.update_layout(
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="#aaa"),
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        yaxis=dict(
            showgrid=True,
            gridcolor="#1f1f1f",
            side="right"
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
            showgrid=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Waiting for data...")
