import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.features import add_features
from src.config import UNIVERSE

MODEL_PATH = os.path.join(BASE_DIR, "models", "long_model_reg.pkl")

FEATURE_COLS = [
    # Returns
    "ret_1", "ret_5", "ret_21",
    "mom_63", "mom_126",

    # Volatility
    "vol_21", "vol_63",

    # Trend / structure
    "ma_ratio_21_63",
    "drawdown_63",

    # New features
    "rsi",
    "macd_signal",
    "bb_position",
    "volume_ratio",
    "dist_sma_200",
    "dist_sma_50"
]

st.set_page_config(layout="wide", page_title="AI Long Regression Dashboard")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(ticker):
    path = os.path.join(BASE_DIR, f"data/raw/D1/{ticker}.US_D1.csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df["ticker"] = ticker
    df_feat = add_features(df)
    return df, df_feat

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Controls")
ticker = st.sidebar.selectbox("Select Stock", sorted(UNIVERSE))

df, df_feat = load_data(ticker)

if len(df_feat) < 200:
    st.warning("Not enough data.")
    st.stop()

# =========================
# PREDICTION (REGRESSION)
# =========================
X_latest = df_feat[FEATURE_COLS].tail(1)

predicted_return = float(model.predict(X_latest)[0])

signal = "BUY" if predicted_return > 0 else "SELL"

confidence = min(abs(predicted_return) * 8, 1)

# =========================
# HEADER
# =========================
latest_price = df.iloc[-1]["close"]

st.title(f"{ticker} — Long Model (Regression)")

st.markdown("### 📈 Expected Return (6M Horizon)")
st.metric(
    label="Model Prediction",
    value=f"{predicted_return*100:.2f}%",
)

st.markdown("### 📊 Signal")
st.metric(
    label="Direction",
    value=signal,
)

st.markdown("### 🔥 Confidence")
st.progress(confidence)

# =========================
# RISK METRICS
# =========================
returns = df_feat["ret_1"]

vol = returns.std() * np.sqrt(252)
sharpe = returns.mean() / returns.std() * np.sqrt(252)

st.markdown("---")
st.subheader("Risk Metrics")

col1, col2 = st.columns(2)

col1.metric("Annual Volatility", f"{vol*100:.2f}%")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")

# =========================
# FEATURE DISPLAY
# =========================
with st.expander("Model Features (Latest Values)"):
    feat_df = df_feat[FEATURE_COLS].tail(1).T
    feat_df.columns = ["Value"]
    st.dataframe(feat_df)

# =========================
# EXPORT
# =========================
st.markdown("---")

export_df = pd.DataFrame([{
    "Ticker": ticker,
    "Date": df.iloc[-1]["datetime"],
    "Price": latest_price,
    "Expected_Return": predicted_return,
    "Signal": signal,
    "Confidence": confidence
}])

csv = export_df.to_csv(index=False)

st.download_button(
    label="Download Prediction CSV",
    data=csv,
    file_name=f"{ticker}_prediction.csv",
    mime="text/csv"
)
