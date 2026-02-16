import streamlit as st
import pandas as pd
import joblib
import os
import sys

# path settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.features import add_features

st.title("Long Model")

# Sidebar
ticker = st.sidebar.selectbox(
    "Select stock",
    ["AAPL", "MSFT", "NVDA"]
)

# ===== 1️⃣ DATA LOAD =====
data_path = f"data/raw/D1/{ticker}.US_D1.csv"

df = pd.read_csv(data_path)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# add ticker column
df["ticker"] = ticker

# ===== 2️⃣ FEATURE ENGINEERING =====
df_feat = add_features(df)

# ===== 3️⃣ LOAD MODEL =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "models", "long_model.pkl")

model = joblib.load(MODEL_PATH)

feature_cols = [
    "ret_1","ret_5","ret_21",
    "mom_63","mom_126",
    "vol_21","vol_63",
    "ma_ratio_21_63",
    "drawdown_63"
]

# predict latest data point
X_latest = df_feat[feature_cols].tail(1)

proba = model.predict_proba(X_latest)[0,1]
score_up = 1 - proba

direction = "UP" if score_up > 0.5 else "DOWN"
confidence = score_up if score_up > 0.5 else 1 - score_up

# ===== 4️⃣ UI =====
st.subheader("Model Prediction")

if direction == "UP":
    st.success(f"↑ UP  %{round(confidence*100)}")
else:
    st.error(f"↓ DOWN %{round(confidence*100)}")

# ===== 5️⃣ PRICE CHART =====
st.subheader("Price Chart")
st.line_chart(df.set_index("datetime")["close"])
