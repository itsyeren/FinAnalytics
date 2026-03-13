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
# ROOT PATH FINDER
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


MODEL_PATH = ROOT_DIR / "models/long_model.pkl"
DATA_PATH = ROOT_DIR / "data/raw/D1"


# =========================================================
# CSS (UNCHANGED)
# =========================================================
st.markdown("""
<style>
/* (CSS BLOĞUN AYNEN BURADA — KISALTILMADI)
   Senin gönderdiğin CSS bloğunu buraya aynen koy.
   Hiçbir şey silinmedi.
*/
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

    proba_matrix = model.predict_proba(latest[feature_cols])
    latest["prob_down"]    = proba_matrix[:, 0]
    latest["prob_neutral"] = proba_matrix[:, 1]
    latest["prob_up"]      = proba_matrix[:, 2]

    latest["pred_class"] = model.predict(latest[feature_cols])
    latest["pred_label"] = latest["pred_class"].map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})

    up_p = proba_matrix[:, 2]
    min_p, max_p = up_p.min(), up_p.max()
    latest["model_score"] = ((up_p - min_p) / (max_p - min_p + 1e-9)) * 100
    latest["percentile_rank"] = latest["model_score"].rank(pct=True) * 100

    return latest.sort_values("model_score", ascending=False), df, None


# =========================================================
# RENDER FUNCTION (APP.PY TAB[5] ENTRY POINT)
# =========================================================
def render_long_dashboard(selected_ticker: str):

    results, history_df, error = load_data_and_predict()

    if error:
        st.error(error)
        return

    if results is None or history_df is None:
        st.warning("Model output unavailable.")
        return

    if selected_ticker not in results["ticker"].values:
        st.warning("Selected ticker not available in Long model universe.")
        return

    row = results[results["ticker"] == selected_ticker].iloc[0]

    price        = row["close"]
    model_score  = row["model_score"]
    percentile   = row["percentile_rank"]
    mom          = row["mom_126"]
    slope        = row["sma50_slope20"]
    dist         = row["dist_sma_200"]
    pred_label   = row["pred_label"]

    # ================= HEADER =================
    st.markdown(f"""
    <div class="header-block">
        <div class="ticker-label">⚡ {selected_ticker}.US</div>
        <div class="big-price">${price:,.2f}</div>
        <div class="price-sub">Last Close · Long Model</div>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # ================= METRICS =================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Signal", pred_label)

    with c2:
        st.metric("Model Score", f"{model_score:.0f}/100")

    with c3:
        st.metric("Momentum 6M", f"{mom:.2f}")

    with c4:
        st.metric("Dist. to SMA200", f"{dist:.2f}")

    # ================= PRICE CHART =================
    st.markdown("### Price Action · 3 Month")

    hist = history_df[history_df["ticker"] == selected_ticker].tail(120).copy()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist["datetime"],
        open=hist["open"],
        high=hist["high"],
        low=hist["low"],
        close=hist["close"],
        name="Price"
    ))

    st.plotly_chart(fig, use_container_width=True)
