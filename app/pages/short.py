import warnings
import pickle
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Proje kök dizinini yola ekle (App.py'dan veya doğrudan çalıştırıldığında hata almamak için)
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Mevcut modüllerinizden importlar
# Not: Bu modüllerin kök dizinde veya src içinde olduğundan emin olun
try:
    from config import TICKERS, TICKER_TO_NAME, DATA_DIR, HORIZONS, to_yf_symbol
    from features import build_features, get_feature_cols
except ImportError:
    st.error("Konfigürasyon dosyaları (config.py, features.py) bulunamadı. Lütfen dosya yolunu kontrol edin.")

warnings.filterwarnings("ignore")

def rgba(hex6: str, a: float) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live(ticker: str) -> pd.DataFrame | None:
    try:
        sym = to_yf_symbol(ticker)
        df = yf.download(sym, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_kaggle(ticker: str) -> pd.DataFrame | None:
    base = ticker.replace(".US", "") if ticker.endswith(".US") else ticker
    paths = [DATA_DIR / f"{base}.US_D1.csv", DATA_DIR / f"{ticker}_D1.csv", DATA_DIR / f"{ticker}.csv"]
    for p in paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                df.columns = [c.strip().title() for c in df.columns]
                dc = next((c for c in df.columns if c.lower() in ("date", "datetime")), df.columns[0])
                df[dc] = pd.to_datetime(df[dc], errors="coerce")
                df = df.dropna(subset=[dc]).set_index(dc).sort_index()
                return df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric)
            except: continue
    return None

def load_bundle(ticker, h, algo):
    p = BASE_DIR / "models" / ticker / f"{algo}_{h}d.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None

@st.cache_data(ttl=300, show_spinner=False)
def run_prediction(ticker: str) -> dict:
    raw = load_kaggle(ticker)
    if raw is None: return {}
    full = build_features(raw)
    feat_cols = get_feature_cols(full)
    anchor = full.index[-1]
    
    out = {"anchor": str(anchor.date())}
    for h in HORIZONS:
        hk, probs = f"{h}d", {}
        for algo in ["rf", "lgbm", "xgb"]:
            b = load_bundle(ticker, h, algo)
            if b is None: continue
            X = full.loc[[anchor], b.get("feat_cols", feat_cols)].values.astype(np.float32)
            pr = float(b["model"].predict_proba(X)[0, 1]) if algo != "lgbm" else float(b["model"].predict(X)[0])
            probs[algo] = {"prob": pr, "thr": b.get("threshold", 0.5)}
        
        if probs:
            avg = np.mean([v["prob"] for v in probs.values()])
            avg_thr = np.mean([v["thr"] for v in probs.values()])
            sig = "UP" if avg >= avg_thr else "DOWN"
            out[hk] = {
                "signal": sig, "confidence": avg if sig == "UP" else 1 - avg,
                "prob_up": avg, "votes_up": sum(1 for v in probs.values() if v["prob"] >= v["thr"]),
                "n_algos": len(probs), "algo_probs": {k: v["prob"] for k in probs}
            }
    return out

def render_short_dashboard(selected_ticker: str):
    """App.py içinden çağrılacak ana fonksiyon"""
    
    if not selected_ticker:
        st.info("Lütfen dashboard üzerinden bir hisse seçin.")
        return

    # Stil tanımlamaları (App.py'da yoksa buraya eklenir)
    st.markdown("""
    <style>
    .pred-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:32px;}
    .pred-card{border-radius:12px;padding:20px;border:1px solid;position:relative;overflow:hidden;}
    .card-up{background:#02120a;border-color:rgba(0,196,122,.2);}
    .card-down{background:#120205;border-color:rgba(255,68,85,.2);}
    .sig-up{color:#00c47a; font-weight:700;} .sig-down{color:#ff4455; font-weight:700;}
    </style>
    """, unsafe_allow_html=True)

    # Veri Çekme
    live_df = fetch_live(selected_ticker)
    preds = run_prediction(selected_ticker)

    # Tahmin Kartları
    st.subheader(f"Kısa Vadeli Tahminler: {selected_ticker}")
    
    cols = st.columns(4)
    h_labels = {"1d": "1 Gün", "3d": "3 Gün", "5d": "5 Gün", "7d": "7 Gün"}
    
    for i, hk in enumerate(["1d", "3d", "5d", "7d"]):
        with cols[i]:
            if hk in preds:
                p = preds[hk]
                color = "#00c47a" if p["signal"] == "UP" else "#ff4455"
                st.metric(label=h_labels[hk], value=p["signal"], delta=f"%{p['confidence']*100:.0f} Güven")
                st.progress(p["confidence"])
            else:
                st.write(f"{h_labels[hk]} verisi yok")

    # Grafik Bölümü
    if live_df is not None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=live_df.index, open=live_df['Open'], high=live_df['High'], 
                                     low=live_df['Low'], close=live_df['Close'], name="Fiyat"))
        
        # Tahmin Okları (Basitleştirilmiş)
        if preds:
            last_date = live_df.index[-1]
            last_price = live_df['Close'].iloc[-1]
            for hk, offset in zip(["1d", "3d", "5d", "7d"], [1, 3, 5, 7]):
                if hk in preds:
                    p = preds[hk]
                    direction = 1 if p["signal"] == "UP" else -1
                    fig.add_annotation(x=last_date + pd.Timedelta(days=offset), y=last_price * (1 + direction * 0.02),
                                       text=f"{hk} {p['signal']}", showarrow=True, arrowhead=2)

        fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# App.py doğrudan çağırırsa çalışması için:
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # Test için bir ticker
    render_short_dashboard("AAPL")
