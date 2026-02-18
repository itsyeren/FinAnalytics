"""
short.py — Consumer Staples Short Model Dashboard
==================================================
Kaggle verisi ile 1-7 iş günü horizon tahminleri
LightGBM + XGBoost + LogReg ensemble (majority vote)
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# =========================================================
# ROOT PATH
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
    st.error("Proje kök dizini bulunamadı.")
    st.stop()

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from config import TICKERS, HORIZONS, MODELS_DIR
    from features import build_features, get_feature_cols
except ImportError as e:
    st.error(f"Import hatası: {e}")
    st.stop()

ALL_STOCKS_CSV = ROOT_DIR / "data" / "all_stocks.csv"
TICKER_TO_NAME = {v: k for k, v in TICKERS.items()}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Short Model — FinAnalytics",
    layout="wide",
    page_icon="⚡"
)

# =========================================================
# DARK AMBER THEME CSS (from long.py)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:    #080A0D;
    --bg-secondary:  #0D1117;
    --bg-card:       #111820;
    --bg-card-hover: #161E28;
    --border:        #1C2432;
    --border-accent: #FFB000;
    --amber:         #FFB000;
    --amber-dim:     #CC8C00;
    --amber-glow:    rgba(255, 176, 0, 0.08);
    --green:         #2ECC71;
    --red:           #E74C3C;
    --text-primary:  #E8EAF0;
    --text-secondary:#8B93A5;
    --text-dim:      #4A5568;
    --mono:          'Space Mono', monospace;
    --sans:          'Outfit', sans-serif;
}

.stApp {
    background-color: var(--bg-primary) !important;
    font-family: var(--sans);
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(255,176,0,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,176,0,0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4 {
    color: var(--text-primary) !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
}
p, span, label { color: var(--text-secondary) !important; }

.header-block { padding: 32px 0 24px 0; }
.ticker-label {
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 3px; color: var(--amber) !important;
    text-transform: uppercase; margin-bottom: 6px;
}
.big-price {
    font-family: var(--mono); font-size: 64px; font-weight: 700;
    color: var(--text-primary) !important; letter-spacing: -2px; line-height: 1;
    text-shadow: 0 0 40px rgba(255,176,0,0.12);
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--amber), transparent);
    margin: 24px 0; opacity: 0.4;
}

.metric-card {
    background: var(--bg-card); padding: 20px 22px;
    border-radius: 10px; border: 1px solid var(--border);
    overflow: hidden; transition: border-color 0.25s, transform 0.2s;
}
.metric-card:hover {
    border-color: var(--amber); transform: translateY(-2px);
}
.card-label {
    font-family: var(--mono); font-size: 9px;
    letter-spacing: 2.5px; color: var(--text-dim) !important;
    text-transform: uppercase; margin-bottom: 10px;
}
.card-value {
    font-family: var(--mono); font-size: 22px; font-weight: 700;
    color: var(--text-primary) !important; margin-bottom: 6px;
}
.card-sub {
    font-size: 11px; color: var(--text-dim) !important;
    font-family: var(--mono);
}

.signal-badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: 2px;
}
.signal-up { background: rgba(46,204,113,0.12); color: #2ECC71 !important; border: 1px solid rgba(46,204,113,0.3); }
.signal-down { background: rgba(231,76,60,0.12); color: #E74C3C !important; border: 1px solid rgba(231,76,60,0.3); }

.section-header {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 3px; text-transform: uppercase;
    color: var(--text-dim) !important;
    margin: 32px 0 16px 0;
    display: flex; align-items: center; gap: 10px;
}
.section-header::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(ttl=3600)
def load_all_stocks():
    if ALL_STOCKS_CSV.exists():
        return pd.read_csv(ALL_STOCKS_CSV, index_col=["Ticker", "Date"], parse_dates=["Date"])
    from data_loader import load_all
    return load_all()


def get_ticker_data(combined, ticker):
    try:
        df = combined.loc[ticker].copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except KeyError:
        return None


def load_model(ticker, horizon, algo):
    path = ROOT_DIR / "models" / ticker / f"{algo}_{horizon}d.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid #1C2432;">
        <span style="font-size:22px;background:linear-gradient(135deg,#FFB000,#FF6B00);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">⚡</span>
        <span style="font-family:'Space Mono',monospace;font-size:13px;font-weight:700;letter-spacing:3px;color:#E8EAF0;text-transform:uppercase;">Short Model</span>
    </div>
    """, unsafe_allow_html=True)

    ticker_list = list(TICKERS.values())
    sel_ticker = st.selectbox(
        "SELECT TICKER",
        ticker_list,
        format_func=lambda t: f"{t} — {TICKER_TO_NAME.get(t, t)}"
    )

    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-top:16px;padding:10px 14px;background:rgba(46,204,113,0.06);border:1px solid rgba(46,204,113,0.15);border-radius:6px;">
        <div style="width:7px;height:7px;border-radius:50%;background:#2ECC71;box-shadow:0 0 8px #2ECC71;"></div>
        <span style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:1.5px;color:#2ECC71;text-transform:uppercase;">System Active</span>
    </div>
    <div style="font-size:11px;color:#4A5568;margin-top:12px;line-height:1.7;">
        LightGBM + XGBoost + LogReg<br>
        1-7 Day Horizon · Majority Vote
    </div>
    <div style="font-size:11px;color:#4A5568;margin-top:24px;padding-top:16px;border-top:1px solid #1C2432;">
        ⚠ Model-based signal output.<br>Not investment advice.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN VIEW
# =========================================================
combined = load_all_stocks()
raw_df = get_ticker_data(combined, sel_ticker)

if raw_df is None or len(raw_df) < 200:
    st.error(f"{sel_ticker} için yeterli veri bulunamadı.")
    st.stop()

price = raw_df["Close"].iloc[-1]

# Header
st.markdown(f"""
<div class="header-block">
    <div class="ticker-label">⚡ {sel_ticker} · SHORT MODEL</div>
    <div class="big-price">${price:,.2f}</div>
    <div style="font-size:12px;font-family:'Space Mono',monospace;color:#4A5568;letter-spacing:2px;margin-top:8px;text-transform:uppercase;">Last Close · 1-7 Day Forecast</div>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

# Feature hesapla + tahmin
try:
    full_df = build_features(raw_df)
    feat_cols = get_feature_cols(full_df)
except Exception as e:
    st.error(f"Feature hatası: {e}")
    st.stop()

# Tüm horizon tahminleri
all_horizons = {}
for h in HORIZONS:
    algo_results = {}
    for algo in ["lgbm", "xgb", "logreg"]:
        bundle = load_model(sel_ticker, h, algo)
        if bundle is None:
            continue
        model = bundle["model"]
        scaler = bundle.get("scaler")
        model_feats = bundle.get("feat_cols", feat_cols)

        try:
            X_pred = full_df[model_feats].dropna().iloc[[-1]].values.astype(np.float32)
            if algo == "lgbm":
                prob = float(model.predict(X_pred)[0])
            elif algo == "xgb":
                prob = float(model.predict_proba(X_pred)[0, 1])
            else:
                X_sc = scaler.transform(X_pred)
                prob = float(model.predict_proba(X_sc)[0, 1])

            threshold = bundle.get("threshold", 0.5)
            algo_results[algo] = {"prob": prob, "label": "UP" if prob >= threshold else "DOWN"}
        except Exception:
            continue

    if algo_results:
        up = sum(1 for v in algo_results.values() if v["label"] == "UP")
        down = len(algo_results) - up
        avg_prob = np.mean([v["prob"] for v in algo_results.values()])
        all_horizons[f"{h}d"] = {
            "vote": "UP" if up > down else "DOWN",
            "up_count": up, "down_count": down,
            "avg_prob": round(avg_prob, 4),
            "algos": algo_results
        }

# Horizon kartları
if all_horizons:
    cols = st.columns(len(all_horizons))
    for i, (h_key, h_data) in enumerate(all_horizons.items()):
        with cols[i]:
            vote = h_data["vote"]
            badge_cls = "signal-up" if vote == "UP" else "signal-down"
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">{h_key} Horizon</div>
                <div style="margin:8px 0"><span class="signal-badge {badge_cls}">{vote}</span></div>
                <div class="card-sub">
                    {h_data['up_count']}↗ / {h_data['down_count']}↘<br>
                    Prob: {h_data['avg_prob']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning(f"{sel_ticker} için short model bulunamadı.")

# Candlestick grafik
st.markdown('<div class="section-header">Price Action · 3 Month</div>', unsafe_allow_html=True)

hist = raw_df.tail(90).copy()
hist["ema20"] = hist["Close"].ewm(span=20).mean()
hist["ema50"] = hist["Close"].ewm(span=50).mean()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist["Open"], high=hist["High"],
    low=hist["Low"], close=hist["Close"],
    increasing_line_color="#2ECC71",
    increasing_fillcolor="rgba(46,204,113,0.75)",
    decreasing_line_color="#E74C3C",
    decreasing_fillcolor="rgba(231,76,60,0.75)",
    name="Price", whiskerwidth=0.5
))

fig.add_trace(go.Scatter(
    x=hist.index, y=hist["ema20"],
    line=dict(color="rgba(255,176,0,0.8)", width=1.2, dash="dot"),
    name="EMA20"
))

fig.add_trace(go.Scatter(
    x=hist.index, y=hist["ema50"],
    line=dict(color="rgba(52,152,219,0.8)", width=1.2),
    name="EMA50"
))

vol_colors = [
    "rgba(46,204,113,0.18)" if c >= o else "rgba(231,76,60,0.18)"
    for c, o in zip(hist["Close"], hist["Open"])
]

fig.add_trace(go.Bar(
    x=hist.index, y=hist["Volume"],
    yaxis="y2", marker_color=vol_colors,
    name="Volume", showlegend=False
))

fig.update_layout(
    plot_bgcolor="#080A0D",
    paper_bgcolor="#080A0D",
    font=dict(color="#8B93A5", family="Space Mono, monospace", size=11),
    height=550,
    margin=dict(l=10, r=60, t=20, b=20),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(13,17,23,0.8)", bordercolor="#1C2432", borderwidth=1,
        font=dict(size=10, color="#8B93A5"),
        orientation="h", y=1.04, x=0
    ),
    xaxis=dict(showgrid=True, gridcolor="rgba(28,36,50,0.5)", zeroline=False,
               tickfont=dict(size=10, color="#4A5568"), tickformat="%b %d"),
    yaxis=dict(showgrid=True, gridcolor="rgba(28,36,50,0.5)", zeroline=False,
               side="right", tickfont=dict(size=10, color="#4A5568"), tickprefix="$"),
    yaxis2=dict(overlaying="y", side="left", showgrid=False, showticklabels=False),
    hoverlabel=dict(bgcolor="#111820", bordercolor="#1C2432",
                    font=dict(color="#E8EAF0", size=11, family="Space Mono, monospace")),
)

fig.add_shape(
    type="line", xref="paper", yref="paper",
    x0=0, y0=1, x1=0.15, y1=1,
    line=dict(color="#FFB000", width=2),
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#4A5568;font-size:0.75rem;padding:2rem 0;">
    ⚠ Model çıktıları yatırım tavsiyesi değildir. Eğitim/araştırma amaçlıdır.
</div>
""", unsafe_allow_html=True)
