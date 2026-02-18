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
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Trend Hunter AI",
    layout="wide",
    page_icon="⚡"
)


# =========================================================
# ENHANCED DARK THEME CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ===============================
   CSS VARIABLES
================================ */
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

/* ===============================
   BASE
================================ */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: var(--sans);
}

/* Subtle grid texture overlay */
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

/* ===============================
   SIDEBAR
================================ */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1.2rem !important;
}

/* Sidebar brand header */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}

.brand-icon {
    font-size: 22px;
    background: linear-gradient(135deg, var(--amber), #FF6B00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-name {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--text-primary) !important;
    text-transform: uppercase;
}

/* Status dot */
.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 16px;
    padding: 10px 14px;
    background: rgba(46, 204, 113, 0.06);
    border: 1px solid rgba(46, 204, 113, 0.15);
    border-radius: 6px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.status-text {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 1.5px;
    color: var(--green) !important;
    text-transform: uppercase;
}

.sidebar-caption {
    font-size: 11px;
    color: var(--text-dim) !important;
    margin-top: 12px;
    line-height: 1.7;
    padding-left: 2px;
}

/* ===============================
   TYPOGRAPHY (GLOBAL)
================================ */
h1, h2, h3, h4 {
    color: var(--text-primary) !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}

p, span, label {
    color: var(--text-secondary) !important;
}

/* ===============================
   HEADER BLOCK (PRICE AREA)
================================ */
.header-block {
    padding: 32px 0 24px 0;
    position: relative;
}

.ticker-label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--amber) !important;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.big-price {
    font-family: var(--mono);
    font-size: 64px;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -2px;
    line-height: 1;
    text-shadow: 0 0 40px rgba(255,176,0,0.12);
}

.price-sub {
    font-size: 12px;
    font-family: var(--mono);
    color: var(--text-dim) !important;
    letter-spacing: 2px;
    margin-top: 8px;
    text-transform: uppercase;
}

/* Amber separator line */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--amber), transparent);
    margin: 24px 0;
    opacity: 0.4;
}

/* ===============================
   METRIC CARDS  (4-COL GRID)
================================ */
.metric-card {
    background: var(--bg-card);
    padding: 20px 22px;
    border-radius: 10px;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, background 0.25s, transform 0.2s;
    cursor: default;
}

/* Subtle top accent line */
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--amber), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}

.metric-card:hover {
    border-color: var(--amber);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
}

.metric-card:hover::before {
    opacity: 1;
}

.card-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 2.5px;
    color: var(--text-dim) !important;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.card-value {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary) !important;
    line-height: 1.2;
    margin-bottom: 6px;
}

.card-value.top-rank {
    font-size: 16px;
    color: var(--amber) !important;
    letter-spacing: 1px;
}

.card-sub {
    font-size: 11px;
    color: var(--text-dim) !important;
    font-family: var(--mono);
    line-height: 1.5;
}

/* Score bar */
.score-bar-wrapper {
    margin-top: 10px;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--amber-dim), var(--amber));
    transition: width 0.6s ease;
}

/* ===============================
   RANKINGS TABLE / LEADERBOARD
================================ */
.rank-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12px;
}

.rank-table th {
    color: var(--text-dim) !important;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

.rank-table td {
    padding: 10px 14px;
    color: var(--text-secondary) !important;
    border-bottom: 1px solid rgba(28, 36, 50, 0.5);
}

.rank-table tr:hover td {
    background: var(--amber-glow);
    color: var(--text-primary) !important;
}

.rank-num {
    color: var(--text-dim) !important;
    font-size: 10px;
}

.rank-ticker {
    color: var(--amber) !important;
    font-weight: 700;
    letter-spacing: 1px;
}

.rank-score-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(255,176,0,0.1);
    color: var(--amber) !important;
    font-size: 11px;
}

/* ===============================
   SECTION HEADERS
================================ */
.section-header {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim) !important;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ===============================
   STREAMLIT WIDGET OVERRIDES
================================ */
div[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px;
    font-family: var(--mono) !important;
    font-size: 13px;
}

div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--amber) !important;
}

div[data-testid="stSelectbox"] label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: var(--text-dim) !important;
    text-transform: uppercase;
}

/* Plotly chart border */
.js-plotly-plot {
    border-radius: 10px;
    overflow: hidden;
}

/* ===============================
   SCROLLBAR
================================ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }

/* ===============================
   SIGNAL BADGE
================================ */
.signal-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.signal-up      { background: rgba(46,204,113,0.12); color: #2ECC71 !important; border: 1px solid rgba(46,204,113,0.3); }
.signal-down    { background: rgba(231,76,60,0.12);  color: #E74C3C !important; border: 1px solid rgba(231,76,60,0.3); }
.signal-neutral { background: rgba(139,147,165,0.1); color: #8B93A5 !important; border: 1px solid rgba(139,147,165,0.2); }

/* ===============================
   PROB BARS (3-CLASS)
================================ */
.prob-row {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 6px;
}

.prob-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 1px;
    width: 20px;
    color: var(--text-dim) !important;
    text-transform: uppercase;
    flex-shrink: 0;
}

.prob-track {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.prob-fill-up      { height: 100%; border-radius: 2px; background: #2ECC71; }
.prob-fill-neutral { height: 100%; border-radius: 2px; background: #4A5568; }
.prob-fill-down    { height: 100%; border-radius: 2px; background: #E74C3C; }

.prob-pct {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim) !important;
    width: 28px;
    text-align: right;
    flex-shrink: 0;
}

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

    ALL_STOCKS_PATH = ROOT_DIR / "data/all_stocks.csv"
    if not ALL_STOCKS_PATH.exists():
        return None, None, f"Data file not found: {ALL_STOCKS_PATH}"

    try:
        # Load cache
        df_all = pd.read_csv(ALL_STOCKS_PATH)
        # Ensure column names match expectations
        # CSV Cache columns: Ticker, Date, Open, High, Low, Close, Volume, (Adj Close)
        # We need: ticker, datetime, open, high, low, close, volume
        
        # Renaissance of column names
        cols_map = {
            "Ticker": "ticker", "Date": "datetime", 
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        }
        df_all = df_all.rename(columns=cols_map)
        
        # Ensure datetime
        df_all["datetime"] = pd.to_datetime(df_all["datetime"])
        
        # Filter for universe
        # Clean tickers (remove .US if present in universe list matching)
        # Actually our UNIVERSE list likely has pure tickers or with .US
        # Let's standardize to what's in the CSV
        
        dfs = []
        for t in UNIVERSE:
            # Try finding t or t.US
            mask = df_all["ticker"].isin([t, f"{t}.US"])
            if mask.any():
                temp = df_all[mask].copy()
                # Normalize ticker name to just t for consistency
                temp["ticker"] = t 
                dfs.append(temp.tail(400)) # Keep last 400 for feature calc
            
        if not dfs:
            return None, None, "No matching data in all_stocks.csv"
            
        df = pd.concat(dfs)
        df = df.sort_values(["ticker", "datetime"])
        
    except Exception as e:
        return None, None, f"Error loading data: {e}"

    df = add_features(df, normalize=True)

    last_date = df["datetime"].max()
    latest = df[df["datetime"] == last_date].copy()

    if latest.empty:
        return None, None, "Empty latest data"

    # 3-class model: 0=Down, 1=Neutral, 2=Up
    proba_matrix = model.predict_proba(latest[feature_cols])  # shape (n, 3)
    latest["prob_down"]    = proba_matrix[:, 0]
    latest["prob_neutral"] = proba_matrix[:, 1]
    latest["prob_up"]      = proba_matrix[:, 2]

    # Predicted class
    latest["pred_class"] = model.predict(latest[feature_cols])
    latest["pred_label"] = latest["pred_class"].map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})

    # Ranking score: Up probability, normalized 0-100
    up_p = proba_matrix[:, 2]
    min_p, max_p = up_p.min(), up_p.max()
    latest["model_score"] = ((up_p - min_p) / (max_p - min_p + 1e-9)) * 100

    # Percentile rank
    latest["percentile_rank"] = latest["model_score"].rank(pct=True) * 100

    return latest.sort_values("model_score", ascending=False), df, None


results, history_df, error = load_data_and_predict()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">⚡</span>
        <span class="brand-name">Trend Hunter</span>
    </div>
    """, unsafe_allow_html=True)

    if error:
        st.error(error)
        sel_ticker = None
    else:
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:9px;letter-spacing:2px;color:#4A5568;text-transform:uppercase;margin-bottom:6px;">SELECT TICKER</div>', unsafe_allow_html=True)
        sel_ticker = st.selectbox("", results["ticker"], label_visibility="collapsed")

        st.markdown("""
        <div class="status-row">
            <div class="status-dot"></div>
            <span class="status-text">System Active</span>
        </div>
        <div class="sidebar-caption">
            3-Class Ranking Model<br>
            22–63 Day Horizon • Daily Rebalance
        </div>
        <div class="sidebar-caption" style="margin-top:24px;padding-top:16px;border-top:1px solid #1C2432;">
            ⚠ Model-based ranking output.<br>Not investment advice.
        </div>
        """, unsafe_allow_html=True)

    # Top 5 mini-leaderboard in sidebar (sorted by Up score, descending)
    if results is not None and not error:
        st.markdown('<div class="section-header" style="margin-top:32px;">Top Ranked</div>', unsafe_allow_html=True)
        top5 = results.head(5)

        header_html = (
            '<table class="rank-table">'
            '<thead><tr>'
            '<th style="width:28px"></th>'
            '<th>Ticker</th>'
            '<th>Signal</th>'
            '<th>Score</th>'
            '</tr></thead><tbody>'
        )
        body_parts = []
        for i, (_, r) in enumerate(top5.iterrows()):
            ticker = str(r['ticker'])
            score  = f"{r['model_score']:.0f}"
            label  = str(r.get('pred_label', 'UP'))
            badge_cls = {"UP": "signal-up", "DOWN": "signal-down", "NEUTRAL": "signal-neutral"}.get(label, "signal-neutral")
            body_parts.append(
                '<tr>'
                + f'<td class="rank-num">#{i+1}</td>'
                + f'<td class="rank-ticker">{ticker}</td>'
                + f'<td><span class="signal-badge {badge_cls}" style="font-size:9px;padding:1px 6px;">{label}</span></td>'
                + f'<td><span class="rank-score-pill">{score}</span></td>'
                + '</tr>'
            )

        full_table = header_html + "".join(body_parts) + "</tbody></table>"
        st.markdown(full_table, unsafe_allow_html=True)


# =========================================================
# MAIN VIEW
# =========================================================
if sel_ticker and history_df is not None:

    row = results[results["ticker"] == sel_ticker].iloc[0]
    price        = row["close"]
    model_score  = row["model_score"]
    percentile   = row["percentile_rank"]
    mom          = row["mom_126"]
    slope        = row["sma50_slope20"]
    dist         = row["dist_sma_200"]
    pred_label   = row["pred_label"]          # "UP" / "NEUTRAL" / "DOWN"
    prob_up      = row["prob_up"]             # 0-1
    prob_neutral = row["prob_neutral"]
    prob_down    = row["prob_down"]

    is_top = percentile >= 90

    # ── PRICE HEADER ──────────────────────────────────────
    st.markdown(f"""
    <div class="header-block">
        <div class="ticker-label">⚡ {sel_ticker}.US</div>
        <div class="big-price">${price:,.2f}</div>
        <div class="price-sub">Last Close &nbsp;·&nbsp; Daily Chart</div>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # ── METRIC CARDS ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        score_pct  = f"{model_score:.0f}"
        perc_str   = f"{percentile:.0f}"
        badge_cls  = {"UP": "signal-up", "DOWN": "signal-down", "NEUTRAL": "signal-neutral"}.get(pred_label, "signal-neutral")

        up_w  = f"{prob_up * 100:.0f}%"
        ne_w  = f"{prob_neutral * 100:.0f}%"
        dn_w  = f"{prob_down * 100:.0f}%"
        up_p  = f"{prob_up * 100:.0f}%"
        ne_p  = f"{prob_neutral * 100:.0f}%"
        dn_p  = f"{prob_down * 100:.0f}%"

        card1_html = (
            '<div class="metric-card">'
            '<div class="card-label">Model Signal</div>'
            f'<span class="signal-badge {badge_cls}">{pred_label}</span>'
            f'<div class="card-sub">Score {score_pct}/100 &nbsp;·&nbsp; P{perc_str}</div>'
            # UP bar
            '<div class="prob-row">'
            '<span class="prob-label">UP</span>'
            '<div class="prob-track"><div class="prob-fill-up" style="width:' + up_w + '"></div></div>'
            '<span class="prob-pct">' + up_p + '</span>'
            '</div>'
            # NEUTRAL bar
            '<div class="prob-row">'
            '<span class="prob-label">NEU</span>'
            '<div class="prob-track"><div class="prob-fill-neutral" style="width:' + ne_w + '"></div></div>'
            '<span class="prob-pct">' + ne_p + '</span>'
            '</div>'
            # DOWN bar
            '<div class="prob-row">'
            '<span class="prob-label">DN</span>'
            '<div class="prob-track"><div class="prob-fill-down" style="width:' + dn_w + '"></div></div>'
            '<span class="prob-pct">' + dn_p + '</span>'
            '</div>'
            '</div>'
        )
        st.markdown(card1_html, unsafe_allow_html=True)

    with c2:
        mom_color = "#2ECC71" if mom > 0 else "#E74C3C"
        mom_val   = ("+" if mom > 0 else "") + f"{mom:.2f}"
        st.markdown(
            '<div class="metric-card">'
            '<div class="card-label">Momentum 6M</div>'
            f'<div class="card-value" style="color:{mom_color} !important;">{mom_val}</div>'
            '<div class="card-sub">126-day return (normalized)</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c3:
        slope_color = "#2ECC71" if slope > 0 else "#E74C3C"
        slope_val   = ("+" if slope > 0 else "") + f"{slope:.3f}"
        st.markdown(
            '<div class="metric-card">'
            '<div class="card-label">Trend Slope</div>'
            f'<div class="card-value" style="color:{slope_color} !important;">{slope_val}</div>'
            '<div class="card-sub">SMA50 slope (20-day window)</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c4:
        dist_color = "#2ECC71" if dist > 0 else "#E74C3C"
        dist_val   = ("+" if dist > 0 else "") + f"{dist:.2f}"
        st.markdown(
            '<div class="metric-card">'
            '<div class="card-label">Dist. to SMA200</div>'
            f'<div class="card-value" style="color:{dist_color} !important;">{dist_val}</div>'
            '<div class="card-sub">Relative positioning</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── PRICE CHART ───────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:36px;">Price Action · 3 Month</div>', unsafe_allow_html=True)

    hist = history_df[history_df["ticker"] == sel_ticker].tail(120).copy()

    hist["ema20"]  = hist["close"].ewm(span=20).mean()
    hist["ema50"]  = hist["close"].ewm(span=50).mean()
    hist["sma200"] = hist["close"].rolling(200).mean()

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=hist["datetime"],
        open=hist["open"],
        high=hist["high"],
        low=hist["low"],
        close=hist["close"],
        increasing_line_color="#2ECC71",
        increasing_fillcolor="rgba(46,204,113,0.75)",
        decreasing_line_color="#E74C3C",
        decreasing_fillcolor="rgba(231,76,60,0.75)",
        name="Price",
        whiskerwidth=0.5
    ))

    # Moving averages
    fig.add_trace(go.Scatter(
        x=hist["datetime"], y=hist["ema20"],
        line=dict(color="rgba(255,176,0,0.8)", width=1.2, dash="dot"),
        name="EMA20"
    ))

    fig.add_trace(go.Scatter(
        x=hist["datetime"], y=hist["ema50"],
        line=dict(color="rgba(52,152,219,0.8)", width=1.2),
        name="EMA50"
    ))

    fig.add_trace(go.Scatter(
        x=hist["datetime"], y=hist["sma200"],
        line=dict(color="rgba(155,89,182,0.9)", width=1.8),
        name="SMA200"
    ))

    # Volume bars
    vol_colors = [
        "rgba(46,204,113,0.18)" if c >= o else "rgba(231,76,60,0.18)"
        for c, o in zip(hist["close"], hist["open"])
    ]

    fig.add_trace(go.Bar(
        x=hist["datetime"],
        y=hist["volume"],
        yaxis="y2",
        marker_color=vol_colors,
        name="Volume",
        showlegend=False
    ))

    fig.update_layout(
        plot_bgcolor="#080A0D",
        paper_bgcolor="#080A0D",
        font=dict(color="#8B93A5", family="Space Mono, monospace", size=11),
        height=580,
        margin=dict(l=10, r=60, t=20, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#1C2432",
            borderwidth=1,
            font=dict(size=10, color="#8B93A5"),
            orientation="h",
            y=1.04, x=0
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(28,36,50,0.5)",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, color="#4A5568"),
            tickformat="%b %d",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(28,36,50,0.5)",
            gridwidth=1,
            zeroline=False,
            side="right",
            tickfont=dict(size=10, color="#4A5568"),
            tickprefix="$",
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
            showgrid=False,
            showticklabels=False,
        ),
        hoverlabel=dict(
            bgcolor="#111820",
            bordercolor="#1C2432",
            font=dict(color="#E8EAF0", size=11, family="Space Mono, monospace"),
        ),
    )

    # Subtle amber top border via shape
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0, y0=1, x1=0.15, y1=1,
        line=dict(color="#FFB000", width=2),
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;height:300px;flex-direction:column;gap:12px;">
        <div style="font-size:32px;">📡</div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;letter-spacing:3px;color:#4A5568;text-transform:uppercase;">Awaiting Signal</div>
    </div>
    """, unsafe_allow_html=True)
