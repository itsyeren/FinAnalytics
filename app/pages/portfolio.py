"""
portfolio.py — Portföy Yönetimi & 3-Model Tahmin Sayfası
=========================================================
Sidebar'dan hisse seçip portföye ekle → Toplam değer, ağırlıklar, temettü →
Short / Mid / Long model tahminleri ile portföy projeksiyonu.
"""

import hashlib
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

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

# =========================================================
# IMPORTS (projeden)
# =========================================================
try:
    from config import TICKERS as ROOT_TICKERS, HORIZONS, MODELS_DIR
    from features import build_features, get_feature_cols
    from src.config import UNIVERSE as LONG_UNIVERSE
    from src.features import add_features as add_long_features
except ImportError as e:
    st.error(f"Import hatası: {e}")
    st.stop()

# Alpaca API (Mid model) — optional
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    ALPACA_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")

    if ALPACA_KEY and ALPACA_SECRET:
        from alpaca.data.historical import StockHistoricalDataClient
        alpaca_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        HAS_ALPACA = True
    else:
        HAS_ALPACA = False
        alpaca_client = None
except Exception:
    HAS_ALPACA = False
    alpaca_client = None

# Ticker listesi (tüm hisseler)
ALL_TICKERS_MAP = ROOT_TICKERS.copy()
TICKER_TO_NAME = {v: k for k, v in ALL_TICKERS_MAP.items()}

# Model dizinleri
SHORT_MODELS_DIR = ROOT_DIR / "models"
LONG_MODEL_PATH = ROOT_DIR / "models" / "long_model.pkl"
ALL_STOCKS_CSV = ROOT_DIR / "data" / "all_stocks.csv"
from config import DATA_DIR as KAGGLE_DATA_DIR  # ~/Downloads/archive/D1
MULTI_SEKTOR_DIR = ROOT_DIR / "multi_sektor_analiz"
SECTOR_JSON = MULTI_SEKTOR_DIR / "reports" / "sector_optimized_params.json"
if not SECTOR_JSON.exists():
    SECTOR_JSON = MULTI_SEKTOR_DIR / "sector_optimized_params.json"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Portföy Yönetimi — FinAnalytics",
    layout="wide",
    page_icon="💼"
)

# =========================================================
# CSS — Multi Sektör teması (Inter, gradient kartlar)
# + Short model için dark candlestick tema override
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-primary:    #0a192f;
    --bg-secondary:  #112240;
    --bg-card:       #1a1a2e;
    --bg-card-alt:   #16213e;
    --border:        rgba(255,255,255,0.08);
    --accent-teal:   #64ffda;
    --accent-purple: #667eea;
    --accent-pink:   #c084fc;
    --green:         #00b894;
    --red:           #e17055;
    --amber:         #FFB000;
    --text-primary:  #e6f1ff;
    --text-secondary:#8892b0;
    --text-dim:      #4A5568;
    --mono:          'Space Mono', monospace;
    --sans:          'Inter', sans-serif;
}

.stApp {
    font-family: var(--sans);
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a192f 0%, #112240 100%);
}

/* Metric cards */
.pf-metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-alt) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    text-align: center;
    transition: transform 0.2s ease;
}
.pf-metric-card:hover { transform: translateY(-2px); }
.pf-metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.pf-metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}
.pf-metric-delta-up { color: var(--accent-teal); font-size: 1rem; font-weight: 600; }
.pf-metric-delta-down { color: var(--red); font-size: 1rem; font-weight: 600; }

/* Signal badges */
.pf-signal-al {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: #fff; padding: 0.35rem 1rem; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}
.pf-signal-bekle {
    background: linear-gradient(135deg, #fdcb6e, #f39c12);
    color: #2d3436; padding: 0.35rem 1rem; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}
.pf-signal-sat {
    background: linear-gradient(135deg, #e17055, #d63031);
    color: #fff; padding: 0.35rem 1rem; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}

/* Long model signal badges (amber theme) */
.pf-signal-up {
    background: rgba(46,204,113,0.12); color: #2ECC71 !important;
    border: 1px solid rgba(46,204,113,0.3);
    padding: 3px 10px; border-radius: 4px;
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: 2px; display: inline-block;
}
.pf-signal-down {
    background: rgba(231,76,60,0.12); color: #E74C3C !important;
    border: 1px solid rgba(231,76,60,0.3);
    padding: 3px 10px; border-radius: 4px;
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: 2px; display: inline-block;
}
.pf-signal-neutral {
    background: rgba(139,147,165,0.1); color: #8B93A5 !important;
    border: 1px solid rgba(139,147,165,0.2);
    padding: 3px 10px; border-radius: 4px;
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: 2px; display: inline-block;
}

/* Section header */
.pf-section-header {
    font-size: 1.2rem; font-weight: 600; color: #ccd6f6;
    margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(100, 255, 218, 0.2);
}

/* Dashboard header */
.pf-header h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.pf-header p { color: #8892b0; font-size: 0.95rem; }

/* Hide branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Table */
.pf-table {
    width: 100%; border-collapse: collapse; font-size: 0.9rem;
}
.pf-table th {
    background: rgba(100, 255, 218, 0.1); color: #64ffda;
    padding: 0.8rem; text-align: left; font-weight: 600;
}
.pf-table td {
    padding: 0.7rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #ccd6f6;
}

/* Dividend badge */
.pf-div-yes {
    background: rgba(46,204,113,0.15); color: #2ECC71;
    padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600;
}
.pf-div-no {
    background: rgba(139,147,165,0.12); color: #8892b0;
    padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def signal_badge_mid(signal: str) -> str:
    css = {"AL": "pf-signal-al", "BEKLE": "pf-signal-bekle", "SAT": "pf-signal-sat"}.get(signal, "pf-signal-bekle")
    return f'<span class="{css}">{signal}</span>'


def signal_badge_long(label: str) -> str:
    css = {"UP": "pf-signal-up", "DOWN": "pf-signal-down", "NEUTRAL": "pf-signal-neutral"}.get(label, "pf-signal-neutral")
    return f'<span class="{css}">{label}</span>'


@st.cache_data(ttl=3600)
def fetch_dividend_info(tickers: list) -> dict:
    """yfinance ile temettü bilgisi çeker."""
    result = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            # Try multiple keys for robustness
            div_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0
            div_yield = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0
            
            result[t] = {
                "pays_dividend": div_rate > 0,
                "forward_annual_div": round(div_rate, 4),
                "div_yield_pct": round(div_yield * 100, 2),
            }
        except Exception:
            result[t] = {"pays_dividend": False, "forward_annual_div": 0, "div_yield_pct": 0}
    return result



@st.cache_data(ttl=3600)
def load_sector_json() -> dict | None:
    if SECTOR_JSON.exists():
        with open(SECTOR_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_mid_prediction(report: dict, ticker: str):
    """sector_optimized_params.json'dan tahmin bul."""
    if report is None:
        return None, None
    for sector_name, sector_data in report.get("sectors", {}).items():
        if ticker in sector_data.get("stock_predictions", {}):
            return sector_data["stock_predictions"][ticker], sector_name
    return None, None



@st.cache_data(ttl=3600)
def fetch_alpaca_history(symbol: str, days_back: int = 365):
    """Alpaca API'den fiyat verisi çeker (mid model için)."""
    if not HAS_ALPACA or alpaca_client is None:
        return None
    try:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d")
        )
        bars = alpaca_client.get_stock_bars(req)
        df = bars.df.reset_index()
        df = df[df["symbol"] == symbol].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_all_stocks_cache():
    """data/all_stocks.csv veya Kaggle D1 dizininden tüm hisseleri yükle."""
    if ALL_STOCKS_CSV.exists():
        df = pd.read_csv(ALL_STOCKS_CSV, index_col=["Ticker", "Date"], parse_dates=["Date"])
        return df
    # Fallback: Kaggle D1 dizininden oku
    from data_loader import load_all
    return load_all()


def load_kaggle_ticker(ticker: str) -> pd.DataFrame | None:
    """all_stocks.csv cache veya Kaggle D1'den ticker verisi yükler."""
    try:
        combined = _load_all_stocks_cache()
        if ticker in combined.index.get_level_values("Ticker"):
            df = combined.loc[ticker].copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            needed = ["Open", "High", "Low", "Close", "Volume"]
            if all(c in df.columns for c in needed):
                df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
                df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
                if len(df) >= 200:
                    return df
    except Exception as e:
        st.error(f"Veri yükleme hatası ({ticker}): {e}")
    return None


def load_short_model(ticker: str, horizon: int, algo: str):
    """Short model pkl dosyasını yükler."""
    path = SHORT_MODELS_DIR / ticker / f"{algo}_{horizon}d.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================
# SESSION STATE — Portföy
# =========================================================
if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = {}

portfolio = st.session_state["portfolio"]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 💼 Portföy Oluştur")
    st.markdown("---")

    ticker_options = list(ALL_TICKERS_MAP.values())
    selected = st.selectbox(
        "🔍 Hisse Seçin",
        ticker_options,
        format_func=lambda t: f"{t} — {TICKER_TO_NAME.get(t, t)}",
        key="pf_ticker_select"
    )

    col_q, col_p = st.columns(2)
    with col_q:
        quantity = st.number_input("Adet", min_value=1, value=10, step=1, key="pf_qty")
    with col_p:
        buy_price = st.number_input("Alış ($)", min_value=0.01, value=100.0, step=0.01, key="pf_price")

    if st.button("➕ Portföye Ekle", use_container_width=True):
        if selected:
            portfolio[selected] = {"quantity": int(quantity), "buy_price": float(buy_price)}
            st.success(f"✓ {selected} eklendi ({quantity} adet @ ${buy_price:.2f})")
            st.rerun()

    # Mevcut portföy listesi
    if portfolio:
        st.markdown("---")
        st.markdown("### 📋 Portföyüm")
        remove_ticker = None
        for t, info in portfolio.items():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**{t}** — {info['quantity']} ad. @ ${info['buy_price']:.2f}")
            with c2:
                if st.button("🗑️", key=f"rm_{t}"):
                    remove_ticker = t
        if remove_ticker:
            del portfolio[remove_ticker]
            st.rerun()

        st.markdown("---")
        if st.button("🧹 Portföyü Temizle", use_container_width=True):
            st.session_state["portfolio"] = {}
            st.rerun()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="pf-header" style="text-align:center; padding:1rem 0 2rem 0;">
    <h1>💼 Portföy Yönetimi & Tahmin</h1>
    <p>Hisseleriniz için Short · Mid · Long model tahminleri</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# BOŞ PORTFÖY
# =========================================================
if not portfolio:
    st.info("Portföyünüz boş. Sol menüden hisse ekleyerek başlayın.")
    st.stop()


# =========================================================
# VERİ TOPLAMA — Son fiyatlar & temettü
# =========================================================
tickers_in_portfolio = list(portfolio.keys())

# Temettü bilgisi (yfinance)
with st.spinner("Temettü bilgileri yükleniyor..."):
    div_info = fetch_dividend_info(tickers_in_portfolio)

# Son fiyatları yfinance'den al (güvenilir ve güncel)
@st.cache_data(ttl=900)
def get_current_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="5d")
            if not h.empty:
                prices[t] = float(h["Close"].iloc[-1])
        except Exception:
            prices[t] = 0.0
    return prices

with st.spinner("Güncel fiyatlar yükleniyor..."):
    current_prices = get_current_prices(tickers_in_portfolio)


# =========================================================
# PORTFÖY ÖZET KARTLARI
# =========================================================
st.markdown('<div class="pf-section-header">📊 Portföy Özeti</div>', unsafe_allow_html=True)

# Hesaplamalar
total_value = 0
total_cost = 0
rows_summary = []

for t in tickers_in_portfolio:
    qty = portfolio[t]["quantity"]
    buy = portfolio[t]["buy_price"]
    cur = current_prices.get(t, buy)
    value = qty * cur
    cost = qty * buy
    total_value += value
    total_cost += cost

    div_data = div_info.get(t, {})
    annual_div = div_data.get("forward_annual_div", 0)
    est_div_income = qty * annual_div

    rows_summary.append({
        "ticker": t,
        "name": TICKER_TO_NAME.get(t, t),
        "qty": qty,
        "buy_price": buy,
        "current_price": cur,
        "value": value,
        "cost": cost,
        "pnl": value - cost,
        "pnl_pct": ((cur / buy) - 1) * 100 if buy > 0 else 0,
        "pays_div": div_data.get("pays_dividend", False),
        "div_yield": div_data.get("div_yield_pct", 0),
        "annual_div_per_share": annual_div,
        "est_annual_div_income": est_div_income,
    })

total_pnl = total_value - total_cost
total_pnl_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0
total_div_income = sum(r["est_annual_div_income"] for r in rows_summary)

# Metric Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="pf-metric-card">
        <div class="pf-metric-label">Toplam Değer</div>
        <div class="pf-metric-value">${total_value:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="pf-metric-card">
        <div class="pf-metric-label">Toplam Maliyet</div>
        <div class="pf-metric-value">${total_cost:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    delta_cls = "pf-metric-delta-up" if total_pnl >= 0 else "pf-metric-delta-down"
    st.markdown(f"""
    <div class="pf-metric-card">
        <div class="pf-metric-label">Toplam Kar/Zarar</div>
        <div class="pf-metric-value">${total_pnl:+,.2f}</div>
        <div class="{delta_cls}">{total_pnl_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="pf-metric-card">
        <div class="pf-metric-label">Tahmini Yıllık Temettü</div>
                <div class="pf-metric-value">${total_div_income:,.2f}</div>
        <div style="color:#8892b0;font-size:0.75rem;margin-top:4px;">yfinance forward dividend</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# PROJECTION CHARTS
# =========================================================
st.markdown('<div class="pf-section-header">📈 Portföy Değer Projeksiyonu</div>', unsafe_allow_html=True)

# Calculate Projections
report = load_sector_json()
proj_current = total_value
proj_1m = 0
proj_3m = 0

asset_proj_3m = [] # For Pie Chart

for r in rows_summary:
    t = r["ticker"]
    qty = r["qty"]
    current_p = r["current_price"]

    # Get Mid Predictions
    pred, _ = get_mid_prediction(report, t)

    # 1 Month Logic
    if pred and "tahmin_1ay" in pred:
        p_1m = pred["tahmin_1ay"]
    else:
        p_1m = current_p # Fallback to current if no pred

    # 3 Month Logic
    if pred and "tahmin_3ay" in pred:
        p_3m = pred["tahmin_3ay"]
    else:
        p_3m = current_p # Fallback

    val_1m = qty * p_1m
    val_3m = qty * p_3m

    proj_1m += val_1m
    proj_3m += val_3m

    asset_proj_3m.append({"ticker": t, "value_3m": val_3m})

# Data for Chart
x_dates = ["Bugün", "1 Ay Sonra", "3 Ay Sonra"]
y_values = [proj_current, proj_1m, proj_3m]

# Calculate Growth for Metric Display
growth_1m = ((proj_1m / proj_current) - 1) * 100 if proj_current > 0 else 0
growth_3m = ((proj_3m / proj_current) - 1) * 100 if proj_current > 0 else 0

# Short Term (1 Week) Interpolation
# Assume linear path to 1 Month target
days_1m = 30
days_1w = 7
if proj_current > 0:
    daily_growth = (proj_1m / proj_current) ** (1/days_1m) - 1
    proj_1w = proj_current * ((1 + daily_growth) ** days_1w)
else:
    proj_1w = 0
    
growth_1w = ((proj_1w / proj_current) - 1) * 100 if proj_current > 0 else 0

col_chart, col_pie_3m = st.columns([2, 1])

with col_chart:
    fig_proj = go.Figure()
    
    # Area Chart
    fig_proj.add_trace(go.Scatter(
        x=x_dates, y=y_values,
        mode='lines+markers+text',
        fill='tozeroy',
        line=dict(color='#667eea', width=4),
        marker=dict(size=10, color='#764ba2', line=dict(width=2, color='white')),
        text=[f"${v:,.0f}" for v in y_values],
        textposition="top center",
        name='Portföy Değeri'
    ))
    
    fig_proj.update_layout(
        title=f"Tahmini Büyüme Eğrisi (1 Ay - 3 Ay)",
        font=dict(family="Space Mono", color="#ccd6f6"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        xaxis=dict(
            showgrid=False, 
            type="date",
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1H", step="day", stepmode="backward"),
                    dict(count=1, label="1A", step="month", stepmode="backward"),
                    dict(count=3, label="3A", step="month", stepmode="backward"),
                    dict(step="all", label="Tümü")
                ]),
                bgcolor="#1C2432",
                activecolor="#667eea",
                font=dict(color="white")
            )
        ),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickprefix="$"),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
        hovermode="x unified"
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # 3-Stage Progress Metrics
    st.markdown('<div style="font-family:\'Space Mono\';font-size:0.85rem;color:#8892b0;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Tahmini Büyüme (3 Aşamalı)</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    
    with m1:
        d_cls = "pf-metric-delta-up" if growth_1w >= 0 else "pf-metric-delta-down"
        st.markdown(f"""
        <div class="pf-metric-card" style="padding:15px;">
            <div class="pf-metric-label">Kısa Vade (1 Hafta)</div>
            <div class="pf-metric-value" style="font-size:1.4rem">${proj_1w:,.2f}</div>
            <div class="{d_cls}">{growth_1w:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        d_cls = "pf-metric-delta-up" if growth_1m >= 0 else "pf-metric-delta-down"
        st.markdown(f"""
        <div class="pf-metric-card" style="padding:15px;">
            <div class="pf-metric-label">Orta Vade (1 Ay)</div>
            <div class="pf-metric-value" style="font-size:1.4rem">${proj_1m:,.2f}</div>
            <div class="{d_cls}">{growth_1m:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        d_cls = "pf-metric-delta-up" if growth_3m >= 0 else "pf-metric-delta-down"
        st.markdown(f"""
        <div class="pf-metric-card" style="padding:15px;">
            <div class="pf-metric-label">Uzun Vade (3 Ay)</div>
            <div class="pf-metric-value" style="font-size:1.4rem">${proj_3m:,.2f}</div>
            <div class="{d_cls}">{growth_3m:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

with col_pie_3m:
    st.markdown('<div style="text-align:center;font-size:0.9rem;color:#8892b0;margin-bottom:10px;">3 Ay Sonraki Hedef Dağılım</div>', unsafe_allow_html=True)

    labels_3m = [x["ticker"] for x in asset_proj_3m]
    values_3m = [x["value_3m"] for x in asset_proj_3m]

    fig_pie_3m = px.pie(
        names=labels_3m, values=values_3m,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.5
    )
    fig_pie_3m.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color="#8892b0"),
        showlegend=False
    )
    fig_pie_3m.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie_3m, use_container_width=True)

# Ağırlık Pie Chart + Tablo
col_pie, col_table = st.columns([1, 2])

with col_pie:
    labels = [r["ticker"] for r in rows_summary]
    values = [r["value"] for r in rows_summary]
    fig_pie = px.pie(
        names=labels, values=values,
        title="Portföy Ağırlıkları",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#8892b0"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    # Detay tablosu (HTML)
    table_html = '<table class="pf-table"><thead><tr>'
    headers = ["Hisse", "Adet", "Alış", "Güncel", "Değer", "K/Z", "Temettü", "Yıl. Temettü"]
    for h in headers:
        table_html += f"<th>{h}</th>"
    table_html += "</tr></thead><tbody>"

    for r in rows_summary:
        pnl_color = "#64ffda" if r["pnl"] >= 0 else "#ff6b6b"
        div_badge = '<span class="pf-div-yes">✓ Evet</span>' if r["pays_div"] else '<span class="pf-div-no">Hayır</span>'
        table_html += f"""<tr>
            <td><strong>{r['ticker']}</strong><br><span style="font-size:0.7rem;color:#8892b0">{r['name']}</span></td>
            <td>{r['qty']}</td>
            <td>${r['buy_price']:.2f}</td>
            <td>${r['current_price']:.2f}</td>
            <td>${r['value']:,.2f}<br><span style="font-size:0.7rem;color:#8892b0">{r['value']/total_value*100:.1f}%</span></td>
            <td style="color:{pnl_color}">${r['pnl']:+,.2f}<br>{r['pnl_pct']:+.1f}%</td>
            <td>{div_badge}<br><span style="font-size:0.7rem;color:#8892b0">{r['div_yield']:.2f}%</span></td>
            <td>${r['est_annual_div_income']:,.2f}</td>
        </tr>"""

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)


# =========================================================
# SHORT MODEL BÖLÜMÜ — Candlestick + Tahminler
# =========================================================
st.markdown('<div class="pf-section-header">⚡ Short Model — Kısa Vadeli Tahminler (1-7 İş Günü)</div>', unsafe_allow_html=True)
st.caption("Veri: Kaggle CSV | Model: LightGBM + XGBoost + LogReg | Candlestick grafik")

short_results = {}
for t in tickers_in_portfolio:
    raw_df = load_kaggle_ticker(t)
    if raw_df is None:
        short_results[t] = {"error": "Kaggle verisi bulunamadı"}
        continue

    try:
        full_df = build_features(raw_df)
        feat_cols = get_feature_cols(full_df)
        latest = full_df[feat_cols].dropna().iloc[[-1]]
        if latest.empty:
            short_results[t] = {"error": "Yeterli veri yok"}
            continue

        X = latest.values.astype(np.float32)
        horizons = {}

        for h in HORIZONS:
            algo_results = {}
            for algo in ["lgbm", "xgb", "logreg"]:
                bundle = load_short_model(t, h, algo)
                if bundle is None:
                    continue
                model = bundle["model"]
                scaler = bundle.get("scaler")
                model_feats = bundle.get("feat_cols", feat_cols)
                X_pred = full_df[model_feats].dropna().iloc[[-1]].values.astype(np.float32)

                try:
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

            # Majority vote
            if algo_results:
                up = sum(1 for v in algo_results.values() if v["label"] == "UP")
                down = len(algo_results) - up
                avg_prob = np.mean([v["prob"] for v in algo_results.values()])
                horizons[f"{h}d"] = {
                    "vote": "UP" if up > down else "DOWN",
                    "up_count": up, "down_count": down,
                    "avg_prob": round(avg_prob, 4),
                    "algos": algo_results
                }

        short_results[t] = {"horizons": horizons, "df": raw_df}
    except Exception as e:
        short_results[t] = {"error": str(e)}

# Short model sonuçları göster
for t in tickers_in_portfolio:
    res = short_results.get(t, {})
    if "error" in res:
        st.warning(f"**{t}**: {res['error']}")
        continue

    with st.expander(f"⚡ {t} — {TICKER_TO_NAME.get(t, t)}", expanded=(len(tickers_in_portfolio) <= 3)):
        # Candlestick grafik
        raw_df = res.get("df")
        if raw_df is not None and len(raw_df) > 0:
            hist = raw_df.tail(90).copy()

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"],
                increasing_line_color="#2ECC71",
                increasing_fillcolor="rgba(46,204,113,0.75)",
                decreasing_line_color="#E74C3C",
                decreasing_fillcolor="rgba(231,76,60,0.75)",
                name="Fiyat", whiskerwidth=0.5
            ))

            # EMA
            if len(hist) >= 20:
                hist["ema20"] = hist["Close"].ewm(span=20).mean()
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist["ema20"],
                    line=dict(color="rgba(255,176,0,0.8)", width=1.2, dash="dot"),
                    name="EMA20"
                ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,10,13,0.8)",
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_rangeslider_visible=False,
                font=dict(family="Space Mono", color="#8B93A5"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Horizon tahminleri
        horizons = res.get("horizons", {})
        if horizons:
            cols = st.columns(len(horizons))
            for i, (h_key, h_data) in enumerate(horizons.items()):
                with cols[i]:
                    vote = h_data["vote"]
                    badge = signal_badge_long(vote)
                    st.markdown(f"""
                    <div class="pf-metric-card" style="padding:1rem;">
                        <div class="pf-metric-label">{h_key} Horizon</div>
                        <div style="margin:6px 0">{badge}</div>
                        <div style="font-size:0.75rem;color:#8892b0">
                            {h_data['up_count']}↗ / {h_data['down_count']}↘ | Prob: {h_data['avg_prob']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Short model bulunamadı.")


# =========================================================
# MID MODEL BÖLÜMÜ — Alpaca API + sector_optimized_params.json
# =========================================================
st.markdown('<div class="pf-section-header">📈 Mid Model — Orta Vadeli Tahminler (1-3 Ay)</div>', unsafe_allow_html=True)
st.caption("Veri: Alpaca API | Model: RandomForest + LightGBM Regressor | Fiyat tahmini")

report = load_sector_json()

if report is None:
    st.warning("sector_optimized_params.json bulunamadı. `multi_sektor_analiz/main_ibrahim.py` çalıştırarak rapor oluşturun.")
else:
    mid_predictions = {}
    for t in tickers_in_portfolio:
        pred, sector = get_mid_prediction(report, t)
        if pred:
            mid_predictions[t] = {"pred": pred, "sector": sector}

    if not mid_predictions:
        st.info("Portföydeki hisseler için mid model tahmini bulunamadı. (Yalnızca multi_sektor_analiz'daki hisseler desteklenir)")
    else:
        # Metric cards
        mid_cols = st.columns(len(mid_predictions))
        for i, (t, data) in enumerate(mid_predictions.items()):
            pred = data["pred"]
            with mid_cols[i] if len(mid_predictions) > 1 else st.container():
                d1m = pred.get("getiri_1ay_pct", 0)
                d3m = pred.get("getiri_3ay_pct", 0)
                d1_cls = "pf-metric-delta-up" if d1m > 0 else "pf-metric-delta-down"
                d3_cls = "pf-metric-delta-up" if d3m > 0 else "pf-metric-delta-down"

                st.markdown(f"""
                <div class="pf-metric-card" style="margin-bottom:12px;">
                    <div class="pf-metric-label">{t} — {data['sector']}</div>
                    <div style="display:flex;justify-content:space-around;margin-top:8px;">
                        <div>
                            <div style="font-size:0.7rem;color:#8892b0;">1 Ay Tahmin</div>
                            <div class="pf-metric-value" style="font-size:1.3rem;">${pred.get('tahmin_1ay', 0):.2f}</div>
                            <div class="{d1_cls}">{d1m:+.2f}%</div>
                            <div style="margin-top:4px;">{signal_badge_mid(pred.get('sinyal_1ay', 'BEKLE'))}</div>
                        </div>
                        <div>
                            <div style="font-size:0.7rem;color:#8892b0;">3 Ay Tahmin</div>
                            <div class="pf-metric-value" style="font-size:1.3rem;">${pred.get('tahmin_3ay', 0):.2f}</div>
                            <div class="{d3_cls}">{d3m:+.2f}%</div>
                            <div style="margin-top:4px;">{signal_badge_mid(pred.get('sinyal_3ay', 'BEKLE'))}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Portföy Mid Model Projeksiyonu
        projected_1m = 0
        projected_3m = 0
        for t, data in mid_predictions.items():
            if t in portfolio:
                qty = portfolio[t]["quantity"]
                p1 = data["pred"].get("tahmin_1ay", current_prices.get(t, 0))
                p3 = data["pred"].get("tahmin_3ay", current_prices.get(t, 0))
                projected_1m += qty * p1
                projected_3m += qty * p3

        if projected_1m > 0:
            st.markdown(f"""
            <div class="pf-metric-card" style="margin-top:12px;text-align:center;">
                <div class="pf-metric-label">Portföy Mid Model Projeksiyonu (sadece tahmin bulunan hisseler)</div>
                <div style="display:flex;justify-content:center;gap:60px;margin-top:8px;">
                    <div>
                        <div style="font-size:0.8rem;color:#8892b0;">1 Ay Sonra</div>
                        <div class="pf-metric-value" style="font-size:1.5rem;">${projected_1m:,.2f}</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#8892b0;">3 Ay Sonra</div>
                        <div class="pf-metric-value" style="font-size:1.5rem;">${projected_3m:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# LONG MODEL BÖLÜMÜ — Kaggle CSV + long_model.pkl
# =========================================================
st.markdown('<div class="pf-section-header">🔮 Long Model — Uzun Vadeli Tahminler (22-63 İş Günü)</div>', unsafe_allow_html=True)
st.caption("Veri: Kaggle CSV | Model: LGBMClassifier 3-sınıf (UP / NEUTRAL / DOWN)")

if not LONG_MODEL_PATH.exists():
    st.warning("long_model.pkl bulunamadı. `python src/model_long.py` çalıştırarak modeli eğitin.")
else:
    try:
        import joblib
        payload = joblib.load(LONG_MODEL_PATH)
        long_model = payload["model"]
        long_feature_cols = payload["feature_cols"]

        # Long model verisini hazırla
        tickers_for_long = [t for t in tickers_in_portfolio if t in LONG_UNIVERSE]

        if not tickers_for_long:
            st.info("Portföydeki hisseler long model evreninde (UNIVERSE) bulunmuyor.")
        else:
            # all_stocks.csv cache'ten long model verisini hazırla
            dfs_long = []
            for t in tickers_for_long:
                ticker_df = load_kaggle_ticker(t)
                if ticker_df is not None:
                    df_tmp = ticker_df.tail(400).reset_index()
                    df_tmp = df_tmp.rename(columns={"Date": "datetime", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    df_tmp["ticker"] = t
                    dfs_long.append(df_tmp)

            if dfs_long:
                df_long = pd.concat(dfs_long)
                df_long["datetime"] = pd.to_datetime(df_long["datetime"])
                df_long = df_long.sort_values(["ticker", "datetime"])

                df_long = add_long_features(df_long, normalize=True)

                last_date = df_long["datetime"].max()
                latest_long = df_long[df_long["datetime"] == last_date].copy()

                if not latest_long.empty:
                    missing_feats = [c for c in long_feature_cols if c not in latest_long.columns]
                    if missing_feats:
                        st.warning(f"Eksik feature'lar: {missing_feats}")
                    else:
                        proba = long_model.predict_proba(latest_long[long_feature_cols])
                        latest_long["prob_down"] = proba[:, 0]
                        latest_long["prob_neutral"] = proba[:, 1]
                        latest_long["prob_up"] = proba[:, 2]
                        latest_long["pred_class"] = long_model.predict(latest_long[long_feature_cols])
                        latest_long["pred_label"] = latest_long["pred_class"].map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})

                        long_cols = st.columns(min(len(tickers_for_long), 4))
                        for i, t in enumerate(tickers_for_long):
                            row = latest_long[latest_long["ticker"] == t]
                            if row.empty:
                                continue
                            row = row.iloc[0]
                            col_idx = i % len(long_cols)

                            with long_cols[col_idx]:
                                label = row["pred_label"]
                                p_up = row["prob_up"] * 100
                                p_ne = row["prob_neutral"] * 100
                                p_dn = row["prob_down"] * 100

                                st.markdown(f"""
                                <div class="pf-metric-card" style="margin-bottom:12px;padding:1.2rem;">
                                    <div class="pf-metric-label">{t}</div>
                                    <div style="margin:8px 0;">{signal_badge_long(label)}</div>
                                    <div style="font-size:0.75rem;color:#8892b0;margin-top:8px;">
                                        UP: {p_up:.0f}% · NEU: {p_ne:.0f}% · DN: {p_dn:.0f}%
                                    </div>
                                    <div style="margin-top:6px;height:4px;background:#1C2432;border-radius:2px;overflow:hidden;">
                                        <div style="height:100%;width:{p_up:.0f}%;background:#2ECC71;border-radius:2px;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("Long model için güncel veri bulunamadı.")
            else:
                st.info("Portföy hisseleri için veri bulunamadı. 'data/all_stocks.csv' dosyasını kontrol edin.")
    except Exception as e:
        st.error(f"Long model hatası: {e}")


# =========================================================
# BİRLEŞİK TABLO — 3 Model Yan Yana
# =========================================================
st.markdown('<div class="pf-section-header">🎯 Birleşik Model Sinyalleri</div>', unsafe_allow_html=True)

combined_rows = []
for t in tickers_in_portfolio:
    row = {"Hisse": t, "Ad": TICKER_TO_NAME.get(t, t)}

    # Short
    sr = short_results.get(t, {})
    if "horizons" in sr:
        for h_key, h_data in sr["horizons"].items():
            row[f"Short {h_key}"] = h_data["vote"]
    else:
        for h in HORIZONS:
            row[f"Short {h}d"] = "—"

    # Mid
    if report:
        pred, _ = get_mid_prediction(report, t)
        if pred:
            row["Mid 1M"] = pred.get("sinyal_1ay", "—")
            row["Mid 3M"] = pred.get("sinyal_3ay", "—")
        else:
            row["Mid 1M"] = "—"
            row["Mid 3M"] = "—"
    else:
        row["Mid 1M"] = "—"
        row["Mid 3M"] = "—"

    # Long
    if LONG_MODEL_PATH.exists() and "latest_long" in dir() and latest_long is not None:
        lr = latest_long[latest_long["ticker"] == t] if not latest_long.empty else pd.DataFrame()
        if not lr.empty:
            row["Long"] = lr.iloc[0].get("pred_label", "—")
        else:
            row["Long"] = "—"
    else:
        row["Long"] = "—"

    combined_rows.append(row)

if combined_rows:
    df_combined = pd.DataFrame(combined_rows)

    # HTML tablo
    table_html = '<table class="pf-table"><thead><tr>'
    for col in df_combined.columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr></thead><tbody>"

    for _, row in df_combined.iterrows():
        table_html += "<tr>"
        for col in df_combined.columns:
            val = str(row[col])
            if val == "UP" or val == "AL":
                cell = f'<td style="color:#2ECC71;font-weight:600">▲ {val}</td>'
            elif val == "DOWN" or val == "SAT":
                cell = f'<td style="color:#E74C3C;font-weight:600">▼ {val}</td>'
            elif val == "BEKLE" or val == "NEUTRAL":
                cell = f'<td style="color:#f39c12">● {val}</td>'
            else:
                cell = f"<td>{val}</td>"
            table_html += cell
        table_html += "</tr>"

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#8892b0;font-size:0.75rem;padding:2rem 0;">'
    "⚠ Model çıktıları yatırım tavsiyesi değildir. Eğitim/araştırma amaçlıdır."
    "</div>",
    unsafe_allow_html=True
)
