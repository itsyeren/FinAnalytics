"""
app/pages/long.py — Trend Hunter AI · Long-Term Model Dashboard
================================================================
3-Class Ranking Model: UP / NEUTRAL / DOWN
22–63 Day Horizon · render_long_dashboard(selected_ticker) olarak App.py'den çağrılır.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import sys
import os
from pathlib import Path


# =========================================================
# ROOT PATH FINDER
# =========================================================
def _find_project_root():
    current = Path(__file__).resolve()
    for _ in range(6):
        if (current / "src").exists() and (current / "models").exists():
            return current
        current = current.parent
    return None


_ROOT_DIR = _find_project_root()
if _ROOT_DIR and str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

# =========================================================
# IMPORTS
# =========================================================
try:
    from src.config import UNIVERSE
    from src.features import add_features
    _HAS_IMPORTS = True
except ImportError:
    _HAS_IMPORTS = False
    UNIVERSE = []

_MODEL_PATH = _ROOT_DIR / "models/long_model.pkl" if _ROOT_DIR else Path(".")
_ALL_STOCKS_PATH = _ROOT_DIR / "data/all_stocks.csv" if _ROOT_DIR else Path(".")


# =========================================================
# CSS — Scoped for long dashboard
# =========================================================
_LONG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

.long-header-block { padding: 32px 0 24px 0; position: relative; }
.long-ticker-label {
    font-family: 'Space Mono', monospace; font-size: 11px;
    letter-spacing: 3px; color: #FFB000 !important;
    text-transform: uppercase; margin-bottom: 6px;
}
.long-big-price {
    font-family: 'Space Mono', monospace; font-size: 64px; font-weight: 700;
    color: #E8EAF0 !important; letter-spacing: -2px; line-height: 1;
    text-shadow: 0 0 40px rgba(255,176,0,0.12);
}
.long-price-sub {
    font-size: 12px; font-family: 'Space Mono', monospace;
    color: #4A5568 !important; letter-spacing: 2px;
    margin-top: 8px; text-transform: uppercase;
}
.long-section-divider {
    height: 1px; background: linear-gradient(90deg, #FFB000, transparent);
    margin: 24px 0; opacity: 0.4;
}

.long-metric-card {
    background: #111820; padding: 20px 22px; border-radius: 10px;
    border: 1px solid #1C2432; position: relative; overflow: hidden;
    transition: border-color 0.25s, background 0.25s, transform 0.2s; cursor: default;
}
.long-metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #FFB000, transparent); opacity: 0;
    transition: opacity 0.25s;
}
.long-metric-card:hover { border-color: #FFB000; background: #161E28; transform: translateY(-2px); }
.long-metric-card:hover::before { opacity: 1; }

.long-card-label {
    font-family: 'Space Mono', monospace; font-size: 9px;
    letter-spacing: 2.5px; color: #4A5568 !important;
    text-transform: uppercase; margin-bottom: 10px;
}
.long-card-value {
    font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700;
    color: #E8EAF0 !important; line-height: 1.2; margin-bottom: 6px;
}
.long-card-sub {
    font-size: 11px; color: #4A5568 !important;
    font-family: 'Space Mono', monospace; line-height: 1.5;
}

.long-signal-badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-family: 'Space Mono', monospace; font-size: 11px;
    font-weight: 700; letter-spacing: 2px; margin-bottom: 8px;
}
.long-signal-up      { background: rgba(46,204,113,0.12); color: #2ECC71 !important; border: 1px solid rgba(46,204,113,0.3); }
.long-signal-down    { background: rgba(231,76,60,0.12);  color: #E74C3C !important; border: 1px solid rgba(231,76,60,0.3); }
.long-signal-neutral { background: rgba(139,147,165,0.1); color: #8B93A5 !important; border: 1px solid rgba(139,147,165,0.2); }

.long-prob-row { display: flex; align-items: center; gap: 7px; margin-top: 6px; }
.long-prob-label {
    font-family: 'Space Mono', monospace; font-size: 9px;
    letter-spacing: 1px; width: 20px; color: #4A5568 !important;
    text-transform: uppercase; flex-shrink: 0;
}
.long-prob-track { flex: 1; height: 4px; background: #1C2432; border-radius: 2px; overflow: hidden; }
.long-prob-fill-up      { height: 100%; border-radius: 2px; background: #2ECC71; }
.long-prob-fill-neutral { height: 100%; border-radius: 2px; background: #4A5568; }
.long-prob-fill-down    { height: 100%; border-radius: 2px; background: #E74C3C; }
.long-prob-pct {
    font-family: 'Space Mono', monospace; font-size: 9px;
    color: #4A5568 !important; width: 28px; text-align: right; flex-shrink: 0;
}

.long-section-header {
    font-family: 'Space Mono', monospace; font-size: 10px;
    letter-spacing: 3px; text-transform: uppercase; color: #4A5568 !important;
    margin: 32px 0 16px 0; display: flex; align-items: center; gap: 10px;
}
.long-section-header::after { content: ''; flex: 1; height: 1px; background: #1C2432; }

.long-rank-table { width: 100%; border-collapse: collapse; font-family: 'Space Mono', monospace; font-size: 12px; }
.long-rank-table th {
    color: #4A5568 !important; font-size: 9px; letter-spacing: 2px;
    text-transform: uppercase; padding: 10px 14px; text-align: left;
    border-bottom: 1px solid #1C2432;
}
.long-rank-table td {
    padding: 10px 14px; color: #8B93A5 !important;
    border-bottom: 1px solid rgba(28, 36, 50, 0.5);
}
.long-rank-table tr:hover td { background: rgba(255, 176, 0, 0.08); color: #E8EAF0 !important; }
.long-rank-num { color: #4A5568 !important; font-size: 10px; }
.long-rank-ticker { color: #FFB000 !important; font-weight: 700; letter-spacing: 1px; }
.long-rank-score-pill {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    background: rgba(255,176,0,0.1); color: #FFB000 !important; font-size: 11px;
}
</style>
"""


# =========================================================
# DATA + PREDICTION
# =========================================================
@st.cache_data(ttl=3600)
def _load_data_and_predict():
    if not _HAS_IMPORTS:
        return None, None, "src.config veya src.features import edilemedi"

    if not _MODEL_PATH.exists():
        return None, None, "Model not found"

    payload = joblib.load(_MODEL_PATH)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    if not _ALL_STOCKS_PATH.exists():
        return None, None, f"Data file not found: {_ALL_STOCKS_PATH}"

    try:
        df_all = pd.read_csv(_ALL_STOCKS_PATH)
        cols_map = {
            "Ticker": "ticker", "Date": "datetime",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        }
        df_all = df_all.rename(columns=cols_map)
        df_all["datetime"] = pd.to_datetime(df_all["datetime"])

        dfs = []
        for t in UNIVERSE:
            mask = df_all["ticker"].isin([t, f"{t}.US"])
            if mask.any():
                temp = df_all[mask].copy()
                temp["ticker"] = t
                dfs.append(temp.tail(400))

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
    proba_matrix = model.predict_proba(latest[feature_cols])
    latest["prob_down"] = proba_matrix[:, 0]
    latest["prob_neutral"] = proba_matrix[:, 1]
    latest["prob_up"] = proba_matrix[:, 2]

    latest["pred_class"] = model.predict(latest[feature_cols])
    latest["pred_label"] = latest["pred_class"].map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})

    up_p = proba_matrix[:, 2]
    min_p, max_p = up_p.min(), up_p.max()
    latest["model_score"] = ((up_p - min_p) / (max_p - min_p + 1e-9)) * 100
    latest["percentile_rank"] = latest["model_score"].rank(pct=True) * 100

    return latest.sort_values("model_score", ascending=False), df, None


# =========================================================
# ANA RENDER FONKSİYONU
# =========================================================
def render_long_dashboard(selected_ticker: str) -> None:
    """App.py'den çağrılır. Uzun vadeli model sonuçlarını gösterir."""

    st.markdown(_LONG_CSS, unsafe_allow_html=True)

    if _ROOT_DIR is None:
        st.error("Proje kök dizini bulunamadı.")
        return

    results, history_df, error = _load_data_and_predict()

    if error:
        st.error(error)
        return

    if results is None or results.empty:
        st.warning("Model sonuçları yüklenemedi.")
        return

    # ── Top 5 Leaderboard
    st.markdown('<div class="long-section-header">Top Ranked</div>', unsafe_allow_html=True)
    top5 = results.head(5)

    header_html = (
        '<table class="long-rank-table">'
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
        score = f"{r['model_score']:.0f}"
        label = str(r.get('pred_label', 'UP'))
        badge_cls = {"UP": "long-signal-up", "DOWN": "long-signal-down",
                     "NEUTRAL": "long-signal-neutral"}.get(label, "long-signal-neutral")
        body_parts.append(
            '<tr>'
            + f'<td class="long-rank-num">#{i+1}</td>'
            + f'<td class="long-rank-ticker">{ticker}</td>'
            + f'<td><span class="long-signal-badge {badge_cls}" style="font-size:9px;padding:1px 6px;">{label}</span></td>'
            + f'<td><span class="long-rank-score-pill">{score}</span></td>'
            + '</tr>'
        )

    full_table = header_html + "".join(body_parts) + "</tbody></table>"
    st.markdown(full_table, unsafe_allow_html=True)

    # ── Seçili ticker kontrolü
    available_tickers = results["ticker"].tolist()
    if selected_ticker not in available_tickers:
        st.warning(f"**{selected_ticker}** uzun vadeli modelde bulunamadı.")
        return

    sel_ticker = selected_ticker

    row = results[results["ticker"] == sel_ticker].iloc[0]
    price = row["close"]
    model_score = row["model_score"]
    percentile = row["percentile_rank"]
    mom = row["mom_126"]
    slope = row["sma50_slope20"]
    dist = row["dist_sma_200"]
    pred_label = row["pred_label"]
    prob_up = row["prob_up"]
    prob_neutral = row["prob_neutral"]
    prob_down = row["prob_down"]

    # ── PRICE HEADER
    st.markdown(f"""
    <div class="long-header-block">
        <div class="long-ticker-label">⚡ {sel_ticker}.US</div>
        <div class="long-big-price">${price:,.2f}</div>
        <div class="long-price-sub">Last Close &nbsp;·&nbsp; Daily Chart</div>
    </div>
    <div class="long-section-divider"></div>
    """, unsafe_allow_html=True)

    # ── METRIC CARDS
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        score_pct = f"{model_score:.0f}"
        perc_str = f"{percentile:.0f}"
        badge_cls = {"UP": "long-signal-up", "DOWN": "long-signal-down",
                     "NEUTRAL": "long-signal-neutral"}.get(pred_label, "long-signal-neutral")

        up_w = f"{prob_up * 100:.0f}%"
        ne_w = f"{prob_neutral * 100:.0f}%"
        dn_w = f"{prob_down * 100:.0f}%"

        card1_html = (
            '<div class="long-metric-card">'
            '<div class="long-card-label">Model Signal</div>'
            f'<span class="long-signal-badge {badge_cls}">{pred_label}</span>'
            f'<div class="long-card-sub">Score {score_pct}/100 &nbsp;·&nbsp; P{perc_str}</div>'
            # UP bar
            '<div class="long-prob-row">'
            '<span class="long-prob-label">UP</span>'
            '<div class="long-prob-track"><div class="long-prob-fill-up" style="width:' + up_w + '"></div></div>'
            '<span class="long-prob-pct">' + up_w + '</span>'
            '</div>'
            # NEUTRAL bar
            '<div class="long-prob-row">'
            '<span class="long-prob-label">NEU</span>'
            '<div class="long-prob-track"><div class="long-prob-fill-neutral" style="width:' + ne_w + '"></div></div>'
            '<span class="long-prob-pct">' + ne_w + '</span>'
            '</div>'
            # DOWN bar
            '<div class="long-prob-row">'
            '<span class="long-prob-label">DN</span>'
            '<div class="long-prob-track"><div class="long-prob-fill-down" style="width:' + dn_w + '"></div></div>'
            '<span class="long-prob-pct">' + dn_w + '</span>'
            '</div>'
            '</div>'
        )
        st.markdown(card1_html, unsafe_allow_html=True)

    with c2:
        mom_color = "#2ECC71" if mom > 0 else "#E74C3C"
        mom_val = ("+" if mom > 0 else "") + f"{mom:.2f}"
        st.markdown(
            '<div class="long-metric-card">'
            '<div class="long-card-label">Momentum 6M</div>'
            f'<div class="long-card-value" style="color:{mom_color} !important;">{mom_val}</div>'
            '<div class="long-card-sub">126-day return (normalized)</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c3:
        slope_color = "#2ECC71" if slope > 0 else "#E74C3C"
        slope_val = ("+" if slope > 0 else "") + f"{slope:.3f}"
        st.markdown(
            '<div class="long-metric-card">'
            '<div class="long-card-label">Trend Slope</div>'
            f'<div class="long-card-value" style="color:{slope_color} !important;">{slope_val}</div>'
            '<div class="long-card-sub">SMA50 slope (20-day window)</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c4:
        dist_color = "#2ECC71" if dist > 0 else "#E74C3C"
        dist_val = ("+" if dist > 0 else "") + f"{dist:.2f}"
        st.markdown(
            '<div class="long-metric-card">'
            '<div class="long-card-label">Dist. to SMA200</div>'
            f'<div class="long-card-value" style="color:{dist_color} !important;">{dist_val}</div>'
            '<div class="long-card-sub">Relative positioning</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── PRICE CHART
    st.markdown('<div class="long-section-header" style="margin-top:36px;">Price Action · 3 Month</div>',
                unsafe_allow_html=True)

    if history_df is not None:
        hist = history_df[history_df["ticker"] == sel_ticker].tail(120).copy()

        if not hist.empty:
            hist["ema20"] = hist["close"].ewm(span=20).mean()
            hist["ema50"] = hist["close"].ewm(span=50).mean()
            hist["sma200"] = hist["close"].rolling(200).mean()

            fig = go.Figure()

            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=hist["datetime"],
                open=hist["open"], high=hist["high"],
                low=hist["low"], close=hist["close"],
                increasing_line_color="#2ECC71",
                increasing_fillcolor="rgba(46,204,113,0.75)",
                decreasing_line_color="#E74C3C",
                decreasing_fillcolor="rgba(231,76,60,0.75)",
                name="Price", whiskerwidth=0.5
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
                x=hist["datetime"], y=hist["volume"],
                yaxis="y2", marker_color=vol_colors,
                name="Volume", showlegend=False
            ))

            fig.update_layout(
                plot_bgcolor="#080A0D", paper_bgcolor="#080A0D",
                font=dict(color="#8B93A5", family="Space Mono, monospace", size=11),
                height=580, margin=dict(l=10, r=60, t=20, b=20),
                xaxis_rangeslider_visible=False, hovermode="x unified",
                legend=dict(
                    bgcolor="rgba(13,17,23,0.8)", bordercolor="#1C2432", borderwidth=1,
                    font=dict(size=10, color="#8B93A5"),
                    orientation="h", y=1.04, x=0
                ),
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(28,36,50,0.5)", gridwidth=1,
                    zeroline=False, tickfont=dict(size=10, color="#4A5568"),
                    tickformat="%b %d",
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(28,36,50,0.5)", gridwidth=1,
                    zeroline=False, side="right",
                    tickfont=dict(size=10, color="#4A5568"), tickprefix="$",
                ),
                yaxis2=dict(overlaying="y", side="left", showgrid=False, showticklabels=False),
                hoverlabel=dict(
                    bgcolor="#111820", bordercolor="#1C2432",
                    font=dict(color="#E8EAF0", size=11, family="Space Mono, monospace"),
                ),
            )

            # Amber top accent
            fig.add_shape(
                type="line", xref="paper", yref="paper",
                x0=0, y0=1, x1=0.15, y1=1,
                line=dict(color="#FFB000", width=2),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"{sel_ticker} için geçmiş fiyat verisi bulunamadı.")
    else:
        st.warning("Fiyat verisi yüklenemedi.")
