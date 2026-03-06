"""
src/long.py — Uzun Vadeli Model Dashboard (Trend Hunter AI)
============================================================
3-Sınıflı Sıralama Modeli: YÜKSELİŞ / NÖTR / DÜŞÜŞ
22–63 Gün Ufku · Analysis.py'den render_long_dashboard(selected_ticker) olarak çağrılır.
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
# PROJE KÖK BULMA
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
# IMPORT
# =========================================================
try:
    from src.config import UNIVERSE
    from src.features import add_features
    _HAS_IMPORTS = True
except ImportError:
    _HAS_IMPORTS = False
    UNIVERSE = []

try:
    # TICKER_TO_NAME haritasını models/kisa_vadeli/kaynak/config.py'den yükle
    import importlib.util as _ilu
    _cfg_path = _ROOT_DIR / "models" / "short_term" / "src" / "config.py" if _ROOT_DIR else None
    if _cfg_path and _cfg_path.exists():
        _spec = _ilu.spec_from_file_location("_short_config", _cfg_path)
        _short_cfg = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_short_cfg)
        _TICKER_TO_NAME: dict = getattr(_short_cfg, "TICKER_TO_NAME", {})
    else:
        _TICKER_TO_NAME = {}
except Exception:
    _TICKER_TO_NAME = {}

_MODEL_PATH      = _ROOT_DIR / "models" / "long_term" / "long_model.pkl" if _ROOT_DIR else Path(".")
_ALL_STOCKS_PATH = _ROOT_DIR / "data/all_stocks.csv" if _ROOT_DIR else Path(".")


# =========================================================
# CSS — Uzun vadeli dashboard'a özel
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
    font-family: 'Space Mono', monospace; font-size: 32px; font-weight: 700;
    color: #E8EAF0 !important; line-height: 1.15; margin-bottom: 6px;
    letter-spacing: -0.5px;
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
    color: #FFB000 !important; font-size: 9px; letter-spacing: 2px;
    text-transform: uppercase; padding: 10px 16px; text-align: left;
    border-bottom: 1px solid rgba(255,176,0,0.2);
    background: rgba(255,176,0,0.04);
}
.long-rank-table td {
    padding: 11px 16px; color: #8B93A5 !important;
    border-bottom: 1px solid rgba(28,36,50,0.6);
    vertical-align: middle;
}
.long-rank-table tr:hover td { background: rgba(255,176,0,0.06); color: #E8EAF0 !important; }
.long-rank-num   { color: #4A5568 !important; font-size: 10px; }
.long-rank-ticker { color: #FFB000 !important; font-weight: 700; letter-spacing: 1px; font-size: 13px; }
.long-rank-score-pill {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    background: rgba(255,176,0,0.1); color: #FFB000 !important;
    font-size: 11px; border: 1px solid rgba(255,176,0,0.2);
}
.long-rank-bar-wrap { width: 80px; height: 5px; background: #1C2432; border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 8px; }
.long-rank-bar-fill { height: 100%; border-radius: 3px; }

/* Score card ızgarası — eşit yükseklik */
.long-card-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.long-metric-card {
    background: #111820; padding: 20px 22px; border-radius: 10px;
    border: 1px solid #1C2432; position: relative; overflow: hidden;
    transition: border-color 0.25s, background 0.25s, transform 0.2s;
    cursor: default; min-height: 180px;
    display: flex; flex-direction: column; justify-content: flex-start;
}
</style>
"""


# =========================================================
# VERİ + TAHMİN
# =========================================================
@st.cache_data(ttl=3600)
def _load_data_and_predict():
    if not _HAS_IMPORTS:
        return None, None, "src.config veya src.features import edilemedi"

    if not _MODEL_PATH.exists():
        return None, None, (
            "Model dosyası bulunamadı: models/long_term/long_model.pkl\n"
            "Eğitim için: python src/model_long.py"
        )

    payload = joblib.load(_MODEL_PATH)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    if not _ALL_STOCKS_PATH.exists():
        return None, None, f"Veri dosyası bulunamadı: {_ALL_STOCKS_PATH}"

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
            return None, None, "all_stocks.csv'de eşleşen veri bulunamadı"

        df = pd.concat(dfs)
        df = df.sort_values(["ticker", "datetime"])

    except Exception as e:
        return None, None, f"Veri yükleme hatası: {e}"

    df = add_features(df, normalize=True)

    last_date = df["datetime"].max()
    latest = df[df["datetime"] == last_date].copy()

    if latest.empty:
        return None, None, "Son tarih verisi boş"

    # 3-sınıflı model: 0=Düşüş, 1=Nötr, 2=Yükseliş
    proba_matrix = model.predict_proba(latest[feature_cols])
    latest["prob_down"] = proba_matrix[:, 0]
    latest["prob_neutral"] = proba_matrix[:, 1]
    latest["prob_up"] = proba_matrix[:, 2]

    latest["pred_class"] = model.predict(latest[feature_cols])
    latest["pred_label"] = latest["pred_class"].map({0: "DÜŞÜŞ", 1: "NÖTR", 2: "YÜKSELİŞ"})

    up_p = proba_matrix[:, 2]
    min_p, max_p = up_p.min(), up_p.max()
    latest["model_score"] = ((up_p - min_p) / (max_p - min_p + 1e-9)) * 100
    latest["percentile_rank"] = latest["model_score"].rank(pct=True) * 100

    return latest.sort_values("model_score", ascending=False), df, None


# =========================================================
# ANA RENDER FONKSİYONU
# =========================================================
def render_long_dashboard(selected_ticker: str) -> None:
    """Analiz.py'den çağrılır. Uzun vadeli model sonuçlarını gösterir."""

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

    # ── En Yüksek 5 Sıralama
    st.markdown('<div class="long-section-header">En Yüksek Sıralama</div>', unsafe_allow_html=True)
    top5 = results.head(5)

    rows_html = []
    for i, (_, r) in enumerate(top5.iterrows()):
        ticker    = str(r["ticker"])
        cname     = _TICKER_TO_NAME.get(ticker, ticker)
        score     = float(r["model_score"])
        label     = str(r.get("pred_label", "YÜKSELİŞ"))
        mom_val   = float(r.get("mom_126", 0))
        slope_val = float(r.get("sma50_slope20", 0))
        prob_up   = float(r.get("prob_up", 0))

        badge_cls = {
            "YÜKSELİŞ": "long-signal-up",
            "DÜŞÜŞ":    "long-signal-down",
            "NÖTR":     "long-signal-neutral",
        }.get(label, "long-signal-neutral")
        mom_color   = "#2ECC71" if mom_val   >= 0 else "#E74C3C"
        slope_color = "#2ECC71" if slope_val >= 0 else "#E74C3C"
        bar_w  = max(0, min(100, int(prob_up * 100)))
        bar_color = "#2ECC71" if label == "YÜKSELİŞ" else ("#E74C3C" if label == "DÜŞÜŞ" else "#4A5568")

        rows_html.append(
            "<tr>"
            + f'<td class="long-rank-num">#{i+1}</td>'
            + f'<td class="long-rank-ticker">{cname}'
            +   f'<br><span style="font-size:9px;color:#4A5568;font-weight:400;letter-spacing:1px;">{ticker}</span></td>'
            + f'<td><span class="long-signal-badge {badge_cls}" style="font-size:9px;padding:2px 8px;">{label}</span></td>'
            + f'<td style="color:{mom_color};text-align:right;">{("+" if mom_val >= 0 else "")}{mom_val:.2f}</td>'
            + f'<td style="color:{slope_color};text-align:right;">{("+" if slope_val >= 0 else "")}{slope_val:.3f}</td>'
            + f'<td style="min-width:120px;">'
            +   f'<span class="long-rank-score-pill">{score:.0f}</span>'
            +   f'<span class="long-rank-bar-wrap"><div class="long-rank-bar-fill" style="width:{bar_w}%;background:{bar_color};"></div></span>'
            + "</td>"
            + "</tr>"
        )

    full_table = (
        '<table class="long-rank-table">'
        '<thead><tr>'
        '<th style="width:28px"></th>'
        '<th>Hisse</th>'
        '<th>Sinyal</th>'
        '<th style="text-align:right;">Momentum 6A</th>'
        '<th style="text-align:right;">Eğim</th>'
        '<th>Skor</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + "</tbody></table>"
    )
    st.markdown(full_table, unsafe_allow_html=True)

    # ── Seçili ticker kontrolü
    available_tickers = results["ticker"].tolist()
    if selected_ticker not in available_tickers:
        st.warning(f"**{selected_ticker}** uzun vadeli model evreninde bulunamadı.")
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

    # ── FİYAT BAŞLIĞI
    st.markdown(f"""
    <div class="long-header-block">
        <div class="long-ticker-label">⚡ {sel_ticker}.US</div>
        <div class="long-big-price">${price:,.2f}</div>
        <div class="long-price-sub">Son Kapanış &nbsp;·&nbsp; Günlük Grafik</div>
    </div>
    <div class="long-section-divider"></div>
    """, unsafe_allow_html=True)

    # ── METRİK KARTLARI (CSS ızgarası — eşit yükseklik)
    score_pct = f"{model_score:.0f}"
    perc_str  = f"{percentile:.0f}"
    badge_cls = {
        "YÜKSELİŞ": "long-signal-up",
        "DÜŞÜŞ":    "long-signal-down",
        "NÖTR":     "long-signal-neutral",
    }.get(pred_label, "long-signal-neutral")
    up_pct  = int(prob_up     * 100)
    ne_pct  = int(prob_neutral * 100)
    dn_pct  = int(prob_down   * 100)
    mom_color   = "#2ECC71" if mom   >= 0 else "#E74C3C"
    slope_color = "#2ECC71" if slope >= 0 else "#E74C3C"
    dist_color  = "#2ECC71" if dist  >= 0 else "#E74C3C"

    cards_html = f"""
    <div class="long-card-grid">

      <!-- Kart 1: Model Sinyali -->
      <div class="long-metric-card">
        <div class="long-card-label">Model Sinyali</div>
        <span class="long-signal-badge {badge_cls}">{pred_label}</span>
        <div class="long-card-sub" style="margin-top:6px;">Skor {score_pct}/100 &nbsp;·&nbsp; P{perc_str}</div>
        <div style="margin-top:10px;">
          <div class="long-prob-row">
            <span class="long-prob-label">YÜK</span>
            <div class="long-prob-track"><div class="long-prob-fill-up" style="width:{up_pct}%"></div></div>
            <span class="long-prob-pct">{up_pct}%</span>
          </div>
          <div class="long-prob-row">
            <span class="long-prob-label">NÖT</span>
            <div class="long-prob-track"><div class="long-prob-fill-neutral" style="width:{ne_pct}%"></div></div>
            <span class="long-prob-pct">{ne_pct}%</span>
          </div>
          <div class="long-prob-row">
            <span class="long-prob-label">DÜŞ</span>
            <div class="long-prob-track"><div class="long-prob-fill-down" style="width:{dn_pct}%"></div></div>
            <span class="long-prob-pct">{dn_pct}%</span>
          </div>
        </div>
      </div>

      <!-- Kart 2: Momentum -->
      <div class="long-metric-card">
        <div class="long-card-label">Momentum 6 Ay</div>
        <div class="long-card-value" style="color:{mom_color} !important;">{'+' if mom >= 0 else ''}{mom:.2f}</div>
        <div class="long-card-sub">126 günlük getiri (normalize)</div>
      </div>

      <!-- Kart 3: Trend Eğimi -->
      <div class="long-metric-card">
        <div class="long-card-label">Trend Eğimi</div>
        <div class="long-card-value" style="color:{slope_color} !important;">{'+' if slope >= 0 else ''}{slope:.3f}</div>
        <div class="long-card-sub">SMA50 eğimi (20 günlük pencere)</div>
      </div>

      <!-- Kart 4: SMA200'e Uzaklık -->
      <div class="long-metric-card">
        <div class="long-card-label">SMA200 Uzaklığı</div>
        <div class="long-card-value" style="color:{dist_color} !important;">{'+' if dist >= 0 else ''}{dist:.2f}</div>
        <div class="long-card-sub">Göreli konumlanma</div>
      </div>

    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── FİYAT GRAFİĞİ
    st.markdown('<div class="long-section-header" style="margin-top:36px;">Fiyat Hareketi · 3 Ay</div>',
                unsafe_allow_html=True)

    if history_df is not None:
        hist = history_df[history_df["ticker"] == sel_ticker].tail(120).copy()

        if not hist.empty:
            hist["ema20"] = hist["close"].ewm(span=20).mean()
            hist["ema50"] = hist["close"].ewm(span=50).mean()
            hist["sma200"] = hist["close"].rolling(200).mean()

            fig = go.Figure()

            # Mum grafiği
            fig.add_trace(go.Candlestick(
                x=hist["datetime"],
                open=hist["open"], high=hist["high"],
                low=hist["low"], close=hist["close"],
                increasing_line_color="#2ECC71",
                increasing_fillcolor="rgba(46,204,113,0.75)",
                decreasing_line_color="#E74C3C",
                decreasing_fillcolor="rgba(231,76,60,0.75)",
                name="Fiyat", whiskerwidth=0.5
            ))

            # Hareketli ortalamalar
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

            # Hacim çubukları
            vol_colors = [
                "rgba(46,204,113,0.18)" if c >= o else "rgba(231,76,60,0.18)"
                for c, o in zip(hist["close"], hist["open"])
            ]
            fig.add_trace(go.Bar(
                x=hist["datetime"], y=hist["volume"],
                yaxis="y2", marker_color=vol_colors,
                name="Hacim", showlegend=False
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

            # Altın rengi üst aksan çizgisi
            fig.add_shape(
                type="line", xref="paper", yref="paper",
                x0=0, y0=1, x1=0.15, y1=1,
                line=dict(color="#FFB000", width=2),
            )

            st.plotly_chart(fig, width='stretch')
        else:
            st.warning(f"{sel_ticker} için geçmiş fiyat verisi bulunamadı.")
    else:
        st.warning("Fiyat verisi yüklenemedi.")
