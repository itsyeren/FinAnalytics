"""
src/mid.py — Orta Vadeli Tahmin Dashboard
==========================================
RandomForest + LightGBM Regressor · 1A / 3A tahminler.
Analysis.py'den render_mid_dashboard(selected_ticker) olarak çağrılır.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Proje kök dizini ─────────────────────────────────────────────────────────
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

# ── Alpaca API (opsiyonel) ────────────────────────────────────────────────────
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from dotenv import load_dotenv

    load_dotenv(_ROOT_DIR / ".env" if _ROOT_DIR else ".env")

    _ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
    _ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")

    if _ALPACA_KEY and _ALPACA_SECRET:
        from alpaca.data.historical import StockHistoricalDataClient
        _client    = StockHistoricalDataClient(_ALPACA_KEY, _ALPACA_SECRET)
        _HAS_ALPACA = True
    else:
        _HAS_ALPACA = False
        _client     = None
except Exception:
    _HAS_ALPACA = False
    _client     = None

# ── JSON rapor yolu ───────────────────────────────────────────────────────────
_MULTI_SEKTOR_DIR = _ROOT_DIR / "models" / "mid_term" if _ROOT_DIR else Path(".")
_SECTOR_JSON      = _MULTI_SEKTOR_DIR / "sector_optimized_params.json"
if not _SECTOR_JSON.exists():
    _SECTOR_JSON  = _MULTI_SEKTOR_DIR / "src" / "reports" / "sector_optimized_params.json"


# ── CSS ───────────────────────────────────────────────────────────────────────
_MID_CSS = """
<style>
/* Orta vadeli score card — Analysis.py sc-card grid ile uyumlu */
.mid-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.mid-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 20px 18px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 110px;
    transition: border-color 0.2s, background 0.2s;
}
.mid-card:hover {
    border-color: rgba(99,179,237,0.4);
    background: rgba(99,179,237,0.07);
}
.mid-card-label {
    font-size: 11px;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    line-height: 1.3;
}
.mid-card-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e8edf5;
    line-height: 1.2;
}
.mid-delta-up   { color: #64ffda; font-size: 0.85rem; font-weight: 600; margin-top: 4px; }
.mid-delta-down { color: #ff6b6b; font-size: 0.85rem; font-weight: 600; margin-top: 4px; }

/* Sinyal pilleri */
.mid-sig-al {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: #fff; padding: 5px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}
.mid-sig-bekle {
    background: linear-gradient(135deg, #fdcb6e, #f39c12);
    color: #2d3436; padding: 5px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}
.mid-sig-sat {
    background: linear-gradient(135deg, #e17055, #d63031);
    color: #fff; padding: 5px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}

/* Bölüm başlığı */
.mid-section-header {
    font-size: 1.05rem; font-weight: 600; color: #ccd6f6;
    margin: 24px 0 12px 0; padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}
</style>
"""


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _load_json_report():
    if _SECTOR_JSON.exists():
        with open(_SECTOR_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=3600)
def _fetch_stock_history(symbol: str, days_back: int = 504) -> pd.DataFrame:
    if not _HAS_ALPACA or _client is None:
        return pd.DataFrame()
    try:
        end   = datetime.now()
        start = end - timedelta(days=days_back)
        req   = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        bars = _client.get_stock_bars(req)
        df   = bars.df.reset_index()
        df   = df[df["symbol"] == symbol].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def _get_stock_prediction(report, symbol):
    if report is None:
        return None, None
    for sector_name, sector_data in report.get("sectors", {}).items():
        if symbol in sector_data.get("stock_predictions", {}):
            pred = sector_data["stock_predictions"][symbol]
            return pred, sector_name
    return None, None


def _signal_badge(signal: str) -> str:
    """Ham sinyal metnine göre renkli badge HTML üretir."""
    cls = {
        "AL":    "mid-sig-al",
        "BEKLE": "mid-sig-bekle",
        "SAT":   "mid-sig-sat",
    }.get(signal, "mid-sig-bekle")
    return f'<span class="{cls}">{signal}</span>'


def _card(label: str, value_html: str, delta_html: str = "") -> str:
    """Eşit boyutlu score card HTML bloğu üretir."""
    return (
        f'<div class="mid-card">'
        f'<div class="mid-card-label">{label}</div>'
        f'<div class="mid-card-value">{value_html}</div>'
        f'{delta_html}'
        f'</div>'
    )


def _delta(pct: float) -> str:
    cls = "mid-delta-up" if pct >= 0 else "mid-delta-down"
    return f'<div class="{cls}">{pct:+.2f}%</div>'


# ── Ana render fonksiyonu ─────────────────────────────────────────────────────
def render_mid_dashboard(selected_ticker: str) -> None:
    """Analysis.py'den çağrılır. Orta vadeli model tahminlerini gösterir."""

    st.markdown(_MID_CSS, unsafe_allow_html=True)

    if _ROOT_DIR is None:
        st.error("Proje kök dizini bulunamadı.")
        return

    report = _load_json_report()
    if report is None:
        st.error(
            "sector_optimized_params.json bulunamadı! "
            "Raporu oluşturmak için: `python models/mid_term/src/train.py` çalıştırın."
        )
        return

    # Rapordaki tüm hisseleri topla
    all_stocks = []
    stock_sector: dict[str, str] = {}
    for sector_name, sector_data in report.get("sectors", {}).items():
        for ticker in sector_data.get("stock_predictions", {}).keys():
            all_stocks.append(ticker)
            stock_sector[ticker] = sector_name

    if not all_stocks:
        st.error("Raporda tahmin bulunamadı!")
        return

    # Sidebar'dan gelen ticker raporda yoksa alternatif seçim
    if selected_ticker in all_stocks:
        active_stock = selected_ticker
    else:
        st.info(
            f"**{selected_ticker}** için orta vadeli model raporu bulunamadı. "
            "Aşağıdan rapordaki bir hisseyi seçebilirsiniz."
        )
        active_stock = st.selectbox(
            "🔍 Rapordaki hisse seçin",
            all_stocks,
            format_func=lambda x: f"{x}  ({stock_sector.get(x, '?')})",
            key="mid_stock_select",
        )

    pred, sector_name = _get_stock_prediction(report, active_stock)
    if pred is None:
        st.error(f"{active_stock} için tahmin verisi bulunamadı!")
        return

    # ── Başlık ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="mid-section-header">'
        f'📊 {active_stock} '
        f'<span style="font-size:0.85rem;color:#8892b0;font-weight:400;">({sector_name})</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 5 eşit boyutlu score card ────────────────────────────────────────────
    d1m = pred.get("getiri_1ay_pct", 0)
    d3m = pred.get("getiri_3ay_pct", 0)

    cards_html = "".join([
        _card("Son Fiyat",   f"${pred['son_fiyat']:.2f}"),
        _card("1 Ay Tahmin", f"${pred['tahmin_1ay']:.2f}", _delta(d1m)),
        _card("1 Ay Sinyal", _signal_badge(pred.get("sinyal_1ay", "BEKLE"))),
        _card("3 Ay Tahmin", f"${pred['tahmin_3ay']:.2f}", _delta(d3m)),
        _card("3 Ay Sinyal", _signal_badge(pred.get("sinyal_3ay", "BEKLE"))),
    ])
    st.markdown(f'<div class="mid-grid">{cards_html}</div>', unsafe_allow_html=True)

    # ── Grafik kontrolleri ───────────────────────────────────────────────────
    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_l:
        time_range = st.radio(
            "Zaman Aralığı",
            ["6 Ay", "1 Yıl", "2 Yıl"],
            index=1, horizontal=True, key="mid_time_range",
        )
    with ctrl_r:
        show_sma = st.checkbox("SMA 50 & 200", value=True, key="mid_show_sma")

    days_back = {"6 Ay": 180, "1 Yıl": 365, "2 Yıl": 730}[time_range]

    # ── Fiyat + tahmin grafiği ───────────────────────────────────────────────
    st.markdown('<div class="mid-section-header">📈 Fiyat Grafiği &amp; Tahminler</div>',
                unsafe_allow_html=True)

    with st.spinner(f"{active_stock} fiyat verisi yükleniyor…"):
        df_hist = _fetch_stock_history(active_stock, days_back=days_back + 60)

    if not df_hist.empty:
        df_hist = df_hist.sort_values("timestamp")
        df_hist["sma_50"]  = df_hist["close"].rolling(50).mean()
        df_hist["sma_200"] = df_hist["close"].rolling(200).mean()

        last_date  = df_hist["timestamp"].iloc[-1]
        last_price = float(df_hist["close"].iloc[-1])
        date_1m    = last_date + timedelta(days=30)
        date_3m    = last_date + timedelta(days=90)

        fig = go.Figure()

        # Fiyat serisi
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"], y=df_hist["close"],
            mode="lines", name=f"{active_stock} Fiyat",
            line=dict(color="#63b3ed", width=2.5),
            fill="tozeroy", fillcolor="rgba(99,179,237,0.07)",
        ))

        # SMA'lar
        if show_sma:
            fig.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["sma_50"],
                mode="lines", name="SMA 50",
                line=dict(color="#ffd93d", width=1.5, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["sma_200"],
                mode="lines", name="SMA 200",
                line=dict(color="#ff6b6b", width=1.5, dash="dash"),
            ))

        # 1 Ay tahmini
        fig.add_trace(go.Scatter(
            x=[last_date, date_1m],
            y=[last_price, pred["tahmin_1ay"]],
            mode="lines+markers",
            name=f"1 Ay Tahmin  ${pred['tahmin_1ay']:.2f}",
            line=dict(color="#64ffda", width=2.5, dash="dot"),
            marker=dict(size=[0, 12], color="#64ffda", symbol="diamond",
                        line=dict(width=2, color="#0a192f")),
        ))

        # 3 Ay tahmini
        fig.add_trace(go.Scatter(
            x=[last_date, date_3m],
            y=[last_price, pred["tahmin_3ay"]],
            mode="lines+markers",
            name=f"3 Ay Tahmin  ${pred['tahmin_3ay']:.2f}",
            line=dict(color="#c084fc", width=2.5, dash="dot"),
            marker=dict(size=[0, 12], color="#c084fc", symbol="star",
                        line=dict(width=2, color="#0a192f")),
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,25,47,0.5)",
            height=480,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
                font=dict(size=11, color="#8892b0"),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                title=dict(text="Fiyat ($)", font=dict(color="#8892b0", size=11)),
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        if not _HAS_ALPACA:
            st.warning(
                "⚠️ Alpaca API yapılandırılmamış — `.env` dosyasına "
                "`ALPACA_API_KEY` ve `ALPACA_API_SECRET` ekleyin."
            )
        else:
            st.warning(f"⚠️ {active_stock} için fiyat verisi alınamadı.")

    # ── Alt bilgi ────────────────────────────────────────────────────────────
    meta = report.get("metadata", {})
    if meta:
        parts = []
        if meta.get("created_at"):
            parts.append(f"Son güncelleme: {meta['created_at']}")
        if meta.get("train_range"):
            parts.append(f"Eğitim aralığı: {meta['train_range']}")
        if meta.get("signal_threshold"):
            parts.append(f"Sinyal eşiği: {meta['signal_threshold']}")
        if parts:
            st.markdown(
                f'<div style="text-align:center;color:#3d4f6b;font-size:0.78rem;'
                f'padding:20px 0 8px;">{" · ".join(parts)}</div>',
                unsafe_allow_html=True,
            )
    st.caption("⚠️ Model tahminleri eğitim/araştırma amaçlıdır. Yatırım tavsiyesi değildir.")
