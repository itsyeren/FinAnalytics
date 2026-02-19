"""
app/pages/mid.py — Multi-Sektör Orta Vadeli Tahmin Dashboard
=============================================================
Alpaca API + sector_optimized_params.json
RandomForest + LightGBM Regressor · 1M / 3M tahminler
render_mid_dashboard(selected_ticker) olarak App.py'den çağrılır.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# ROOT PATH
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

# Alpaca API
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from dotenv import load_dotenv

    if _ROOT_DIR:
        load_dotenv(_ROOT_DIR / ".env")
    else:
        load_dotenv()

    _ALPACA_KEY = os.getenv("ALPACA_API_KEY")
    _ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")

    if _ALPACA_KEY and _ALPACA_SECRET:
        from alpaca.data.historical import StockHistoricalDataClient
        _client = StockHistoricalDataClient(_ALPACA_KEY, _ALPACA_SECRET)
        _HAS_ALPACA = True
    else:
        _HAS_ALPACA = False
        _client = None
except Exception:
    _HAS_ALPACA = False
    _client = None

_MULTI_SEKTOR_DIR = _ROOT_DIR / "multi_sektor_analiz" if _ROOT_DIR else Path(".")
_SECTOR_JSON = _MULTI_SEKTOR_DIR / "reports" / "sector_optimized_params.json"
if not _SECTOR_JSON.exists():
    _SECTOR_JSON = _MULTI_SEKTOR_DIR / "sector_optimized_params.json"


# =========================================================
# CSS
# =========================================================
_MID_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.mid-metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    text-align: center;
    transition: transform 0.2s ease;
}
.mid-metric-card:hover { transform: translateY(-2px); }
.mid-metric-label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.mid-metric-value { font-size: 1.8rem; font-weight: 700; color: #e6f1ff; }
.mid-metric-delta-up { color: #64ffda; font-size: 1rem; font-weight: 600; }
.mid-metric-delta-down { color: #ff6b6b; font-size: 1rem; font-weight: 600; }

.mid-signal-al {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: #fff; padding: 0.4rem 1.2rem; border-radius: 20px;
    font-weight: 700; font-size: 1rem; display: inline-block;
}
.mid-signal-bekle {
    background: linear-gradient(135deg, #fdcb6e, #f39c12);
    color: #2d3436; padding: 0.4rem 1.2rem; border-radius: 20px;
    font-weight: 700; font-size: 1rem; display: inline-block;
}
.mid-signal-sat {
    background: linear-gradient(135deg, #e17055, #d63031);
    color: #fff; padding: 0.4rem 1.2rem; border-radius: 20px;
    font-weight: 700; font-size: 1rem; display: inline-block;
}

.mid-section-header {
    font-size: 1.2rem; font-weight: 600; color: #ccd6f6;
    margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(100, 255, 218, 0.2);
}
</style>
"""


# =========================================================
# HELPERS
# =========================================================
@st.cache_data(ttl=3600)
def _load_json_report():
    if _SECTOR_JSON.exists():
        with open(_SECTOR_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=3600)
def _fetch_stock_history(symbol, days_back=504):
    if not _HAS_ALPACA or _client is None:
        return pd.DataFrame()
    try:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d")
        )
        bars = _client.get_stock_bars(req)
        df = bars.df.reset_index()
        df = df[df["symbol"] == symbol].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def _get_stock_prediction(report, symbol):
    if report is None:
        return None, None, None
    for sector_name, sector_data in report.get("sectors", {}).items():
        if symbol in sector_data.get("stock_predictions", {}):
            pred = sector_data["stock_predictions"][symbol]
            model_data = sector_data.get("models", {})
            return pred, model_data, sector_name
    return None, None, None


def _signal_badge(signal):
    css_class = {
        "AL": "mid-signal-al",
        "BEKLE": "mid-signal-bekle",
        "SAT": "mid-signal-sat",
    }.get(signal, "mid-signal-bekle")
    return f'<span class="{css_class}">{signal}</span>'


# =========================================================
# ANA RENDER FONKSİYONU
# =========================================================
def render_mid_dashboard(selected_ticker: str) -> None:
    """App.py'den çağrılır. Orta vadeli model tahminlerini gösterir."""

    st.markdown(_MID_CSS, unsafe_allow_html=True)

    if _ROOT_DIR is None:
        st.error("Proje kök dizini bulunamadı.")
        return

    report = _load_json_report()
    if report is None:
        st.error("sector_optimized_params.json bulunamadı!")
        return

    # Rapordaki tüm hisseleri bul
    dashboard_stocks = []
    stock_info = {}
    for sector_name, sector_data in report.get("sectors", {}).items():
        for ticker in sector_data.get("stock_predictions", {}).keys():
            dashboard_stocks.append(ticker)
            stock_info[ticker] = {"name": ticker, "sector": sector_name, "emoji": "📊"}

    if not dashboard_stocks:
        st.error("Raporda tahmin bulunamadı!")
        return

    # Eğer App.py'den gelen ticker raporda varsa onu kullan,
    # yoksa kullanıcıya rapordaki hisselerden seçtir
    if selected_ticker in dashboard_stocks:
        active_stock = selected_ticker
    else:
        st.info(f"**{selected_ticker}** için orta vadeli model raporu bulunamadı. "
                f"Aşağıdan rapordaki bir hisseyi seçebilirsiniz.")
        active_stock = st.selectbox(
            "🔍 Rapordaki hisse seçin",
            dashboard_stocks,
            format_func=lambda x: f"{stock_info[x]['emoji']} {x} ({stock_info[x]['sector']})",
            key="mid_stock_select",
        )

    # Ayarlar
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        time_range = st.radio("📅 Zaman Aralığı", ["6 Ay", "1 Yıl", "2 Yıl"],
                              index=1, horizontal=True, key="mid_time_range")
    with col_cfg2:
        show_sma = st.checkbox("📉 SMA 50 & SMA 200 göster", value=True, key="mid_show_sma")

    pred, model_data, sector_name = _get_stock_prediction(report, active_stock)

    if pred is None:
        st.error(f"{active_stock} için tahmin verisi bulunamadı!")
        return

    days_map = {"6 Ay": 180, "1 Yıl": 365, "2 Yıl": 730}
    days_back = days_map[time_range]

    info = stock_info[active_stock]

    # ── Metric kartları
    st.markdown(
        f'<div class="mid-section-header">{info["emoji"]} {active_stock} '
        f'<span style="font-size:0.85rem;color:#8892b0;">({sector_name})</span></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="mid-metric-card">
            <div class="mid-metric-label">Son Fiyat</div>
            <div class="mid-metric-value">${pred['son_fiyat']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        d1m = pred.get('getiri_1ay_pct', 0)
        d_cls = "mid-metric-delta-up" if d1m > 0 else "mid-metric-delta-down"
        st.markdown(f"""
        <div class="mid-metric-card">
            <div class="mid-metric-label">1 Ay Tahmin</div>
            <div class="mid-metric-value">${pred['tahmin_1ay']:.2f}</div>
            <div class="{d_cls}">{d1m:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="mid-metric-card">
            <div class="mid-metric-label">1 Ay Sinyal</div>
            <div style="margin-top:0.5rem;">{_signal_badge(pred.get('sinyal_1ay', 'BEKLE'))}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        d3m = pred.get('getiri_3ay_pct', 0)
        d_cls = "mid-metric-delta-up" if d3m > 0 else "mid-metric-delta-down"
        st.markdown(f"""
        <div class="mid-metric-card">
            <div class="mid-metric-label">3 Ay Tahmin</div>
            <div class="mid-metric-value">${pred['tahmin_3ay']:.2f}</div>
            <div class="{d_cls}">{d3m:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="mid-metric-card">
            <div class="mid-metric-label">3 Ay Sinyal</div>
            <div style="margin-top:0.5rem;">{_signal_badge(pred.get('sinyal_3ay', 'BEKLE'))}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fiyat grafiği
    st.markdown('<div class="mid-section-header">📊 Fiyat Grafiği & Tahminler</div>',
                unsafe_allow_html=True)

    with st.spinner(f"{active_stock} fiyat verisi yükleniyor..."):
        df_hist = _fetch_stock_history(active_stock, days_back=days_back + 60)

    if not df_hist.empty:
        df_hist = df_hist.sort_values("timestamp")
        df_hist["sma_50"] = df_hist["close"].rolling(50).mean()
        df_hist["sma_200"] = df_hist["close"].rolling(200).mean()

        last_date = df_hist["timestamp"].iloc[-1]
        last_price = df_hist["close"].iloc[-1]
        date_1m = last_date + timedelta(days=30)
        date_3m = last_date + timedelta(days=90)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"], y=df_hist["close"],
            mode="lines", name=f"{active_stock} Fiyat",
            line=dict(color="#667eea", width=2.5),
            fill="tozeroy", fillcolor="rgba(102, 126, 234, 0.08)"
        ))

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

        fig.add_trace(go.Scatter(
            x=[last_date, date_1m], y=[last_price, pred["tahmin_1ay"]],
            mode="lines+markers",
            name=f"1 Ay Tahmin (${pred['tahmin_1ay']:.2f})",
            line=dict(color="#64ffda", width=2.5, dash="dot"),
            marker=dict(size=[0, 14], color="#64ffda", symbol="diamond",
                        line=dict(width=2, color="#0a192f")),
        ))

        fig.add_trace(go.Scatter(
            x=[last_date, date_3m], y=[last_price, pred["tahmin_3ay"]],
            mode="lines+markers",
            name=f"3 Ay Tahmin (${pred['tahmin_3ay']:.2f})",
            line=dict(color="#c084fc", width=2.5, dash="dot"),
            marker=dict(size=[0, 14], color="#c084fc", symbol="star",
                        line=dict(width=2, color="#0a192f")),
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 25, 47, 0.6)",
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=11, color="#8892b0"), bgcolor="rgba(0,0,0,0)"
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zeroline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zeroline=False,
                       title=dict(text="Fiyat ($)", font=dict(color="#8892b0"))),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        if not _HAS_ALPACA:
            st.warning("Alpaca API yapılandırılmamış. .env dosyasına ALPACA_API_KEY ve ALPACA_API_SECRET ekleyin.")
        else:
            st.warning("Fiyat verisi yüklenemedi.")

    # ── Karşılaştırma tablosu
    st.markdown('<div class="mid-section-header">🏆 Tüm Hisseler Karşılaştırması</div>',
                unsafe_allow_html=True)

    comparison_data = []
    for stock in dashboard_stocks:
        p, _, sn = _get_stock_prediction(report, stock)
        if p:
            comparison_data.append({
                "Hisse": f"{stock_info[stock]['emoji']} {stock}",
                "Sektör": sn,
                "Son Fiyat": f"${p['son_fiyat']:.2f}",
                "1M Tahmin": f"${p['tahmin_1ay']:.2f}",
                "1M Getiri": f"{p.get('getiri_1ay_pct', 0):+.2f}%",
                "1M Sinyal": p.get('sinyal_1ay', '—'),
                "3M Tahmin": f"${p['tahmin_3ay']:.2f}",
                "3M Getiri": f"{p.get('getiri_3ay_pct', 0):+.2f}%",
                "3M Sinyal": p.get('sinyal_3ay', '—'),
            })

    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # ── Model performansı
    if model_data:
        with st.expander("🔬 Model Performans Metrikleri", expanded=False):
            col_1m, col_3m = st.columns(2)

            for col, key, label in [(col_1m, "1m", "1 Aylık"), (col_3m, "3m", "3 Aylık")]:
                with col:
                    if key in model_data:
                        m = model_data[key]
                        st.markdown(f"**📅 {label} Model — {m.get('best_model', '?')}**")
                        metrics = {
                            "MAE": m.get("regression_scores", {}).get("mae", 0),
                            "RMSE": m.get("regression_scores", {}).get("rmse", 0),
                            "R²": m.get("regression_scores", {}).get("r2", 0),
                        }
                        st.dataframe(
                            pd.DataFrame({"Metrik": metrics.keys(),
                                          "Değer": [f"{v:.4f}" for v in metrics.values()]}),
                            use_container_width=True, hide_index=True
                        )

    # ── Footer
    if report and "metadata" in report:
        st.markdown(
            f'<div style="text-align:center;color:#8892b0;font-size:0.8rem;padding:2rem 0;">'
            f'Son güncelleme: {report["metadata"].get("created_at", "?")} | '
            f'Eğitim aralığı: {report["metadata"].get("train_range", "?")} | '
            f'Sinyal eşiği: {report["metadata"].get("signal_threshold", "?")}'
            f'</div>',
            unsafe_allow_html=True
        )
