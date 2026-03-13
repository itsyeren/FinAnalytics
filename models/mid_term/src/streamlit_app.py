"""
streamlit_app.py - Hisse Senedi Tahmin Dashboard
NVDA, COST, KMB icin time series + tahmin gorsellestirmesi
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# Alpaca veri cekimi icin
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import client

# ==============================================================================
# SAYFA AYARLARI
# ==============================================================================
st.set_page_config(
    page_title="Hisse Tahmin Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Dark theme overrides */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e6f1ff;
    }
    .metric-delta-up {
        color: #64ffda;
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-delta-down {
        color: #ff6b6b;
        font-size: 1rem;
        font-weight: 600;
    }

    /* Signal badge */
    .signal-al {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: #fff;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .signal-bekle {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: #2d3436;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .signal-sat {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: #fff;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    /* Header */
    .dashboard-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .dashboard-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .dashboard-header p {
        color: #8892b0;
        font-size: 0.95rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ccd6f6;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(100, 255, 218, 0.2);
    }

    /* Model info box */
    .model-info-box {
        background: rgba(100, 255, 218, 0.05);
        border: 1px solid rgba(100, 255, 218, 0.15);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }

    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .styled-table th {
        background: rgba(100, 255, 218, 0.1);
        color: #64ffda;
        padding: 0.8rem;
        text-align: left;
        font-weight: 600;
    }
    .styled-table td {
        padding: 0.7rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #ccd6f6;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a192f 0%, #112240 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# VERI YUKLEME
# ==============================================================================
DASHBOARD_STOCKS = ["NVDA", "COST", "KMB"]

STOCK_INFO = {
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Teknoloji", "emoji": "🟢"},
    "COST": {"name": "Costco Wholesale", "sector": "Perakende_Temel", "emoji": "🔵"},
    "KMB": {"name": "Kimberly-Clark", "sector": "Ev_Kisisel", "emoji": "🟣"},
}


@st.cache_data(ttl=3600)
def load_json_report():
    """sector_optimized_params.json dosyasini yukle."""
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "reports", "sector_optimized_params.json"
    )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def fetch_stock_history(symbol, days_back=504):
    """Alpaca API'den gecmis fiyat verisini cek."""
    end = datetime.now()
    start = end - timedelta(days=days_back)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d")
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()
    df = df[df["symbol"] == symbol].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_stock_prediction(report, symbol):
    """JSON rapordaki tahmin verisini bul."""
    for sector_name, sector_data in report["sectors"].items():
        if symbol in sector_data.get("stock_predictions", {}):
            pred = sector_data["stock_predictions"][symbol]
            model_data = sector_data["models"]
            return pred, model_data, sector_name
    return None, None, None


def signal_badge(signal):
    """Sinyal icin HTML badge."""
    css_class = {
        "AL": "signal-al",
        "BEKLE": "signal-bekle",
        "SAT": "signal-sat"
    }.get(signal, "signal-bekle")
    return f'<span class="{css_class}">{signal}</span>'


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## 📊 Dashboard Ayarları")
    st.markdown("---")

    selected_stock = st.selectbox(
        "🔍 Hisse Seçin",
        DASHBOARD_STOCKS,
        format_func=lambda x: f"{STOCK_INFO[x]['emoji']} {x} - {STOCK_INFO[x]['name']}"
    )

    time_range = st.radio(
        "📅 Zaman Aralığı",
        ["6 Ay", "1 Yıl", "2 Yıl"],
        index=1
    )

    show_sma = st.checkbox("📉 SMA 50 & SMA 200 göster", value=True)
    show_volume = st.checkbox("📊 Hacim grafiği göster", value=False)

    st.markdown("---")
    st.markdown(
        '<p style="color:#8892b0;font-size:0.75rem;">Multi-Sektör Fiyat Tahmin Sistemi<br>'
        'RandomForest + LightGBM Regressor</p>',
        unsafe_allow_html=True
    )

# ==============================================================================
# ANA ICERIK
# ==============================================================================
# Header
st.markdown("""
<div class="dashboard-header">
    <h1>📈 Hisse Senedi Tahmin Dashboard</h1>
    <p>Multi-Sektör Fiyat Tahmin Sistemi — NVDA • COST • KMB</p>
</div>
""", unsafe_allow_html=True)

# Veri yukle
report = load_json_report()
pred, model_data, sector_name = get_stock_prediction(report, selected_stock)

if pred is None:
    st.error(f"{selected_stock} için tahmin verisi bulunamadı!")
    st.stop()

# Zaman araligini gun sayisina cevir
days_map = {"6 Ay": 180, "1 Yıl": 365, "2 Yıl": 730}
days_back = days_map[time_range]

# Gecmis veriyi cek
with st.spinner(f"{selected_stock} verisi yükleniyor..."):
    df_hist = fetch_stock_history(selected_stock, days_back=days_back + 60)

info = STOCK_INFO[selected_stock]

# ==============================================================================
# METRIK KARTLARI
# ==============================================================================
st.markdown(
    f'<div class="section-header">{info["emoji"]} {selected_stock} — {info["name"]} '
    f'<span style="font-size:0.85rem;color:#8892b0;">({sector_name})</span></div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Son Fiyat</div>
        <div class="metric-value">${pred['son_fiyat']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    delta_1m = pred['getiri_1ay_pct']
    delta_class = "metric-delta-up" if delta_1m > 0 else "metric-delta-down"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">1 Ay Tahmin</div>
        <div class="metric-value">${pred['tahmin_1ay']:.2f}</div>
        <div class="{delta_class}">{delta_1m:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">1 Ay Sinyal</div>
        <div style="margin-top:0.5rem;">{signal_badge(pred['sinyal_1ay'])}</div>
        <div style="color:#8892b0;font-size:0.75rem;margin-top:0.3rem;">{pred['model_1m']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    delta_3m = pred['getiri_3ay_pct']
    delta_class = "metric-delta-up" if delta_3m > 0 else "metric-delta-down"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">3 Ay Tahmin</div>
        <div class="metric-value">${pred['tahmin_3ay']:.2f}</div>
        <div class="{delta_class}">{delta_3m:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">3 Ay Sinyal</div>
        <div style="margin-top:0.5rem;">{signal_badge(pred['sinyal_3ay'])}</div>
        <div style="color:#8892b0;font-size:0.75rem;margin-top:0.3rem;">{pred['model_3m']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# TIME SERIES GRAFIGI
# ==============================================================================
st.markdown('<div class="section-header">📊 Fiyat Grafiği & Tahminler</div>', unsafe_allow_html=True)

# SMA hesapla
if not df_hist.empty:
    df_hist = df_hist.sort_values("timestamp")
    df_hist["sma_50"] = df_hist["close"].rolling(50).mean()
    df_hist["sma_200"] = df_hist["close"].rolling(200).mean()

    # Son tarihi al
    last_date = df_hist["timestamp"].iloc[-1]
    last_price = df_hist["close"].iloc[-1]

    # Tahmin tarihlerini hesapla (is gunleri)
    date_1m = last_date + timedelta(days=30)
    date_3m = last_date + timedelta(days=90)

    # --- PLOTLY GRAFIK ---
    fig = go.Figure()

    # Hacim (opsiyonel, alt grafik olarak)
    if show_volume:
        fig = go.Figure(
            data=[],
            layout=go.Layout(
                xaxis=dict(domain=[0, 1]),
                yaxis=dict(domain=[0.25, 1], title="Fiyat ($)"),
                yaxis2=dict(domain=[0, 0.2], title="Hacim", anchor="x"),
            )
        )
        # Hacim barlari
        colors = ['#00b894' if c >= o else '#e17055'
                  for c, o in zip(df_hist["close"], df_hist["open"])]
        fig.add_trace(go.Bar(
            x=df_hist["timestamp"],
            y=df_hist["volume"],
            marker_color=colors,
            opacity=0.4,
            name="Hacim",
            yaxis="y2",
            showlegend=False
        ))
    else:
        fig = go.Figure()

    # Fiyat cizgisi (candlestick yerine clean line)
    fig.add_trace(go.Scatter(
        x=df_hist["timestamp"],
        y=df_hist["close"],
        mode="lines",
        name=f"{selected_stock} Fiyat",
        line=dict(color="#667eea", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(102, 126, 234, 0.08)",
        yaxis="y" if not show_volume else "y"
    ))

    # SMA cizgileri
    if show_sma:
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["sma_50"],
            mode="lines",
            name="SMA 50",
            line=dict(color="#ffd93d", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["sma_200"],
            mode="lines",
            name="SMA 200",
            line=dict(color="#ff6b6b", width=1.5, dash="dash"),
        ))

    # 1M Tahmin noktasi
    fig.add_trace(go.Scatter(
        x=[last_date, date_1m],
        y=[last_price, pred["tahmin_1ay"]],
        mode="lines+markers",
        name=f"1 Ay Tahmin (${pred['tahmin_1ay']:.2f})",
        line=dict(color="#64ffda", width=2.5, dash="dot"),
        marker=dict(size=[0, 14], color="#64ffda", symbol="diamond",
                    line=dict(width=2, color="#0a192f")),
    ))

    # 3M Tahmin noktasi
    fig.add_trace(go.Scatter(
        x=[last_date, date_3m],
        y=[last_price, pred["tahmin_3ay"]],
        mode="lines+markers",
        name=f"3 Ay Tahmin (${pred['tahmin_3ay']:.2f})",
        line=dict(color="#c084fc", width=2.5, dash="dot"),
        marker=dict(size=[0, 14], color="#c084fc", symbol="star",
                    line=dict(width=2, color="#0a192f")),
    ))

    # 1M yatay cizgi
    fig.add_hline(
        y=pred["tahmin_1ay"],
        line_dash="dot",
        line_color="rgba(100, 255, 218, 0.3)",
        annotation_text=f"1M: ${pred['tahmin_1ay']:.2f}",
        annotation_position="right",
        annotation_font_color="#64ffda",
        annotation_font_size=11,
    )

    # 3M yatay cizgi
    fig.add_hline(
        y=pred["tahmin_3ay"],
        line_dash="dot",
        line_color="rgba(192, 132, 252, 0.3)",
        annotation_text=f"3M: ${pred['tahmin_3ay']:.2f}",
        annotation_position="right",
        annotation_font_color="#c084fc",
        annotation_font_size=11,
    )

    # Son fiyat annotasyonu
    fig.add_annotation(
        x=last_date,
        y=last_price,
        text=f"Son: ${last_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#667eea",
        font=dict(color="#667eea", size=12, family="Inter"),
        bgcolor="rgba(10, 25, 47, 0.9)",
        bordercolor="#667eea",
        borderwidth=1,
        borderpad=6,
        ax=-60,
        ay=-40,
    )

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 25, 47, 0.6)",
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#8892b0"),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
            zeroline=False,
            title=dict(text="Fiyat ($)", font=dict(color="#8892b0")),
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Fiyat verisi yüklenemedi.")

# ==============================================================================
# 3 HISSE KARSILASTIRMA TABLOSU
# ==============================================================================
st.markdown('<div class="section-header">🏆 Top 3 Hisse Karşılaştırması</div>', unsafe_allow_html=True)

comparison_data = []
for stock in DASHBOARD_STOCKS:
    p, _, sn = get_stock_prediction(report, stock)
    if p:
        comparison_data.append({
            "Hisse": f"{STOCK_INFO[stock]['emoji']} {stock}",
            "Sektör": sn,
            "Son Fiyat": f"${p['son_fiyat']:.2f}",
            "1M Tahmin": f"${p['tahmin_1ay']:.2f}",
            "1M Getiri": f"{p['getiri_1ay_pct']:+.2f}%",
            "1M Sinyal": p['sinyal_1ay'],
            "3M Tahmin": f"${p['tahmin_3ay']:.2f}",
            "3M Getiri": f"{p['getiri_3ay_pct']:+.2f}%",
            "3M Sinyal": p['sinyal_3ay'],
        })

if comparison_data:
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(
        df_comp,
        use_container_width=True,
        hide_index=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MODEL PERFORMANS METRIKLERI (Acilir-Kapanir)
# ==============================================================================
with st.expander("🔬 Model Performans Metrikleri", expanded=False):
    col_1m, col_3m = st.columns(2)

    with col_1m:
        m1 = model_data["1m"]
        st.markdown(f"**📅 1 Aylık Model — {m1['best_model']}**")
        df_m1 = pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "R²", "Sinyal Accuracy", "Sinyal Precision", "Sinyal Recall", "Sinyal F1"],
            "Değer": [
                f"{m1['regression_scores']['mae']:.4f}",
                f"{m1['regression_scores']['rmse']:.4f}",
                f"{m1['regression_scores']['r2']:.4f}",
                f"%{m1['signal_scores']['accuracy']*100:.1f}",
                f"%{m1['signal_scores']['precision']*100:.1f}",
                f"%{m1['signal_scores']['recall']*100:.1f}",
                f"%{m1['signal_scores']['f1']*100:.1f}",
            ]
        })
        st.dataframe(df_m1, use_container_width=True, hide_index=True)

    with col_3m:
        m3 = model_data["3m"]
        st.markdown(f"**📅 3 Aylık Model — {m3['best_model']}**")
        df_m3 = pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "R²", "Sinyal Accuracy", "Sinyal Precision", "Sinyal Recall", "Sinyal F1"],
            "Değer": [
                f"{m3['regression_scores']['mae']:.4f}",
                f"{m3['regression_scores']['rmse']:.4f}",
                f"{m3['regression_scores']['r2']:.4f}",
                f"%{m3['signal_scores']['accuracy']*100:.1f}",
                f"%{m3['signal_scores']['precision']*100:.1f}",
                f"%{m3['signal_scores']['recall']*100:.1f}",
                f"%{m3['signal_scores']['f1']*100:.1f}",
            ]
        })
        st.dataframe(df_m3, use_container_width=True, hide_index=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center;color:#8892b0;font-size:0.8rem;padding:2rem 0;">'
    f'Son güncelleme: {report["metadata"]["created_at"]} | '
    f'Eğitim aralığı: {report["metadata"]["train_range"]} | '
    f'Sinyal eşiği: {report["metadata"]["signal_threshold"]}'
    f'</div>',
    unsafe_allow_html=True
)
