import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
COLORS = {
    'up': '#00ff99',
    'down': '#ff4040',
    'ma20': '#ffaa00',
    'ma50': '#00aaff',
    'ma200': '#ff00ff',
    'volume': '#4a5568',
    'background': '#05070d',
    'card_bg': 'linear-gradient(145deg,#0b111c,#0d1524)',
    'grid': '#151a23'
}

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_21",
    "mom_63", "mom_126",
    "vol_21", "vol_63",
    "ma_ratio_21_63",
    "drawdown_63"
]

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.features import add_features
from src.config import UNIVERSE

st.set_page_config(layout="wide", page_title="AI Trading Dashboard")

# =========================
# STRATEGY DETECTION
# =========================
def detect_strategy_type(model_dir: Path) -> str:
    """
    Try to detect strategy type from model metadata or config
    Returns 'MOMENTUM' or 'MEAN_REVERSION'
    """
    # Check if there's a config file or metadata
    config_file = model_dir / "model_config.txt"
    if config_file.exists():
        with open(config_file, 'r') as f:
            content = f.read()
            if "MEAN_REVERSION" in content:
                return "MEAN_REVERSION"

    # Default to MOMENTUM
    return "MOMENTUM"

# =========================
# CACHING FUNCTIONS
# =========================
@st.cache_resource
def load_model_and_metadata(model_path):
    """Load and cache the ML model with metadata"""
    try:
        model = joblib.load(model_path)

        # Try to load feature importance if available
        model_dir = Path(model_path).parent
        fi_path = model_dir.parent / "results" / "feature_importance.csv"

        feature_importance = None
        if fi_path.exists():
            feature_importance = pd.read_csv(fi_path)

        # Detect strategy type
        strategy_type = detect_strategy_type(model_dir)

        return model, feature_importance, strategy_type
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        return None, None, "MOMENTUM"

@st.cache_data(ttl=3600)
def load_and_process_data(ticker, base_dir):
    """Load and process data with caching (1 hour TTL)"""
    try:
        data_path = os.path.join(base_dir, f"data/raw/D1/{ticker}.US_D1.csv")
        df = pd.read_csv(data_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["ticker"] = ticker

        # Add features
        df_feat = add_features(df).dropna().copy()

        # Add technical indicators
        df_feat["MA50"] = df_feat["close"].rolling(50).mean()
        df_feat["MA200"] = df_feat["close"].rolling(200).mean()
        df_feat["volume_ma"] = df_feat["volume"].rolling(20).mean() if "volume" in df_feat.columns else None

        return df, df_feat
    except FileNotFoundError:
        st.error(f"❌ Data file not found for {ticker}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None, None

def calculate_risk_metrics(df_feat):
    """Calculate risk metrics"""
    returns = df_feat["ret_1"].dropna()

    metrics = {
        "volatility": returns.std() * np.sqrt(252) * 100,
        "sharpe": (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
        "max_drawdown": df_feat["drawdown_63"].iloc[-1] * 100 if "drawdown_63" in df_feat.columns else 0,
        "current_vol": df_feat["vol_21"].iloc[-1] * 100 if "vol_21" in df_feat.columns else 0
    }

    return metrics

# =========================
# PREDICTION LOGIC
# =========================
def generate_predictions(model, df_feat, strategy_type):
    """
    Generate predictions based on strategy type

    For MOMENTUM:
        - High probability = Stock will go UP (buy signal)
        - Score interpretation: Higher is better

    For MEAN_REVERSION:
        - High probability = Stock is OVERSOLD (buy signal)
        - Score interpretation: Look for extreme values
    """
    X_latest = df_feat[FEATURE_COLS].tail(1)

    # Validate features
    if X_latest.isnull().any().any():
        raise ValueError("Missing values in features")

    # Get probability of UP class
    proba_up = model.predict_proba(X_latest)[0, 1]

    if strategy_type == "MOMENTUM":
        # For momentum: high prob = buy signal
        score_base = float(proba_up)
    else:  # MEAN_REVERSION
        # For mean reversion: we want extremes
        # Low prob (oversold) = buy signal
        # High prob (overbought) = sell signal
        # Convert to "buy signal strength"
        score_base = float(1 - proba_up)

    return score_base, proba_up

# =========================
# HORIZON SCORING
# =========================
def calculate_horizon_scores(score_base, df_feat, strategy_type):
    """
    Calculate horizon scores with trend calibration

    Adjusts base score based on momentum and volatility
    """
    # Get calibration factors
    mom_126 = df_feat["mom_126"].iloc[-1]
    vol_63 = df_feat["vol_63"].iloc[-1]

    # Momentum adjustment (stronger effect for longer horizons)
    mom_adj = np.tanh(mom_126) * 0.04

    # Volatility penalty (higher vol = more uncertainty)
    vol_adj = np.tanh(vol_63) * 0.02

    def horizon_score(multiplier):
        adjusted = score_base + mom_adj * multiplier - vol_adj * multiplier
        return float(np.clip(adjusted, 0, 1))

    horizons = {
        "1 Month": horizon_score(0.5),
        "3 Months": horizon_score(0.8),
        "6 Months": horizon_score(1.0),
        "12 Months": horizon_score(1.2)
    }

    return horizons

# =========================
# UI STYLING
# =========================
st.markdown(f"""
<style>
.stApp {{ background-color:{COLORS['background']}; }}

.block-container {{
    max-width:1600px;
    padding-top:1rem;
}}

/* HEADER */
.price-wrapper {{ text-align:center; margin-top:10px; }}
.price {{
    font-size:52px;
    font-weight:800;
    letter-spacing:0.5px;
}}
.price-pill {{
    padding:6px 14px;
    border-radius:999px;
    font-weight:700;
    margin-left:12px;
    font-size:14px;
}}
.green-pill {{
    background:rgba(0,255,120,0.15);
    color:{COLORS['up']};
    border:1px solid rgba(0,255,120,0.4);
}}
.red-pill {{
    background:rgba(255,60,60,0.15);
    color:{COLORS['down']};
    border:1px solid rgba(255,60,60,0.4);
}}

.strategy-badge {{
    display:inline-block;
    padding:4px 12px;
    border-radius:999px;
    font-size:11px;
    font-weight:700;
    letter-spacing:1px;
    margin-left:10px;
    background:rgba(100,150,255,0.15);
    color:#64a0ff;
    border:1px solid rgba(100,150,255,0.4);
}}

.section-label {{
    font-size:11px;
    letter-spacing:1.5px;
    opacity:0.4;
    margin-top:18px;
    text-transform:uppercase;
}}

/* CARDS */
.card {{
    background:{COLORS['card_bg']};
    border-radius:16px;
    padding:22px;
    transition:0.3s ease;
    position:relative;
    overflow:hidden;
    min-height:190px;
}}

.card.up {{
    border:1px solid rgba(0,255,120,0.6);
    box-shadow:0 0 25px rgba(0,255,120,0.25);
}}

.card.down {{
    border:1px solid rgba(255,60,60,0.6);
    box-shadow:0 0 25px rgba(255,60,60,0.25);
}}

.card.highlight {{
    transform:scale(1.05);
}}

.card:hover {{
    transform:translateY(-4px);
}}

.card-title {{
    font-size:12px;
    opacity:0.5;
    text-transform:uppercase;
}}

.card-direction {{
    font-size:26px;
    font-weight:800;
    margin-top:6px;
}}

.up-text {{ color:{COLORS['up']}; }}
.down-text {{ color:{COLORS['down']}; }}

.card-metrics {{
    margin-top:10px;
    font-size:15px;
    font-weight:700;
}}

.card-meta {{
    margin-top:4px;
    font-size:11px;
    opacity:0.5;
}}

.conf-bar {{
    height:8px;
    border-radius:999px;
    background:rgba(255,255,255,0.08);
    margin-top:14px;
    overflow:hidden;
}}

.conf-fill-up {{
    height:100%;
    background:linear-gradient(90deg,#003d1f,{COLORS['up']});
}}

.conf-fill-down {{
    height:100%;
    background:linear-gradient(90deg,#4a0000,{COLORS['down']});
}}

/* RISK METRICS */
.risk-metric {{
    background:rgba(255,255,255,0.03);
    border-radius:12px;
    padding:16px;
    margin:8px 0;
}}

.risk-label {{
    font-size:11px;
    opacity:0.5;
    text-transform:uppercase;
}}

.risk-value {{
    font-size:22px;
    font-weight:700;
    margin-top:4px;
}}

hr {{
    border:none;
    height:1px;
    background:{COLORS['grid']};
    margin:20px 0;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Dashboard Controls")
ticker = st.sidebar.selectbox("Select Stock", sorted(UNIVERSE), index=sorted(UNIVERSE).index("AAPL") if "AAPL" in UNIVERSE else 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Chart Settings")
view_days = st.sidebar.slider("Chart Period (days)", 60, 2000, 600)

show_ma50 = st.sidebar.checkbox("Show MA50", value=True)
show_ma200 = st.sidebar.checkbox("Show MA200", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)

# =========================
# LOAD DATA & MODEL
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "models", "long_model.pkl")
model, feature_importance, strategy_type = load_model_and_metadata(MODEL_PATH)

if model is None:
    st.stop()

# Display strategy info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")

strategy_color = "🎯" if strategy_type == "MOMENTUM" else "🔄"
st.sidebar.info(f"""
**Strategy**: {strategy_color} {strategy_type}
**Features**: 9 technical indicators
**Output**: Buy/Sell signal strength

{strategy_type} strategy: {'Buy winners, sell losers' if strategy_type == 'MOMENTUM' else 'Buy losers, sell winners'}
""")

# Show feature importance if available
if feature_importance is not None:
    with st.sidebar.expander("📊 Feature Importance"):
        for _, row in feature_importance.head(5).iterrows():
            st.text(f"{row['feature']}: {row['importance']:.0f}")

df, df_feat = load_and_process_data(ticker, BASE_DIR)

if df is None or df_feat is None:
    st.stop()

if len(df_feat) < 200:
    st.warning("⚠️ Insufficient data for reliable analysis")

# =========================
# GENERATE PREDICTIONS
# =========================
try:
    score_base, proba_up = generate_predictions(model, df_feat, strategy_type)
    horizons = calculate_horizon_scores(score_base, df_feat, strategy_type)

except Exception as e:
    st.error(f"❌ Prediction error: {str(e)}")
    st.stop()

# Strongest conviction
conv_map = {k: abs(v - 0.5) * 2 for k, v in horizons.items()}
strongest = max(conv_map, key=conv_map.get)

# =========================
# CALCULATE RISK METRICS
# =========================
risk_metrics = calculate_risk_metrics(df_feat)

# =========================
# HEADER
# =========================
latest = df.iloc[-1]
prev = df.iloc[-2]

price = latest["close"]
change = price - prev["close"]
change_pct = change / prev["close"] * 100

pill_class = "green-pill" if change >= 0 else "red-pill"

st.markdown(f"""
<div class="price-wrapper">
<span class="price">{ticker} ${price:,.2f}</span>
<span class="price-pill {pill_class}">
{change:+.2f} ({change_pct:+.2f}%)
</span>
<span class="strategy-badge">{strategy_type}</span>
</div>
""", unsafe_allow_html=True)

# =========================
# MODEL CONFIDENCE INDICATOR
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 2, 1])

with col_b:
    confidence_score = abs(score_base - 0.5) * 2
    signal_strength = "STRONG" if confidence_score > 0.5 else "MODERATE" if confidence_score > 0.3 else "WEAK"
    signal_color = COLORS['up'] if score_base > 0.5 else COLORS['down']

    st.markdown(f"""
    <div style="text-align:center; padding:20px; background:rgba(255,255,255,0.03); border-radius:12px;">
        <div style="font-size:11px; opacity:0.5; text-transform:uppercase; letter-spacing:1.5px;">Current Signal</div>
        <div style="font-size:32px; font-weight:800; color:{signal_color}; margin-top:8px;">
            {signal_strength} {'BUY' if score_base > 0.5 else 'SELL'}
        </div>
        <div style="font-size:14px; opacity:0.7; margin-top:8px;">
            Model Confidence: {confidence_score*100:.1f}% • Raw Score: {score_base*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# RISK METRICS ROW
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Risk Metrics</div>', unsafe_allow_html=True)

risk_cols = st.columns(4)

risk_data = [
    ("Volatility (Ann.)", f"{risk_metrics['volatility']:.1f}%", "📊"),
    ("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}", "📈"),
    ("Max Drawdown", f"{risk_metrics['max_drawdown']:.1f}%", "📉"),
    ("Current Vol", f"{risk_metrics['current_vol']:.1f}%", "🌊")
]

for i, (label, value, icon) in enumerate(risk_data):
    with risk_cols[i]:
        st.markdown(f"""
        <div class="risk-metric">
            <div class="risk-label">{icon} {label}</div>
            <div class="risk-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# SIGNAL CARDS
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Signal Dashboard - Time Horizons</div>', unsafe_allow_html=True)

cols = st.columns(4)

for i, (title, score) in enumerate(horizons.items()):
    direction = "UP" if score > 0.5 else "DOWN"
    conviction = abs(score - 0.5) * 2
    conv_pct = int(round(conviction * 100))
    score_pct = int(round(score * 100))

    card_class = "up" if direction == "UP" else "down"
    text_class = "up-text" if direction == "UP" else "down-text"
    fill_class = "conf-fill-up" if direction == "UP" else "conf-fill-down"
    highlight_class = "highlight" if title == strongest else ""

    arrow = "↗" if direction == "UP" else "↘"

    html = f"""<div class="card {card_class} {highlight_class}">
<div class="card-title">{title} {arrow}</div>
<div class="card-direction {text_class}">{direction}</div>
<div class="card-metrics">Score: {score_pct}%</div>
<div class="card-meta">Conviction: {conv_pct}%</div>
<div class="conf-bar">
<div class="{fill_class}" style="width:{conv_pct}%"></div>
</div>
</div>"""

    with cols[i]:
        st.markdown(html, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# CHART
# =========================
df_plot = df.tail(view_days).copy()
df_plot["MA20"] = df_plot["close"].rolling(20).mean()

# Create subplots with volume if requested
if show_volume and "volume" in df_plot.columns:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume")
    )
    has_volume = True
else:
    fig = go.Figure()
    has_volume = False

# Candlestick
candlestick = go.Candlestick(
    x=df_plot["datetime"],
    open=df_plot["open"],
    high=df_plot["high"],
    low=df_plot["low"],
    close=df_plot["close"],
    increasing_line_color=COLORS['up'],
    decreasing_line_color=COLORS['down'],
    name="Price"
)

if has_volume:
    fig.add_trace(candlestick, row=1, col=1)
else:
    fig.add_trace(candlestick)

# MA20
ma20_trace = go.Scatter(
    x=df_plot["datetime"],
    y=df_plot["MA20"],
    line=dict(color=COLORS['ma20'], width=2),
    name="MA20"
)

if has_volume:
    fig.add_trace(ma20_trace, row=1, col=1)
else:
    fig.add_trace(ma20_trace)

# MA50
if show_ma50 and "MA50" in df_feat.columns:
    df_plot["MA50"] = df_feat["MA50"].tail(view_days).values
    ma50_trace = go.Scatter(
        x=df_plot["datetime"],
        y=df_plot["MA50"],
        line=dict(color=COLORS['ma50'], width=2, dash='dash'),
        name="MA50"
    )
    if has_volume:
        fig.add_trace(ma50_trace, row=1, col=1)
    else:
        fig.add_trace(ma50_trace)

# MA200
if show_ma200 and "MA200" in df_feat.columns:
    df_plot["MA200"] = df_feat["MA200"].tail(view_days).values
    ma200_trace = go.Scatter(
        x=df_plot["datetime"],
        y=df_plot["MA200"],
        line=dict(color=COLORS['ma200'], width=2, dash='dot'),
        name="MA200"
    )
    if has_volume:
        fig.add_trace(ma200_trace, row=1, col=1)
    else:
        fig.add_trace(ma200_trace)

# Volume bars
if has_volume:
    colors = [COLORS['up'] if df_plot.iloc[i]['close'] >= df_plot.iloc[i]['open']
              else COLORS['down'] for i in range(len(df_plot))]

    fig.add_trace(
        go.Bar(
            x=df_plot["datetime"],
            y=df_plot["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.5
        ),
        row=2, col=1
    )

# Projection arrows
last_price = df_plot["close"].iloc[-1]
last_date = df_plot["datetime"].iloc[-1]

for idx, (title, score) in enumerate(horizons.items()):
    direction = 1 if score > 0.5 else -1
    slope = (score - 0.5) * 0.4 * direction

    target_date = last_date + pd.Timedelta(days=30 * (idx + 1))
    target_price = last_price * (1 + slope)

    annotation = dict(
        x=target_date,
        y=target_price,
        ax=last_date,
        ay=last_price,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowcolor=COLORS['up'] if direction > 0 else COLORS['down'],
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        text=title.split()[0],
        font=dict(size=10, color='white'),
        bgcolor='rgba(0,0,0,0.7)',
        borderpad=4
    )

    if has_volume:
        annotation['xref'] = 'x'
        annotation['yref'] = 'y'

    fig.add_annotation(annotation)

# Layout
layout_args = dict(
    height=750 if has_volume else 650,
    paper_bgcolor=COLORS['background'],
    plot_bgcolor=COLORS['background'],
    font=dict(color="white"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if has_volume:
    layout_args['xaxis2'] = dict(showgrid=False)
    layout_args['yaxis2'] = dict(showgrid=True, gridcolor=COLORS['grid'])
    layout_args['xaxis_rangeslider_visible'] = False
else:
    layout_args['xaxis_rangeslider_visible'] = False

fig.update_layout(**layout_args)

st.plotly_chart(fig, use_container_width=True)

# =========================
# FEATURE VALUES & ANALYSIS
# =========================
with st.expander("📋 Model Features & Latest Values"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Values")
        feat_data = []
        for feat in FEATURE_COLS:
            value = df_feat[feat].iloc[-1]
            feat_data.append({"Feature": feat, "Value": f"{value:.4f}"})

        feat_df = pd.DataFrame(feat_data)
        st.dataframe(feat_df, use_container_width=True)

    with col2:
        st.subheader("Signal Interpretation")
        st.markdown(f"""
        **Strategy Type**: {strategy_type}

        **Model Output**:
        - Raw probability (UP class): {proba_up*100:.1f}%
        - Adjusted score: {score_base*100:.1f}%

        **What this means**:
        """)

        if strategy_type == "MOMENTUM":
            st.markdown(f"""
            - Score > 50%: **Buy signal** (momentum continuing)
            - Score < 50%: **Sell signal** (momentum reversing)
            - Current: **{score_base*100:.1f}%** → {'✅ BUY' if score_base > 0.5 else '❌ SELL'}
            """)
        else:  # MEAN_REVERSION
            st.markdown(f"""
            - Score > 50%: **Buy signal** (oversold, likely to bounce)
            - Score < 50%: **Sell signal** (overbought, likely to drop)
            - Current: **{score_base*100:.1f}%** → {'✅ BUY' if score_base > 0.5 else '❌ SELL'}
            """)

# =========================
# EXPORT FUNCTIONALITY
# =========================
with st.expander("💾 Export Data"):
    csv = df_feat.tail(100).to_csv(index=False)
    st.download_button(
        label="Download Last 100 Days (CSV)",
        data=csv,
        file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # Prediction summary
    pred_summary = {
        "Ticker": ticker,
        "Date": latest["datetime"].strftime("%Y-%m-%d"),
        "Price": price,
        "Change": change,
        "Change%": change_pct,
        "Strategy": strategy_type,
        "Model_Score": score_base,
        "Signal": "BUY" if score_base > 0.5 else "SELL",
        "Confidence": abs(score_base - 0.5) * 2,
        **{f"{k}_Score": v for k, v in horizons.items()},
        **{f"Risk_{k}": v for k, v in risk_metrics.items()}
    }

    pred_df = pd.DataFrame([pred_summary])
    pred_csv = pred_df.to_csv(index=False)

    st.download_button(
        label="Download Prediction Summary (CSV)",
        data=pred_csv,
        file_name=f"{ticker}_prediction_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# =========================
# FOOTER
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; opacity:0.4; font-size:11px;">
AI Trading Dashboard • {strategy_type} Strategy • Model predictions are for informational purposes only • Not financial advice
</div>
""", unsafe_allow_html=True)
