"""
src/short.py — Kısa Vadeli Model Dashboard
============================================
Seçilen hisse için kısa vadeli tahmin dashboard'unu render eder.
Analysis.py'den render_short_dashboard(ticker) olarak çağrılır.
"""
import warnings
import pickle
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# Proje kökünü sys.path'e ekle
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Kısa vadeli model kaynak modüllerini path'e ekle
_SHORT_MODEL_DIR = _PROJECT_ROOT / "models" / "short_term" / "src"
if str(_SHORT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_SHORT_MODEL_DIR))

from config import TICKERS, TICKER_TO_NAME, HORIZONS, to_yf_symbol
from features import build_features, get_feature_cols


# ── CSS (Analysis.py genel stilini bozmamak için scope'lu)
_SHORT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

.price-big{font-family:'DM Mono',monospace;font-size:2.6rem;font-weight:500;
           color:#e8edf5;letter-spacing:-1px;}
.chg-pos{font-family:'DM Mono',monospace;color:#00c47a;
         background:rgba(0,196,122,.1);padding:3px 12px;border-radius:20px;font-size:.95rem;}
.chg-neg{font-family:'DM Mono',monospace;color:#ff4455;
         background:rgba(255,68,85,.1);padding:3px 12px;border-radius:20px;font-size:.95rem;}
.price-meta{font-size:.76rem;color:#2a3a4c;margin:4px 0 16px;}
.stat-row{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;}
.stat-item{background:#0f1520;border:1px solid #1a2535;border-radius:6px;
           padding:5px 14px;font-family:'DM Mono',monospace;font-size:.73rem;color:#3d4a5c;}

.pred-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:32px;}
.pred-card{border-radius:12px;padding:20px;border:1px solid;
           position:relative;overflow:hidden;}
.pred-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.card-up  {background:#02120a;border-color:rgba(0,196,122,.2);}
.card-up::before{background:linear-gradient(90deg,#00c47a,#34d9a5);}
.card-down{background:#120205;border-color:rgba(255,68,85,.2);}
.card-down::before{background:linear-gradient(90deg,#ff4455,#ff7080);}
.card-none{background:#0b0e17;border-color:#1a2030;}

.c-horizon{font-family:'DM Mono',monospace;font-size:.66rem;color:#253545;
           text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;}
.c-signal{font-size:1.7rem;font-weight:700;line-height:1;margin-bottom:8px;}
.sig-up{color:#00c47a;} .sig-down{color:#ff4455;}
.c-prob-num{font-family:'DM Mono',monospace;font-size:1.1rem;
            font-weight:600;margin-bottom:6px;}
.bar-bg{height:5px;border-radius:3px;background:#161b28;margin:8px 0 6px;overflow:hidden;}
.bar-up  {height:100%;border-radius:3px;
          background:linear-gradient(90deg,#00c47a,#34d9a580);}
.bar-down{height:100%;border-radius:3px;
          background:linear-gradient(90deg,#ff4455,#ff708080);}
.c-sub{font-size:.72rem;color:#253545;}
.votes{display:flex;gap:5px;margin-top:10px;align-items:center;}
.vote{width:8px;height:8px;border-radius:50%;}
.vote-label{font-size:.68rem;color:#253545;margin-left:4px;}

.sec-title{font-size:.64rem;letter-spacing:3px;text-transform:uppercase;
           color:#1a2838;margin-bottom:16px;padding-bottom:10px;
           border-bottom:1px solid #0f1520;}
</style>
"""


def _rgba(hex6: str, a: float) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"


# ─────────────────────────────────────────────
# VERİ
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live(ticker: str) -> pd.DataFrame | None:
    """yfinance — son 3 ay (tahmin için yeterli geçmiş lazım)"""
    try:
        sym = to_yf_symbol(ticker)
        df = yf.download(sym, period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_yf_history(ticker: str) -> pd.DataFrame | None:
    """yfinance — son 2 yıl günlük veri (özellik hesabı için).
    TTL=86400 → günlük önbellek, yerel dosyaya gerek yok."""
    try:
        sym = to_yf_symbol(ticker)
        df = yf.download(sym, period="2y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
        return df if len(df) >= 200 else None
    except Exception:
        return None


def _load_bundle(ticker, h, algo):
    p = _PROJECT_ROOT / "models" / ticker / f"{algo}_{h}d.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


@st.cache_data(ttl=300, show_spinner=False)
def _run_prediction(ticker: str) -> dict:
    """yFinance geçmiş verisi üzerinde tahmin üretir.
    Son satır = bugünün tahmini."""
    raw = _fetch_yf_history(ticker)
    if raw is None:
        return {}

    full = build_features(raw)
    feat_cols = get_feature_cols(full)

    # Son satırda NaN varsa geri git
    anchor = full.index[-1]
    for anchor in reversed(full.index):
        row = full.loc[[anchor], feat_cols]
        if not row.isna().any(axis=1).values[0]:
            break

    out = {"anchor": str(anchor.date())}

    for h in HORIZONS:
        hk, probs = f"{h}d", {}
        for algo in ["rf", "lgbm", "xgb"]:
            b = _load_bundle(ticker, h, algo)
            if b is None:
                continue
            mc = b.get("feat_cols", feat_cols)
            thr = b.get("threshold", 0.5)
            try:
                X = full.loc[[anchor], mc].values.astype(np.float32)
                if algo == "rf":
                    pr = float(b["model"].predict_proba(X)[0, 1])
                elif algo == "lgbm":
                    pr = float(b["model"].predict(X)[0])
                elif algo == "xgb":
                    pr = float(b["model"].predict_proba(X)[0, 1])
                else:
                    pr = float(b["model"].predict_proba(
                        b["scaler"].transform(X))[0, 1])
                probs[algo] = {"prob": round(pr, 4), "thr": thr}
            except Exception:
                continue

        if not probs:
            continue

        avg = float(np.mean([v["prob"] for v in probs.values()]))
        avg_thr = float(np.mean([v["thr"] for v in probs.values()]))
        sig = "YÜKSELİŞ" if avg >= avg_thr else "DÜŞÜŞ"
        conf = avg if sig == "YÜKSELİŞ" else 1 - avg

        out[hk] = {
            "signal":     sig,
            "confidence": round(conf, 4),
            "prob_up":    round(avg, 4),
            "votes_up":   sum(1 for v in probs.values() if v["prob"] >= v["thr"]),
            "n_algos":    len(probs),
            "algo_probs": {k: v["prob"] for k, v in probs.items()},
        }
    return out


# ─────────────────────────────────────────────
# ANA RENDER FONKSİYONU
# ─────────────────────────────────────────────
H_LABELS = {"1d": "1 Gün", "3d": "3 Gün", "5d": "5 Gün", "7d": "7 Gün"}
UP_C = "#00c47a"
DOWN_C = "#ff4455"


def render_short_dashboard(selected_ticker: str) -> None:
    """Analysis.py'den çağrılır. Seçilen hissenin kısa vadeli modelini gösterir."""

    st.markdown(_SHORT_CSS, unsafe_allow_html=True)

    cname = TICKER_TO_NAME.get(selected_ticker, selected_ticker)

    # ── Model / veri durumu
    model_dir = _PROJECT_ROOT / "models" / selected_ticker
    model_ok = model_dir.exists() and any(model_dir.glob("*.pkl"))

    if model_ok:
        n = len(list(model_dir.glob("*.pkl")))
        st.success(f"✓ {selected_ticker}  ({n} model yüklendi)")
    else:
        st.warning(
            f"⚠ {selected_ticker} için eğitilmiş model bulunamadı.\n\n"
            f"```\npoetry run python train.py --ticker {selected_ticker}\n```"
        )

    # ── Canlı fiyat başlığı
    live_df = _fetch_live(selected_ticker)

    if live_df is not None and len(live_df) >= 2:
        lc = float(live_df["Close"].iloc[-1])
        pc = float(live_df["Close"].iloc[-2])
        ch = lc - pc
        cp = ch / pc * 100
        sign = "+" if ch >= 0 else ""
        cls = "chg-pos" if ch >= 0 else "chg-neg"
        ld = live_df.index[-1].date()
        hi = float(live_df["High"].max())
        lo = float(live_df["Low"].min())
        vol = int(live_df["Volume"].iloc[-1])

        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:4px'>"
            f"<span class='price-big'>${lc:,.2f}</span>"
            f"<span class='{cls}'>{sign}{ch:.2f} ({sign}{cp:.2f}%)</span>"
            f"</div>"
            f"<div class='price-meta'>{selected_ticker} · Son kapanış {ld} · Yahoo Finance</div>"
            f"<div class='stat-row'>"
            f"<span class='stat-item'>3A Yüksek &nbsp;${hi:,.2f}</span>"
            f"<span class='stat-item'>3A Düşük &nbsp;${lo:,.2f}</span>"
            f"<span class='stat-item'>Son Hacim &nbsp;{vol:,}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        ref_price = lc
        ref_date = live_df.index[-1]
    else:
        st.warning(f"`{selected_ticker}` canlı fiyat yüklenemedi.")
        ref_price = None
        ref_date = pd.Timestamp(date.today())

    # ── Tahmin
    with st.spinner("Tahmin hesaplanıyor..."):
        preds = _run_prediction(selected_ticker)

    anchor_str = preds.get("anchor", "—")

    # ── Tahmin kartları
    st.markdown(
        f"<div class='sec-title'>Kısa Vadeli Tahminler"
        f"<span style='float:right;font-size:.65rem;color:#1e2a38'>"
        f"Model veri tarihi: {anchor_str}</span></div>",
        unsafe_allow_html=True)

    cards = "<div class='pred-grid'>"
    for hk in ["1d", "3d", "5d", "7d"]:
        if hk not in preds:
            cards += (
                f"<div class='pred-card card-none'>"
                f"<div class='c-horizon'>{H_LABELS[hk]}</div>"
                f"<div class='c-signal' style='color:#1e2a3a'>— —</div>"
                f"<div class='c-sub'>model yok</div></div>"
            )
            continue

        p = preds[hk]
        sig = p["signal"]
        conf = p["confidence"]
        vu = p["votes_up"]
        na = p["n_algos"]

        is_up = sig == "YÜKSELİŞ"
        cc = "card-up" if is_up else "card-down"
        sc = "sig-up" if is_up else "sig-down"
        ar = "↑" if is_up else "↓"
        bc = "bar-up" if is_up else "bar-down"

        dots = "".join(
            f"<span class='vote' style='background:"
            f"{'#00c47a' if (is_up and i < vu) or (not is_up and i < (na - vu)) else '#ff4455'}'></span>"
            for i in range(na)
        )
        algo_names = {"lgbm": "LG", "xgb": "XG", "rf": "RF", "logreg": "LR"}
        vote_detail = " · ".join(
            f"<span style='color:{'#00c47a' if v >= p.get('threshold', 0.5) else '#ff4455'}'>"
            f"{algo_names.get(a, a)}</span>"
            for a, v in p["algo_probs"].items()
        )

        cards += (
            f"<div class='pred-card {cc}'>"
            f"<div class='c-horizon'>{H_LABELS[hk]}</div>"
            f"<div class='c-signal {sc}'>{ar} {sig}</div>"
            f"<div class='c-prob-num' style='color:{'#00c47a' if is_up else '#ff4455'}'>"
            f"%{conf * 100:.0f}</div>"
            f"<div class='bar-bg'><div class='{bc}' style='width:{conf * 100:.0f}%'></div></div>"
            f"<div class='c-sub'>güven skoru</div>"
            f"<div class='votes'>{dots}"
            f"<span class='vote-label'>{vote_detail}</span></div>"
            f"</div>"
        )
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    # ── Grafik — yfinance 1 ay + tahmin okları
    st.markdown("<div class='sec-title'>Son 1 Ay Fiyat Hareketi &amp; Tahmin Yönleri</div>",
                unsafe_allow_html=True)

    if live_df is not None and not live_df.empty:
        chart_df = live_df.tail(22).copy()
    else:
        chart_df = None

    fig = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22],
                        shared_xaxes=True, vertical_spacing=0.03)

    if chart_df is not None and not chart_df.empty:
        # Mum grafiği
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"], close=chart_df["Close"],
            name="Fiyat",
            increasing_line_color=UP_C, decreasing_line_color=DOWN_C,
            increasing_fillcolor=_rgba(UP_C, 0.2),
            decreasing_fillcolor=_rgba(DOWN_C, 0.2),
        ), row=1, col=1)

        # MA20
        ma = chart_df["Close"].rolling(20, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=ma, name="MA20",
            line=dict(color="#2a3a4c", width=1.2, dash="dot"),
            hoverinfo="skip"), row=1, col=1)

        # Hacim
        vc = [_rgba(UP_C, .15) if c >= o else _rgba(DOWN_C, .15)
              for c, o in zip(chart_df["Close"], chart_df["Open"])]
        fig.add_trace(go.Bar(
            x=chart_df.index, y=chart_df["Volume"],
            marker_color=vc, showlegend=False), row=2, col=1)

        # ── Tahmin okları
        if ref_price and preds:
            last_date = chart_df.index[-1]
            COLORS = {"1d": UP_C, "3d": "#34d9a5", "5d": "#f0c040", "7d": "#a78bfa"}
            offsets = {"1d": 1, "3d": 3, "5d": 5, "7d": 7}

            for hk in ["1d", "3d", "5d", "7d"]:
                if hk not in preds:
                    continue
                p = preds[hk]
                col = COLORS[hk]
                dire = 1 if p["signal"] == "YÜKSELİŞ" else -1
                conf = p["confidence"]

                target_x = last_date + pd.tseries.offsets.BDay(offsets[hk])
                target_y = ref_price * (1 + dire * conf * 0.03)

                fig.add_annotation(
                    x=target_x, y=target_y,
                    ax=last_date, ay=ref_price,
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=3, arrowwidth=2.5,
                    arrowcolor=col, showarrow=True,
                )
                label_txt = "YÜKSELİŞ" if p["signal"] == "YÜKSELİŞ" else "DÜŞÜŞ"
                fig.add_annotation(
                    x=target_x,
                    y=target_y * (1 + dire * 0.005),
                    text=f"<b>{hk} {label_txt}</b>", showarrow=False,
                    font=dict(color=col, size=10, family="DM Mono"),
                    bgcolor="#07090f", borderpad=3,
                )

            fig.add_vline(x=last_date,
                          line_color=_rgba("#ffffff", 0.07), line_width=1)
            fig.add_annotation(
                x=last_date, y=1, yref="paper",
                text="<b>Bugün</b>",
                showarrow=False, xanchor="right", yanchor="top",
                font=dict(color="#253545", size=10, family="DM Mono"),
            )

    fig.update_layout(
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font=dict(color="#3d4a5c", family="DM Sans"),
        xaxis_rangeslider_visible=False,
        height=500, margin=dict(l=4, r=4, t=8, b=4),
        legend=dict(bgcolor="#0b0e17", bordercolor="#161b28", borderwidth=1,
                    orientation="h", y=1.04, x=0, font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0b0e17", bordercolor="#161b28"),
        xaxis=dict(
            range=[chart_df.index[0] if chart_df is not None else date.today(),
                   (chart_df.index[-1] if chart_df is not None
                    else pd.Timestamp(date.today())) + pd.tseries.offsets.BDay(10)],
            gridcolor="#0c1018", zeroline=False,
            showspikes=True, spikecolor="#1e2a3a",
            spikemode="across", spikethickness=1,
            tickfont=dict(size=10),
        ),
    )
    fig.update_yaxes(gridcolor="#0c1018", zeroline=False,
                     tickprefix="$", tickfont=dict(size=10))
    fig.update_xaxes(gridcolor="#0c1018", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Olasılık çubukları
    st.markdown("<div class='sec-title'>Ufuk Bazlı Güven Skoru</div>",
                unsafe_allow_html=True)

    prob_rows = []
    for hk in ["1d", "3d", "5d", "7d"]:
        if hk not in preds:
            continue
        p = preds[hk]
        prob_rows.append({
            "h":       H_LABELS[hk],
            "YUKSELIS": round(p["prob_up"] * 100, 1),
            "DUSUS":    round((1 - p["prob_up"]) * 100, 1),
            "sig":     p["signal"],
        })

    if prob_rows:
        pd2 = pd.DataFrame(prob_rows)
        f2 = go.Figure()
        f2.add_trace(go.Bar(
            name="↑ YÜKSELİŞ", x=pd2["h"], y=pd2["YUKSELIS"],
            marker_color=UP_C,
            text=[f"%{v:.0f}" for v in pd2["YUKSELIS"]],
            textposition="inside",
            textfont=dict(color="white", size=14, family="DM Mono")))
        f2.add_trace(go.Bar(
            name="↓ DÜŞÜŞ", x=pd2["h"], y=pd2["DUSUS"],
            marker_color=DOWN_C,
            text=[f"%{v:.0f}" for v in pd2["DUSUS"]],
            textposition="inside",
            textfont=dict(color="white", size=14, family="DM Mono")))
        f2.add_hline(y=50, line_dash="dot", line_color="#1a2535", line_width=1.5)
        f2.update_layout(
            barmode="stack", paper_bgcolor="#07090f", plot_bgcolor="#07090f",
            font=dict(color="#3d4a5c", family="DM Sans"),
            height=190, margin=dict(l=4, r=4, t=4, b=4),
            legend=dict(bgcolor="#0b0e17", bordercolor="#161b28",
                        orientation="h", y=1.2, x=0),
            yaxis=dict(gridcolor="#0c1018", range=[0, 100],
                       ticksuffix="%", tickfont=dict(size=10)),
            xaxis=dict(gridcolor="#0c1018"), bargap=0.25,
        )
        st.plotly_chart(f2, use_container_width=True)
