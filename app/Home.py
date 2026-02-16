"""
app/Home.py — Consumer Staples Short Model
Mantık: yfinance son 1 ay → son kapanıştan tahmin okları geleceğe
Backtest yok. Tarih seçici yok. Sadece UP/DOWN + güven.
"""
import warnings, pickle, sys, json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))

from config import TICKERS, TICKER_TO_NAME, DATA_DIR, HORIZONS, to_yf_symbol
from features import build_features, get_feature_cols

st.set_page_config(page_title="Short Model · Consumer Staples",
                   page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#07090f;}
.stApp{background:#07090f;}
[data-testid="stSidebar"]{background:#0b0e17!important;border-right:1px solid #161b28;}

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
""", unsafe_allow_html=True)


def rgba(hex6: str, a: float) -> str:
    h = hex6.lstrip("#")
    r,g,b = int(h[:2],16),int(h[2:4],16),int(h[4:],16)
    return f"rgba({r},{g},{b},{a})"


# ─────────────────────────────────────────────
# VERİ
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live(ticker: str) -> pd.DataFrame | None:
    """yfinance — son 3 ay (tahmin için yeterli geçmiş lazım)"""
    try:
        sym = to_yf_symbol(ticker)
        df  = yf.download(sym, period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_kaggle(ticker: str) -> pd.DataFrame | None:
    base = ticker.replace(".US","") if ticker.endswith(".US") else ticker
    for p in [DATA_DIR/f"{base}.US_D1.csv",
              DATA_DIR/f"{ticker}_D1.csv",
              DATA_DIR/f"{base.upper()}.US_D1.csv",
              DATA_DIR/f"{ticker}.csv",
              DATA_DIR/f"{base}.csv"]:
        if not p.exists(): continue
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip().title() for c in df.columns]
            df.rename(columns={"Adj Close":"Close","Adj_Close":"Close"}, inplace=True)
            dc = next((c for c in df.columns
                       if c.lower() in ("date","datetime","time")), df.columns[0])
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            df = df.dropna(subset=[dc]).set_index(dc).sort_index()
            df.index.name = "Date"
            cols = ["Open","High","Low","Close","Volume"]
            if not all(c in df.columns for c in cols): continue
            df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
            df = df[(df["Close"]>0)&(df["Volume"]>0)]
            if len(df) >= 200: return df
        except Exception:
            continue
    return None


def load_bundle(ticker, h, algo):
    p = Path(f"models/{ticker}/{algo}_{h}d.pkl")
    return pickle.load(open(p,"rb")) if p.exists() else None


@st.cache_data(ttl=300, show_spinner=False)
def run_prediction(ticker: str) -> dict:
    """
    En son Kaggle verisi üzerinde tahmin üret.
    Son satır = bugünün tahmini.
    """
    raw = load_kaggle(ticker)
    if raw is None: return {}

    full      = build_features(raw)
    feat_cols = get_feature_cols(full)
    anchor    = full.index[-1]           # en son geçerli gün

    # Son satırda NaN varsa geri git
    for anchor in reversed(full.index):
        row = full.loc[[anchor], feat_cols]
        if not row.isna().any(axis=1).values[0]:
            break

    out = {"anchor": str(anchor.date())}

    for h in HORIZONS:
        hk, probs = f"{h}d", {}
        for algo in ["lgbm","xgb","logreg"]:
            b = load_bundle(ticker, h, algo)
            if b is None: continue
            mc  = b.get("feat_cols", feat_cols)
            thr = b.get("threshold", 0.5)
            try:
                X = full.loc[[anchor], mc].values.astype(np.float32)
                if algo=="lgbm":   pr = float(b["model"].predict(X)[0])
                elif algo=="xgb":  pr = float(b["model"].predict_proba(X)[0,1])
                else:              pr = float(b["model"].predict_proba(
                                           b["scaler"].transform(X))[0,1])
                probs[algo] = {"prob": round(pr,4), "thr": thr}
            except Exception:
                continue

        if not probs: continue

        avg     = float(np.mean([v["prob"] for v in probs.values()]))
        avg_thr = float(np.mean([v["thr"]  for v in probs.values()]))
        sig     = "UP" if avg >= avg_thr else "DOWN"
        conf    = avg if sig=="UP" else 1-avg   # UP yönünde güven

        out[hk] = {
            "signal":     sig,
            "confidence": round(conf, 4),     # her zaman >0.5
            "prob_up":    round(avg, 4),
            "votes_up":   sum(1 for v in probs.values() if v["prob"]>=v["thr"]),
            "n_algos":    len(probs),
            "algo_probs": {k: v["prob"] for k,v in probs.items()},
        }
    return out


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Short Model")
    st.markdown("<div style='color:#1e2a38;font-size:.77rem;margin-bottom:18px'>"
                "Consumer Staples · UP / DOWN</div>", unsafe_allow_html=True)
    st.divider()

    opts   = {f"{sym}  —  {name}": sym
              for name,sym in sorted(TICKERS.items(), key=lambda x: x[1])}
    chosen = st.selectbox("Hisse", list(opts.keys()))
    ticker = opts[chosen]
    cname  = TICKER_TO_NAME.get(ticker, ticker)

    st.divider()

    kdf = load_kaggle(ticker)
    model_ok = Path(f"models/{ticker}").exists() and \
               any(Path(f"models/{ticker}").glob("*.pkl"))

    if model_ok:
        n = len(list(Path(f"models/{ticker}").glob("*.pkl")))
        st.success(f"✓ {ticker}  ({n} model)")
    else:
        st.warning(f"⚠ {ticker} modeli yok\n\n"
                   f"```\npoetry run python train.py --ticker {ticker}\n```")

    if kdf is not None:
        last_train = kdf.index[-1].date()
        st.markdown(
            f"<div style='color:#1a2530;font-size:.7rem;margin-top:12px'>"
            f"Eğitim verisi bitiş: {last_train}"
            f"<br>{len(kdf):,} işlem günü</div>",
            unsafe_allow_html=True)

    # ── Model metrikleri
    if model_ok:
        st.divider()
        st.markdown("<div style='font-size:.7rem;color:#1e2a38;margin-bottom:8px'>"
                    "<b>📊 Model Doğruluk Metrikleri</b></div>", unsafe_allow_html=True)
        
        # training_results.json'dan metrikleri çek
        try:
            with open("reports/training_results.json") as f:
                all_results = json.load(f)
            
            if ticker in all_results and all_results[ticker]:
                accs = []
                auc_vals = []
                f1_vals = []
                
                for h_key, algos in all_results[ticker].items():
                    for algo, metrics in algos.items():
                        if "accuracy" in metrics:
                            accs.append(metrics["accuracy"])
                            auc_vals.append(metrics.get("roc_auc", 0))
                            f1_vals.append(metrics.get("f1", 0))
                
                if accs:
                    avg_acc = sum(accs) / len(accs)
                    avg_auc = sum(auc_vals) / len(auc_vals)
                    avg_f1 = sum(f1_vals) / len(f1_vals)
                    
                    st.metric("Ort. Doğruluk", f"{avg_acc:.1%}")
                    st.metric("Ort. ROC-AUC", f"{avg_auc:.3f}")
                    st.metric("Ort. F1-Score", f"{avg_f1:.3f}")
                    
                    st.caption(
                        "📌 Doğruluk: Test setinde modelin %kaçını doğru tahmin ettiği\n"
                        "📌 ROC-AUC: 0.5=random, 1.0=perfect\n"
                        "📌 Yukarıdaki %X değerleri **tahmin güveni**, doğruluk değil"
                    )
        except Exception as e:
            pass


# ─────────────────────────────────────────────
# CANLI FİYAT HEADER
# ─────────────────────────────────────────────
live_df = fetch_live(ticker)

st.markdown(f"### {cname}")

if live_df is not None and len(live_df) >= 2:
    lc   = float(live_df["Close"].iloc[-1])
    pc   = float(live_df["Close"].iloc[-2])
    ch   = lc - pc
    cp   = ch / pc * 100
    sign = "+" if ch >= 0 else ""
    cls  = "chg-pos" if ch >= 0 else "chg-neg"
    ld   = live_df.index[-1].date()
    hi   = float(live_df["High"].max())
    lo   = float(live_df["Low"].min())
    vol  = int(live_df["Volume"].iloc[-1])

    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:4px'>"
        f"<span class='price-big'>${lc:,.2f}</span>"
        f"<span class='{cls}'>{sign}{ch:.2f} ({sign}{cp:.2f}%)</span>"
        f"</div>"
        f"<div class='price-meta'>{ticker} · Son kapanış {ld} · Yahoo Finance</div>"
        f"<div class='stat-row'>"
        f"<span class='stat-item'>1A Yüksek &nbsp;${hi:,.2f}</span>"
        f"<span class='stat-item'>1A Düşük &nbsp;${lo:,.2f}</span>"
        f"<span class='stat-item'>Son Hacim &nbsp;{vol:,}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    ref_price = lc
    ref_date  = live_df.index[-1]
else:
    st.warning(f"`{ticker}` canlı fiyat yüklenemedi")
    ref_price = None
    ref_date  = pd.Timestamp(date.today())


# ─────────────────────────────────────────────
# TAHMİN
# ─────────────────────────────────────────────
with st.spinner("Tahmin hesaplanıyor..."):
    preds = run_prediction(ticker)

anchor_str = preds.get("anchor", "—")
H_LABELS   = {"1d":"1 Gün","3d":"3 Gün","5d":"5 Gün","7d":"7 Gün"}

# ─────────────────────────────────────────────
# TAHMİN KARTLARI
# ─────────────────────────────────────────────
st.markdown(
    f"<div class='sec-title'>Kısa Vadeli Tahminler"
    f"<span style='float:right;font-size:.65rem;color:#1e2a38'>"
    f"Model eğitim verisi: {anchor_str}</span></div>",
    unsafe_allow_html=True)

cards = "<div class='pred-grid'>"
for hk in ["1d","3d","5d","7d"]:
    if hk not in preds:
        cards += (f"<div class='pred-card card-none'>"
                  f"<div class='c-horizon'>{H_LABELS[hk]}</div>"
                  f"<div class='c-signal' style='color:#1e2a3a'>— —</div>"
                  f"<div class='c-sub'>model yok</div></div>")
        continue

    p    = preds[hk]
    sig  = p["signal"]
    conf = p["confidence"]
    vu   = p["votes_up"]
    na   = p["n_algos"]

    cc  = "card-up" if sig=="UP" else "card-down"
    sc  = "sig-up"  if sig=="UP" else "sig-down"
    ar  = "↑"       if sig=="UP" else "↓"
    bc  = "bar-up"  if sig=="UP" else "bar-down"

    dots = "".join(
        f"<span class='vote' style='background:"
        f"{'#00c47a' if (sig=='UP' and i<vu) or (sig=='DOWN' and i<(na-vu)) else '#ff4455'}'></span>"
        for i in range(na)
    )
    algo_names = {"lgbm":"LG","xgb":"XG","logreg":"LR"}
    vote_detail = " · ".join(
        f"<span style='color:{'#00c47a' if v>=p.get('threshold',0.5) else '#ff4455'}'>"
        f"{algo_names.get(a,a)}</span>"
        for a,v in p["algo_probs"].items()
    )

    card_color = "#00c47a" if sig == "UP" else "#ff4455"
    cards += (
        f"<div class='pred-card {cc}'>"
        f"<div class='c-horizon'>{H_LABELS[hk]}</div>"
        f"<div class='c-signal {sc}'>{ar} {sig}</div>"
        f"<div class='c-prob-num' style='color:{card_color}'>"
        f"%{conf*100:.0f}</div>"
        f"<div class='bar-bg'><div class='{bc}' style='width:{conf*100:.0f}%'></div></div>"
        f"<div class='c-sub'>📊 Tahmin Güveni<br><span style='font-size:.6rem;color:#3d4a5c'>(confidence)</span></div>"
        f"<div class='votes'>{dots}"
        f"<span class='vote-label'>{vote_detail}</span></div>"
        f"</div>"
    )
cards += "</div>"
st.markdown(cards, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GRAFİK — yfinance 1 ay + tahmin okları
# ─────────────────────────────────────────────
st.markdown("<div class='sec-title'>Son 1 Ay Fiyat Hareketi & Tahmin Yönleri</div>",
            unsafe_allow_html=True)

# Grafik için sadece son 1 ay
if live_df is not None and not live_df.empty:
    chart_df = live_df.tail(22).copy()    # ~1 ay iş günü
else:
    chart_df = None

fig = make_subplots(rows=2, cols=1, row_heights=[0.78,0.22],
                    shared_xaxes=True, vertical_spacing=0.03)

UP_C   = "#00c47a"
DOWN_C = "#ff4455"

if chart_df is not None and not chart_df.empty:
    # Mum grafiği
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        name="Fiyat",
        increasing_line_color=UP_C,   decreasing_line_color=DOWN_C,
        increasing_fillcolor=rgba(UP_C, 0.2),
        decreasing_fillcolor=rgba(DOWN_C, 0.2),
    ), row=1, col=1)

    # MA20
    ma = chart_df["Close"].rolling(20, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=ma, name="MA20",
        line=dict(color="#2a3a4c", width=1.2, dash="dot"),
        hoverinfo="skip"), row=1, col=1)

    # Hacim
    vc = [rgba(UP_C,.15) if c>=o else rgba(DOWN_C,.15)
          for c,o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df["Volume"],
        marker_color=vc, showlegend=False), row=2, col=1)

    # ── Tahmin okları
    if ref_price and preds:
        last_date = chart_df.index[-1]

        COLORS = {"1d": UP_C, "3d":"#34d9a5", "5d":"#f0c040", "7d":"#a78bfa"}

        # iş günü offset'leri
        offsets = {"1d":1, "3d":3, "5d":5, "7d":7}

        for hk in ["1d","3d","5d","7d"]:
            if hk not in preds: continue
            p      = preds[hk]
            col    = COLORS[hk]
            dire   = 1 if p["signal"]=="UP" else -1
            conf   = p["confidence"]

            # Ok ucu: son kapanıştan ± conf * %3 sapma
            target_x = last_date + pd.tseries.offsets.BDay(offsets[hk])
            target_y  = ref_price * (1 + dire * conf * 0.03)

            # Ok
            fig.add_annotation(
                x=target_x, y=target_y,
                ax=last_date, ay=ref_price,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=3, arrowwidth=2.5,
                arrowcolor=col, showarrow=True,
            )
            # Etiket
            label = f"<b>{hk} {p['signal']}</b>"
            fig.add_annotation(
                x=target_x,
                y=target_y * (1 + dire * 0.005),
                text=label, showarrow=False,
                font=dict(color=col, size=10, family="DM Mono"),
                bgcolor="#07090f", borderpad=3,
            )

        # "Bugün" dikey çizgisi
        fig.add_vline(x=last_date,
                      line_color=rgba("#ffffff",0.07), line_width=1)
        fig.add_annotation(
            x=last_date, y=1, yref="paper",
            text=f"<b>Bugün</b>",
            showarrow=False, xanchor="right", yanchor="top",
            font=dict(color="#253545", size=10, family="DM Mono"),
        )

fig.update_layout(
    paper_bgcolor="#07090f", plot_bgcolor="#07090f",
    font=dict(color="#3d4a5c", family="DM Sans"),
    xaxis_rangeslider_visible=False,
    height=500, margin=dict(l=4,r=4,t=8,b=4),
    legend=dict(bgcolor="#0b0e17", bordercolor="#161b28", borderwidth=1,
                orientation="h", y=1.04, x=0, font=dict(size=11)),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#0b0e17", bordercolor="#161b28"),
    # X eksenini geçmişi + gelecek 10 iş günü gösterecek şekilde genişlet
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


# ─────────────────────────────────────────────
# OLASILIK ÇUBUKLARI
# ─────────────────────────────────────────────
st.markdown("<div class='sec-title'>Horizon Bazlı Güven Skoru</div>",
            unsafe_allow_html=True)

prob_rows = []
for hk in ["1d","3d","5d","7d"]:
    if hk not in preds: continue
    p = preds[hk]
    prob_rows.append({
        "h":    H_LABELS[hk],
        "UP":   round(p["prob_up"]*100, 1),
        "DOWN": round((1-p["prob_up"])*100, 1),
        "sig":  p["signal"],
    })

if prob_rows:
    pd2 = pd.DataFrame(prob_rows)
    f2  = go.Figure()
    f2.add_trace(go.Bar(
        name="↑ UP", x=pd2["h"], y=pd2["UP"],
        marker_color=UP_C,
        text=[f"%{v:.0f}" for v in pd2["UP"]],
        textposition="inside",
        textfont=dict(color="white", size=14, family="DM Mono")))
    f2.add_trace(go.Bar(
        name="↓ DOWN", x=pd2["h"], y=pd2["DOWN"],
        marker_color=DOWN_C,
        text=[f"%{v:.0f}" for v in pd2["DOWN"]],
        textposition="inside",
        textfont=dict(color="white", size=14, family="DM Mono")))
    f2.add_hline(y=50, line_dash="dot", line_color="#1a2535", line_width=1.5)
    f2.update_layout(
        barmode="stack", paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font=dict(color="#3d4a5c", family="DM Sans"),
        height=190, margin=dict(l=4,r=4,t=4,b=4),
        legend=dict(bgcolor="#0b0e17", bordercolor="#161b28",
                    orientation="h", y=1.2, x=0),
        yaxis=dict(gridcolor="#0c1018", range=[0,100],
                   ticksuffix="%", tickfont=dict(size=10)),
        xaxis=dict(gridcolor="#0c1018"), bargap=0.25,
    )
    st.plotly_chart(f2, use_container_width=True)
