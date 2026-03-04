"""
generate_report.py
==================
training_results.json → tek dosya HTML rapor

Çalıştırma:
    poetry run python generate_report.py
    → reports/short_model_report.html
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


RESULTS_PATH = Path("reports/training_results.json")
OUTPUT_PATH  = Path("reports/short_model_report.html")
MODELS_DIR   = Path("models")


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def build_summary(results):
    rows = []
    for ticker, horizons in results.items():
        for h_key, data in horizons.items():
            if "error" in data:
                continue
            for algo, metrics in data.get("algorithms", {}).items():
                if "error" in metrics:
                    continue
                rows.append({
                    "Ticker":   ticker,
                    "Horizon":  h_key,
                    "Algo":     algo.upper(),
                    "Accuracy": metrics["accuracy"],
                    "F1":       metrics["f1"],
                    "ROC_AUC":  metrics["roc_auc"],
                    "N_test":   metrics["n_samples"],
                })
    return pd.DataFrame(rows)


def get_best(df):
    return df.loc[df.groupby(["Ticker","Horizon"])["ROC_AUC"].idxmax()]


def auc_color(v):
    if v >= 0.57:   return "#00d4aa"
    elif v >= 0.53: return "#f0c040"
    else:           return "#ff6b6b"


def feature_importance_html(ticker, horizon, algo):
    path = MODELS_DIR / ticker / f"{algo.lower()}_{horizon}.pkl"
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        model     = bundle["model"]
        feat_cols = bundle["feat_cols"]

        if algo == "LGBM":
            imp = model.feature_importance(importance_type="gain")
        elif algo == "XGB":
            imp = model.feature_importances_
        else:
            imp = np.abs(bundle["model"].coef_[0])

        fi = pd.DataFrame({"feature": feat_cols, "importance": imp})\
               .sort_values("importance", ascending=False).head(15)
        fi["importance"] = fi["importance"] / fi["importance"].max()

        rows_html = ""
        for _, row in fi.iterrows():
            pct = row["importance"] * 100
            rows_html += f"""
            <div class="fi-row">
                <span class="fi-name">{row['feature']}</span>
                <div class="fi-bar-bg">
                    <div class="fi-bar" style="width:{pct:.1f}%"></div>
                </div>
                <span class="fi-val">{pct:.1f}</span>
            </div>"""
        return rows_html
    except Exception:
        return ""


def generate_html(results, df, best):
    now   = datetime.now().strftime("%d %b %Y %H:%M")
    n_ticker   = df["Ticker"].nunique()
    avg_auc    = best["ROC_AUC"].mean()
    n_strong   = (best["ROC_AUC"] >= 0.55).sum()
    best_algo  = best["Algo"].value_counts().idxmax()

    # Heatmap verisi (ticker × horizon)
    pivot = best.pivot_table(index="Ticker", columns="Horizon", values="ROC_AUC")
    horizons = ["1d","3d","5d","7d"]
    pivot = pivot[[h for h in horizons if h in pivot.columns]]
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)

    heatmap_html = "<table class='heatmap'><thead><tr><th>Ticker</th>"
    for h in pivot.columns:
        heatmap_html += f"<th>{h}</th>"
    heatmap_html += "</tr></thead><tbody>"

    for ticker, row in pivot.iterrows():
        heatmap_html += f"<tr><td class='ticker-cell'>{ticker}</td>"
        for h in pivot.columns:
            v = row.get(h, 0)
            bg = auc_color(v) if not pd.isna(v) else "#333"
            opacity = max(0.2, min(1.0, (v - 0.48) / 0.17)) if not pd.isna(v) else 0
            val_str = f"{v:.3f}" if not pd.isna(v) else "—"
            heatmap_html += f"""<td style="background:{bg};opacity:{opacity:.2f};
                color:#fff;text-align:center;font-weight:600">{val_str}</td>"""
        heatmap_html += "</tr>"
    heatmap_html += "</tbody></table>"

    # Algo karşılaştırma
    algo_tbl = df.groupby("Algo")[["Accuracy","F1","ROC_AUC"]].mean().round(4)
    algo_html = "<table class='perf-table'><thead><tr>"
    algo_html += "<th>Algoritma</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>"
    for algo, row in algo_tbl.iterrows():
        color = {"LGBM":"#00d4aa","XGB":"#f0c040","LOGREG":"#ff6b6b"}.get(algo,"#ccc")
        algo_html += f"""<tr>
            <td><span style="color:{color};font-weight:600">{algo}</span></td>
            <td>{row['Accuracy']:.4f}</td>
            <td>{row['F1']:.4f}</td>
            <td style="color:{auc_color(row['ROC_AUC'])};font-weight:600">{row['ROC_AUC']:.4f}</td>
        </tr>"""
    algo_html += "</tbody></table>"

    # Horizon karşılaştırma
    h_tbl = best.groupby("Horizon")[["Accuracy","F1","ROC_AUC"]].mean().round(4)
    h_tbl = h_tbl.reindex(["1d","3d","5d","7d"])
    horizon_html = "<table class='perf-table'><thead><tr>"
    horizon_html += "<th>Horizon</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>"
    for h, row in h_tbl.iterrows():
        horizon_html += f"""<tr>
            <td><span class="badge">{h}</span></td>
            <td>{row['Accuracy']:.4f}</td>
            <td>{row['F1']:.4f}</td>
            <td style="color:{auc_color(row['ROC_AUC'])};font-weight:600">{row['ROC_AUC']:.4f}</td>
        </tr>"""
    horizon_html += "</tbody></table>"

    # En iyi 10 model
    top10 = best.sort_values("ROC_AUC", ascending=False).head(10)
    top10_html = "<table class='perf-table'><thead><tr>"
    top10_html += "<th>Ticker</th><th>Horizon</th><th>Algo</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>"
    for _, row in top10.iterrows():
        color = {"LGBM":"#00d4aa","XGB":"#f0c040","LOGREG":"#ff6b6b"}.get(row["Algo"],"#ccc")
        top10_html += f"""<tr>
            <td><strong>{row['Ticker']}</strong></td>
            <td><span class="badge">{row['Horizon']}</span></td>
            <td><span style="color:{color}">{row['Algo']}</span></td>
            <td>{row['Accuracy']:.4f}</td>
            <td>{row['F1']:.4f}</td>
            <td style="color:{auc_color(row['ROC_AUC'])};font-weight:600">{row['ROC_AUC']:.4f}</td>
        </tr>"""
    top10_html += "</tbody></table>"

    # Her ticker için feature importance (en iyi modelden)
    fi_sections = ""
    for _, row in best.sort_values("ROC_AUC", ascending=False).head(12).iterrows():
        fi_html = feature_importance_html(row["Ticker"], row["Horizon"], row["Algo"])
        if not fi_html:
            continue
        color = {"LGBM":"#00d4aa","XGB":"#f0c040","LOGREG":"#ff6b6b"}.get(row["Algo"],"#ccc")
        fi_sections += f"""
        <div class="fi-card">
            <div class="fi-card-header">
                <span class="fi-ticker">{row['Ticker']}</span>
                <span class="badge">{row['Horizon']}</span>
                <span style="color:{color};font-size:0.8rem">{row['Algo']}</span>
                <span style="color:{auc_color(row['ROC_AUC'])};margin-left:auto;font-weight:600">
                    AUC {row['ROC_AUC']:.3f}
                </span>
            </div>
            {fi_html}
        </div>"""

    # Tam HTML
    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinAnalytics — Short Model Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  *  {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #080b12;
    color: #c8d0e0;
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.6;
  }}
  h1,h2,h3,h4 {{ font-family: 'IBM Plex Mono', monospace; }}

  /* NAV */
  nav {{
    position: sticky; top: 0; z-index: 100;
    background: #0b0f1a;
    border-bottom: 1px solid #1e2535;
    padding: 0 40px;
    display: flex; align-items: center; gap: 32px; height: 56px;
  }}
  nav .brand {{ color: #00d4aa; font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:1rem; }}
  nav a {{ color:#888; text-decoration:none; font-size:0.85rem; transition:color .2s; }}
  nav a:hover {{ color:#00d4aa; }}

  /* LAYOUT */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}
  section {{ margin-bottom: 64px; }}
  h2 {{ font-size: 1.2rem; color: #00d4aa; border-left: 3px solid #00d4aa;
        padding-left: 12px; margin-bottom: 24px; }}

  /* KPI CARDS */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom:40px; }}
  .kpi-card {{
    background: #0f1520;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    transition: border-color .2s;
  }}
  .kpi-card:hover {{ border-color: #00d4aa44; }}
  .kpi-value {{ font-family:'IBM Plex Mono',monospace; font-size:2.2rem; font-weight:600; color:#00d4aa; }}
  .kpi-label {{ font-size:0.72rem; color:#666; text-transform:uppercase; letter-spacing:1.5px; margin-top:4px; }}

  /* TABLES */
  .perf-table {{
    width: 100%; border-collapse: collapse;
    font-size: 0.88rem;
  }}
  .perf-table th {{
    background: #0f1520; color: #888;
    font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
    text-transform:uppercase; letter-spacing:1px;
    padding: 10px 14px; text-align:left;
    border-bottom: 1px solid #1e2535;
  }}
  .perf-table td {{ padding: 10px 14px; border-bottom: 1px solid #111827; }}
  .perf-table tbody tr:hover {{ background: #0f1520; }}

  .heatmap {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  .heatmap th {{ background:#0f1520; color:#888; padding:8px 12px;
                 font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                 text-transform:uppercase; letter-spacing:1px;
                 border-bottom:1px solid #1e2535; }}
  .heatmap td {{ padding:8px 12px; border-bottom:1px solid #111827; }}
  .ticker-cell {{ font-family:'IBM Plex Mono',monospace; font-weight:600; color:#c8d0e0; }}

  .badge {{
    display:inline-block; background:#1a2235; border:1px solid #2d3a52;
    border-radius:4px; padding:1px 8px;
    font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#a0aec0;
  }}

  /* FEATURE IMPORTANCE */
  .fi-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; }}
  .fi-card {{
    background: #0f1520; border: 1px solid #1e2535; border-radius: 10px; padding: 20px;
  }}
  .fi-card-header {{
    display:flex; align-items:center; gap:8px; margin-bottom:16px;
    padding-bottom:12px; border-bottom:1px solid #1e2535;
  }}
  .fi-ticker {{ font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:1rem; color:#e2e8f0; }}
  .fi-row {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
  .fi-name {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#888; width:160px; flex-shrink:0;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .fi-bar-bg {{ flex:1; background:#1a2235; border-radius:2px; height:6px; }}
  .fi-bar {{ background:linear-gradient(90deg,#00d4aa,#00d4aa88); height:6px; border-radius:2px;
             transition:width .3s; }}
  .fi-val {{ font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#666; width:36px; text-align:right; }}

  /* LEGEND */
  .legend {{ display:flex; gap:20px; margin-bottom:16px; font-size:0.8rem; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; }}
  .legend-dot {{ width:10px; height:10px; border-radius:50%; }}

  /* FOOTER */
  footer {{
    text-align:center; padding:40px; color:#444; font-size:0.8rem;
    border-top: 1px solid #1e2535; margin-top:40px;
  }}
</style>
</head>
<body>

<nav>
  <span class="brand">📈 FinAnalytics</span>
  <a href="#overview">Genel Bakış</a>
  <a href="#heatmap">Isı Haritası</a>
  <a href="#algo">Algoritmalar</a>
  <a href="#horizon">Horizonlar</a>
  <a href="#top10">En İyi 10</a>
  <a href="#features">Feature Importance</a>
  <span style="margin-left:auto;color:#444;font-size:0.75rem">{now}</span>
</nav>

<div class="container">

  <!-- HEADER -->
  <div style="margin-bottom:48px;padding-top:8px">
    <h1 style="font-size:1.8rem;color:#e2e8f0;margin-bottom:8px">
      Short Model — Eğitim Raporu
    </h1>
    <p style="color:#666">LightGBM · XGBoost · Logistic Regression &nbsp;|&nbsp;
       1d · 3d · 5d · 7d &nbsp;|&nbsp; UP/DOWN Sınıflandırma</p>
  </div>

  <!-- KPI -->
  <section id="overview">
    <h2>Genel Bakış</h2>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-value">{n_ticker}</div>
        <div class="kpi-label">Ticker</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="color:{auc_color(avg_auc)}">{avg_auc:.3f}</div>
        <div class="kpi-label">Ort. ROC-AUC</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{n_strong}</div>
        <div class="kpi-label">Güçlü Model (≥0.55)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="font-size:1.4rem">{best_algo}</div>
        <div class="kpi-label">En Sık Kazanan</div>
      </div>
    </div>

    <div class="legend">
      <span class="legend-item"><span class="legend-dot" style="background:#00d4aa"></span>≥ 0.57 Güçlü</span>
      <span class="legend-item"><span class="legend-dot" style="background:#f0c040"></span>0.53–0.57 Orta</span>
      <span class="legend-item"><span class="legend-dot" style="background:#ff6b6b"></span>&lt; 0.53 Zayıf</span>
    </div>
  </section>

  <!-- HEATMAP -->
  <section id="heatmap">
    <h2>ROC-AUC Isı Haritası (Ticker × Horizon)</h2>
    <p style="color:#666;font-size:0.85rem;margin-bottom:16px">
      Her hücre o ticker × horizon için en iyi algoritmanın test seti ROC-AUC değerini gösterir.
    </p>
    {heatmap_html}
  </section>

  <!-- ALGORİTMA -->
  <section id="algo">
    <h2>Algoritma Karşılaştırması</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:32px">
      <div>
        <h4 style="color:#888;font-size:0.85rem;margin-bottom:12px">Ortalama Performans</h4>
        {algo_html}
      </div>
      <div>
        <h4 style="color:#888;font-size:0.85rem;margin-bottom:12px">Horizon Bazlı Ortalama (En İyi)</h4>
        {horizon_html}
      </div>
    </div>
  </section>

  <!-- EN İYİ 10 -->
  <section id="top10">
    <h2>En İyi 10 Model (ROC-AUC)</h2>
    {top10_html}
  </section>

  <!-- FEATURE IMPORTANCE -->
  <section id="features">
    <h2>Feature Importance (Top 12 Model)</h2>
    <p style="color:#666;font-size:0.85rem;margin-bottom:20px">
      Her model için normalize edilmiş gain bazlı önem skoru. 100 = en önemli feature.
    </p>
    <div class="fi-grid">
      {fi_sections}
    </div>
  </section>

</div>

<footer>
  FinAnalytics Short Model Report &nbsp;·&nbsp; {now} &nbsp;·&nbsp;
  {n_ticker} ticker · 4 horizon · 3 algoritma
</footer>

</body>
</html>"""
    return html


def main():
    print("Rapor oluşturuluyor...")
    results = load_results()
    df      = build_summary(results)
    best    = get_best(df)

    html = generate_html(results, df, best)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✓ HTML rapor kaydedildi: {OUTPUT_PATH}")
    print(f"  Tarayıcıda açmak için:")
    print(f"  open {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
