"""
predict.py
==========
Eğitilmiş modellerle tek ticker için anlık UP/DOWN tahmini üretir.
Üç algoritmanın sonuçlarını hem ayrı ayrı hem de oy çokluğuyla gösterir.

Kullanım:
    python predict.py AAPL
    python predict.py AAPL --horizon 3
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import HORIZONS, MODELS_DIR
from data_loader import load_all, get_ticker_df
from features import build_features, get_feature_cols


ALGO_LABELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "logreg": "LogReg"}


def load_model_bundle(ticker: str, horizon: int, algo: str) -> dict | None:
    path = MODELS_DIR / ticker / f"{algo}_{horizon}d.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_latest(ticker: str, horizons: list[int] = None):
    """
    Ticker için en güncel barda tüm horizon × algoritma tahminlerini üretir.

    Returns
    -------
    dict  {horizon_str: {algo: {"label":"UP"/"DOWN", "prob": float}}}
    """
    if horizons is None:
        horizons = HORIZONS

    # Veri
    combined = load_all()
    raw_df   = get_ticker_df(combined, ticker)
    full_df  = build_features(raw_df)
    feat_cols = get_feature_cols(full_df)

    # En son satır (NaN içermemeli)
    latest = full_df[feat_cols].dropna().iloc[[-1]]
    if latest.empty:
        raise ValueError("Son bar'da NaN değer var, tahmin yapılamıyor.")

    X_latest = latest.values.astype(np.float32)

    results = {}

    for h in horizons:
        h_key = f"{h}d"
        results[h_key] = {}

        for algo in ["lgbm", "xgb", "logreg"]:
            bundle = load_model_bundle(ticker, h, algo)
            if bundle is None:
                results[h_key][algo] = None
                continue

            model    = bundle["model"]
            scaler   = bundle["scaler"]

            # Sütun sırası model eğitimiyle aynı olmalı
            model_feat_cols = bundle.get("feat_cols", feat_cols)
            X = full_df[model_feat_cols].dropna().iloc[[-1]].values.astype(np.float32)

            try:
                if algo == "lgbm":
                    prob = float(model.predict(X)[0])
                elif algo == "xgb":
                    prob = float(model.predict_proba(X)[0, 1])
                else:  # logreg
                    X_sc = scaler.transform(X)
                    prob = float(model.predict_proba(X_sc)[0, 1])

                results[h_key][algo] = {
                    "label": "UP ↗" if prob >= 0.5 else "DOWN ↘",
                    "prob":  round(prob, 4),
                }
            except Exception as e:
                results[h_key][algo] = {"error": str(e)}

    return results, raw_df


def majority_vote(algo_results: dict) -> dict:
    """
    Üç algoritmanın oyu: oy çokluğu ile nihai karar.
    """
    votes = {"UP": 0, "DOWN": 0}
    probs = []

    for algo, res in algo_results.items():
        if res is None or "error" in res:
            continue
        label_clean = res["label"].split()[0]  # "UP" veya "DOWN"
        votes[label_clean] += 1
        probs.append(res["prob"])

    if not probs:
        return {"label": "N/A", "votes": votes, "avg_prob": None}

    winner = "UP" if votes["UP"] >= votes["DOWN"] else "DOWN"
    label_str = "UP ↗" if winner == "UP" else "DOWN ↘"
    return {
        "label":    label_str,
        "votes":    votes,
        "avg_prob": round(float(np.mean(probs)), 4),
    }


def print_prediction_report(ticker: str, results: dict, raw_df: pd.DataFrame):
    current_price = raw_df["Close"].iloc[-1]
    last_date     = raw_df.index[-1]

    print(f"\n{'='*65}")
    print(f"  TAHMİN RAPORU  |  {ticker}  |  Son kapanış: ${current_price:.2f}  ({last_date.date()})")
    print(f"{'='*65}")

    for h_key, algo_results in results.items():
        vote = majority_vote(algo_results)

        print(f"\n  ─── {h_key.upper()} Horizon ─────────────────────────────")
        print(f"  {'Algoritma':<15} {'Karar':<12} {'Prob':<10}")
        print(f"  {'-'*37}")

        for algo, res in algo_results.items():
            name = ALGO_LABELS.get(algo, algo)
            if res is None:
                print(f"  {name:<15} {'(model yok)':<12}")
            elif "error" in res:
                print(f"  {name:<15} {'HATA':<12}")
            else:
                bar   = "█" * int(res["prob"] * 20)
                print(f"  {name:<15} {res['label']:<12} {res['prob']:.2f}  {bar}")

        print(f"\n  ► OY ÇOKLUĞU:  {vote['label']}  "
              f"({vote['votes']['UP']}↗ / {vote['votes']['DOWN']}↘)  "
              f"Ort.prob={vote['avg_prob']}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Short Model Inference")
    parser.add_argument("ticker", type=str, help="Örn: AAPL")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Tek horizon (1, 3, 5 ya da 7)")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else HORIZONS
    results, raw_df = predict_latest(args.ticker, horizons)
    print_prediction_report(args.ticker, results, raw_df)
