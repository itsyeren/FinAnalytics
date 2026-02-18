"""
report_summary.py - JSON rapor dosyasi uretimi
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from config import (TRAIN_START, TRAIN_END, HORIZON_1M, HORIZON_3M,
                    SIGNAL_THRESHOLD, FEATURES, REPORTS_DIR)


def save_json_report(sector_results, all_predictions, all_stock_metrics):
    """sector_optimized_params.json dosyasini olusturur."""
    params_output = {
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_range": f"{TRAIN_START} -> {TRAIN_END}",
            "model_type": "Regression (Fiyat Tahmini) + Hibrit Sinyal Kalitesi",
            "signal_threshold": f">{SIGNAL_THRESHOLD*100:.0f}% getiri = AL sinyali",
            "horizons": {"1m": f"{HORIZON_1M} is gunu", "3m": f"{HORIZON_3M} is gunu"},
            "total_sectors": len(sector_results),
            "total_stocks_analyzed": len(all_predictions),
            "features_used": FEATURES
        },
        "sectors": {}
    }

    for sector_name, res in sector_results.items():
        sector_data = {"models": {}, "stock_predictions": {}}

        for hz in ["1m", "3m"]:
            m = res["models"][hz]
            clean_params = {}
            for k, v in m["best_params"].items():
                if isinstance(v, (np.integer,)):
                    clean_params[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_params[k] = float(v)
                else:
                    clean_params[k] = v

            sector_data["models"][hz] = {
                "best_model": m["best_name"],
                "best_params": clean_params,
                "regression_scores": {
                    "mae": round(m["best_scores"]["mae"], 4),
                    "rmse": round(m["best_scores"]["rmse"], 4),
                    "r2": round(m["best_scores"]["r2"], 4)
                },
                "signal_scores": {
                    "accuracy": round(m["signal_scores"]["acc"], 4),
                    "precision": round(m["signal_scores"]["prec"], 4),
                    "recall": round(m["signal_scores"]["recall"], 4),
                    "f1": round(m["signal_scores"]["f1"], 4)
                },
                "rf_regression": {
                    "mae": round(m["rf_scores"]["mae"], 4),
                    "rmse": round(m["rf_scores"]["rmse"], 4),
                    "r2": round(m["rf_scores"]["r2"], 4)
                },
                "rf_signal": {
                    "precision": round(m["rf_signal_scores"]["prec"], 4),
                    "recall": round(m["rf_signal_scores"]["recall"], 4),
                    "f1": round(m["rf_signal_scores"]["f1"], 4)
                },
                "lgb_regression": {
                    "mae": round(m["lgb_scores"]["mae"], 4),
                    "rmse": round(m["lgb_scores"]["rmse"], 4),
                    "r2": round(m["lgb_scores"]["r2"], 4)
                },
                "lgb_signal": {
                    "precision": round(m["lgb_signal_scores"]["prec"], 4),
                    "recall": round(m["lgb_signal_scores"]["recall"], 4),
                    "f1": round(m["lgb_signal_scores"]["f1"], 4)
                }
            }

        for pred in all_predictions:
            if pred["Sektor"] == sector_name:
                sector_data["stock_predictions"][pred["Hisse"]] = {
                    "son_fiyat": round(pred["Son_Fiyat"], 2),
                    "tahmin_1ay": round(pred["Tahmin_1Ay"], 2),
                    "getiri_1ay_pct": round(pred["Getiri_1Ay"] * 100, 2),
                    "sinyal_1ay": pred["Sinyal_1Ay"],
                    "tahmin_3ay": round(pred["Tahmin_3Ay"], 2),
                    "getiri_3ay_pct": round(pred["Getiri_3Ay"] * 100, 2),
                    "sinyal_3ay": pred["Sinyal_3Ay"],
                    "model_1m": pred["Model_1M"],
                    "model_3m": pred["Model_3M"]
                }

        sector_data["stocks"] = list(res["stock_info"].keys())
        params_output["sectors"][sector_name] = sector_data

    # Per-stock signal metriklerini ekle
    if all_stock_metrics:
        df_stk_json = pd.DataFrame(all_stock_metrics)
        top7_json = df_stk_json.nlargest(7, "F1")
        params_output["top_stock_signal_quality"] = []
        for _, row in top7_json.iterrows():
            params_output["top_stock_signal_quality"].append({
                "hisse": row["Hisse"],
                "sektor": row["Sektor"],
                "horizon": row["Horizon"],
                "model": row["Model"],
                "accuracy": round(row["Accuracy"], 4),
                "precision": round(row["Precision"], 4),
                "recall": round(row["Recall"], 4),
                "f1": round(row["F1"], 4),
                "mae": round(row["MAE"], 4),
                "n_test": int(row["N_Test"])
            })

    # Kaydet (hem kok dizine hem reports/ icine)
    for filepath in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sector_optimized_params.json"),
        os.path.join(REPORTS_DIR, "sector_optimized_params.json")
    ]:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params_output, f, indent=2, ensure_ascii=False)

    print(f"   -> Kaydedildi: {filepath}")
    return params_output
