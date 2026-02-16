"""
LONG MODEL (63-Day Relative Outperformance Classification)
==========================================================

Goal:
Predict whether a stock will outperform the cross-sectional
market average over next 63 trading days.

Label:
y = 1  if fwd_ret > market_fwd_ret
y = 0  otherwise

Model:
LightGBM Classifier

Output:
- AUC
- Balanced Accuracy
- Top5 forward relative return
- Best 5 stocks by model quality

Saved as:
models/long_model.pkl
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


# =========================
# PATH
# =========================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import UNIVERSE
from src.features import add_features


# =========================
# CONFIG
# =========================
DATA_PATH = PROJECT_ROOT / "data/raw/D1"
MODEL_PATH = PROJECT_ROOT / "models/long_model.pkl"

FEATURE_COLS = [
    "ret_1","ret_5","ret_21",
    "mom_63","mom_126",
    "vol_21","vol_63",
    "ma_ratio_21_63",
    "drawdown_63"
]

HORIZON = 63
MIN_TRAIN_DAYS = 756
TEST_DAYS = 126
STEP_DAYS = 126   # faster folds

MODEL_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# =========================
# DATA LOAD
# =========================
def load_data():
    files = glob.glob(str(DATA_PATH / "*.US_D1.csv"))
    available = {os.path.basename(f).split(".")[0] for f in files}
    tickers = [t for t in UNIVERSE if t in available]

    dfs = []
    for t in tickers:
        d = pd.read_csv(DATA_PATH / f"{t}.US_D1.csv")
        d["ticker"] = t
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["ticker","datetime"]).reset_index(drop=True)
    return df


# =========================
# LABEL (Relative)
# =========================
def add_label(df):
    df = df.copy()

    df["fwd_close"] = df.groupby("ticker")["close"].shift(-HORIZON)
    df["fwd_ret"] = df["fwd_close"] / df["close"] - 1

    # Cross-sectional market forward return
    df["mkt_fwd_ret"] = df.groupby("datetime")["fwd_ret"].transform("mean")

    df["y"] = (df["fwd_ret"] > df["mkt_fwd_ret"]).astype(int)

    return df


# =========================
# WALK FORWARD
# =========================
def create_splits(df):
    dates = pd.Index(df["datetime"].unique()).sort_values()
    splits = []

    for i in range(MIN_TRAIN_DAYS, len(dates) - TEST_DAYS, STEP_DAYS):
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end = dates[i + TEST_DAYS - 1]
        splits.append((train_end, test_start, test_end))

    return splits


# =========================
# TRAIN
# =========================
def train():

    print("="*60)
    print("LONG MODEL — 63 DAY RELATIVE OUTPERFORM")
    print("="*60)

    df = load_data()
    df = add_features(df)
    df = add_label(df)

    df = df.dropna(subset=FEATURE_COLS + ["y","fwd_ret"]).reset_index(drop=True)

    # Cross-sectional feature normalization
    for col in FEATURE_COLS:
        df[col] = df.groupby("datetime")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    splits = create_splits(df)
    print("Folds:", len(splits))

    auc_list = []
    bacc_list = []
    top5_list = []
    all_preds = []

    for k, (train_end, test_start, test_end) in enumerate(splits):

        train_df = df[df["datetime"] <= train_end]
        test_df = df[(df["datetime"] >= test_start) & (df["datetime"] <= test_end)].copy()

        model = LGBMClassifier(**MODEL_PARAMS)
        model.fit(train_df[FEATURE_COLS], train_df["y"])

        proba_up = model.predict_proba(test_df[FEATURE_COLS])[:,1]
        pred = (proba_up >= 0.5).astype(int)

        auc = roc_auc_score(test_df["y"], proba_up)
        bacc = balanced_accuracy_score(test_df["y"], pred)

        auc_list.append(auc)
        bacc_list.append(bacc)

        test_df["p_up"] = proba_up
        all_preds.append(test_df[["datetime","ticker","p_up","fwd_ret","y"]])

        # Top5 outperform candidates
        daily_top5 = []
        for _, g in test_df.groupby("datetime"):
            g = g.sort_values("p_up", ascending=False).head(5)
            daily_top5.append(g["fwd_ret"].mean())

        top5_mean = float(np.mean(daily_top5))
        top5_list.append(top5_mean)

        print(f"Fold {k:02d} | AUC {auc:.3f} | BAcc {bacc:.3f} | Top5Ret {top5_mean:.3f}")

    print("-"*60)
    print("AVG AUC:", np.mean(auc_list))
    print("AVG Balanced Acc:", np.mean(bacc_list))
    print("AVG Top5 Forward Return:", np.mean(top5_list))
    print("-"*60)

    # Combine predictions
    oos = pd.concat(all_preds, ignore_index=True)

    stock_quality = (
        oos.groupby("ticker")
        .agg(
            avg_p_up=("p_up","mean"),
            hit_rate=("y","mean"),
            mean_fwd_ret=("fwd_ret","mean")
        )
        .sort_values("avg_p_up", ascending=False)
    )

    print("\n===== BEST 5 STOCKS BY MODEL =====")
    print(stock_quality.head(5))

    # Final model
    final_model = LGBMClassifier(**MODEL_PARAMS)
    final_model.fit(df[FEATURE_COLS], df["y"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    print("\nModel saved:", MODEL_PATH)


if __name__ == "__main__":
    train()
