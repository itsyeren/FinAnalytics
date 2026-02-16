"""
Institutional Cross-Sectional Long Model (Mean Reversion)
=========================================================

- Target: Cross-sectional relative forward return
- Model: LightGBM Regressor
- Strategy: MEAN REVERSION
- Validation: Walk-forward
- Metrics:
    - IC (direction corrected)
    - Top5 mean return (lowest predicted)
    - Bottom5 mean return (highest predicted)
    - Long-Short spread
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

# =========================
# PATH
# =========================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import UNIVERSE
from src.features import add_features
from src.labels_long import add_long_score


# =========================
# CONFIG
# =========================
class Config:

    DATA_PATH = PROJECT_ROOT / "data/raw/D1"
    MODEL_DIR = PROJECT_ROOT / "models"

    MIN_DATE = "2010-01-04"

    FEATURE_COLS = [
        "ret_1","ret_5","ret_21",
        "mom_63","mom_126",
        "vol_21","vol_63",
        "ma_ratio_21_63",
        "drawdown_63"
    ]

    MIN_TRAIN_DAYS = 756
    TEST_DAYS = 126
    STEP_DAYS = 63

    LGBM_PARAMS = dict(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )


# =========================
# DATA LOAD
# =========================
def load_data():

    files = glob.glob(str(Config.DATA_PATH / "*.US_D1.csv"))
    available = {
        os.path.basename(f).split(".")[0]
        for f in files
    }

    tickers = [t for t in UNIVERSE if t in available]

    dfs = []

    for t in tickers:
        df_tmp = pd.read_csv(Config.DATA_PATH / f"{t}.US_D1.csv")
        df_tmp["ticker"] = t
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] >= Config.MIN_DATE]
    df = df.sort_values(["ticker","datetime"]).reset_index(drop=True)

    return df


# =========================
# WALK FORWARD
# =========================
def create_splits(df):

    dates = pd.Index(df["datetime"].unique()).sort_values()

    splits = []

    for i in range(
        Config.MIN_TRAIN_DAYS,
        len(dates) - Config.TEST_DAYS,
        Config.STEP_DAYS
    ):
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end = dates[i + Config.TEST_DAYS - 1]
        splits.append((train_end, test_start, test_end))

    return splits


# =========================
# MAIN TRAIN
# =========================
def train_model():

    print("="*60)
    print("INSTITUTIONAL MEAN REVERSION MODEL")
    print("="*60)

    df = load_data()

    df = add_features(df)
    df = add_long_score(df)

    df = df.dropna(subset=Config.FEATURE_COLS + ["ret_long_norm"])

    # ===== RELATIVE TARGET =====
    df["target_rel"] = df["ret_long_norm"] - \
        df.groupby("datetime")["ret_long_norm"].transform("mean")

    # ===== CROSS-SECTIONAL NORMALIZATION =====
    for col in Config.FEATURE_COLS:
        df[col] = df.groupby("datetime")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    splits = create_splits(df)
    print("Folds:", len(splits))

    ic_list = []
    top5_list = []
    bottom5_list = []
    spread_list = []

    for k, (train_end, test_start, test_end) in enumerate(splits):

        train = df[df["datetime"] <= train_end]
        test = df[
            (df["datetime"] >= test_start) &
            (df["datetime"] <= test_end)
        ]

        X_train = train[Config.FEATURE_COLS]
        y_train = train["target_rel"]

        X_test = test[Config.FEATURE_COLS]
        y_test = test["target_rel"]

        model = LGBMRegressor(**Config.LGBM_PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        test = test.copy()
        test["pred"] = preds

        # ===== IC (direction corrected) =====
        ic = -spearmanr(preds, y_test).correlation
        ic_list.append(ic)

        # ===== MEAN REVERSION RANKING =====
        daily_top5 = []
        daily_bottom5 = []

        for date, group in test.groupby("datetime"):

            # IMPORTANT: ascending=True
            group_sorted = group.sort_values("pred", ascending=True)

            top5 = group_sorted.head(5)      # lowest prediction
            bottom5 = group_sorted.tail(5)  # highest prediction

            daily_top5.append(top5["ret_long_norm"].mean())
            daily_bottom5.append(bottom5["ret_long_norm"].mean())

        top5_mean = np.mean(daily_top5)
        bottom5_mean = np.mean(daily_bottom5)
        spread = top5_mean - bottom5_mean

        top5_list.append(top5_mean)
        bottom5_list.append(bottom5_mean)
        spread_list.append(spread)

        print(f"Fold {k} | IC: {ic:.4f} | Spread: {spread:.4f}")

    print("-"*60)
    print("AVG IC:", np.mean(ic_list), "±", np.std(ic_list))
    print("AVG TOP5:", np.mean(top5_list))
    print("AVG BOTTOM5:", np.mean(bottom5_list))
    print("AVG SPREAD:", np.mean(spread_list))
    print("-"*60)

    # ===== FINAL MODEL =====
    final_model = LGBMRegressor(**Config.LGBM_PARAMS)
    final_model.fit(
        df[Config.FEATURE_COLS],
        df["target_rel"]
    )

    Config.MODEL_DIR.mkdir(exist_ok=True)
    model_path = Config.MODEL_DIR / "long_model_reg.pkl"
    joblib.dump(final_model, model_path)

    print("Model saved:", model_path)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    train_model()
