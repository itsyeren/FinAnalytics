import sys
import os
from pathlib import Path
import glob
import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report


# =========================================================
# ROOT PATH
# =========================================================
ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

DATA_PATH = Path(ROOT) / "data/raw/D1"
MODEL_SAVE_PATH = Path(ROOT) / "models/long_model.pkl"

# =========================================================
# CONFIGURATION
# =========================================================
FWD_START = 22
FWD_END = 63
VOL_WINDOW = 63
VOL_MULTIPLIER = 0.5  # volatility normalized threshold

FEATURE_COLS = [
    "ret_21",
    "mom_126",
    "ma_ratio_21_63",
    "dist_sma_200",
    "sma50_slope20",
    "mkt_ret_63",
]


# =========================================================
# LOAD + FEATURE + LABEL
# =========================================================
def load_and_preprocess():
    from src.config import UNIVERSE
    from src.features import add_features

    print("Loading data...")

    files = glob.glob(str(DATA_PATH / "*.US_D1.csv"))
    available = {os.path.basename(f).split(".")[0] for f in files}
    tickers = [t for t in UNIVERSE if t in available]

    dfs = []
    for t in tickers:
        df = pd.read_csv(DATA_PATH / f"{t}.US_D1.csv")
        df["ticker"] = t
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    df = add_features(df, normalize=True)

    # -----------------------------------------------------
    # Forward window: t+22 → t+63
    # -----------------------------------------------------
    df["price_fwd_start"] = df.groupby("ticker")["close"].shift(-FWD_START)
    df["price_fwd_end"] = df.groupby("ticker")["close"].shift(-FWD_END)

    df["ret_long"] = (
        df["price_fwd_end"] - df["price_fwd_start"]
    ) / df["close"]

    # Volatility normalization
    df["rolling_vol"] = (
        df.groupby("ticker")["close"]
        .pct_change()
        .rolling(VOL_WINDOW)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["ret_norm"] = df["ret_long"] / (df["rolling_vol"] + 1e-9)

    # -----------------------------------------------------
    # 3-Class Label
    # 0 = Down
    # 1 = Neutral
    # 2 = Up
    # -----------------------------------------------------
    df["target"] = 1  # Neutral default

    df.loc[df["ret_norm"] > VOL_MULTIPLIER, "target"] = 2
    df.loc[df["ret_norm"] < -VOL_MULTIPLIER, "target"] = 0

    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    print("Data Ready:", len(df))
    print("Class Distribution:")
    print(df["target"].value_counts(normalize=True))

    return df


# =========================================================
# WALK-FORWARD VALIDATION
# =========================================================
def walk_forward_evaluate(df, model_params):

    dates = np.array(sorted(df["datetime"].unique()))

    min_train_days = 756  # ~3 years
    test_days = 126       # ~6 months
    step_days = 63        # ~3 months

    train_end = min_train_days
    fold = 0
    accuracies = []

    while train_end + test_days <= len(dates):

        train_dates = dates[:train_end]
        test_dates = dates[train_end:train_end + test_days]

        train_ds = df[df["datetime"].isin(train_dates)]
        test_ds = df[df["datetime"].isin(test_dates)]

        model = LGBMClassifier(**model_params)

        model.fit(
            train_ds[FEATURE_COLS],
            train_ds["target"],
            eval_set=[(test_ds[FEATURE_COLS], test_ds["target"])],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )

        preds = model.predict(test_ds[FEATURE_COLS])
        acc = (preds == test_ds["target"]).mean()
        accuracies.append(acc)

        print(f"Fold {fold} Accuracy: {acc:.4f}")

        fold += 1
        train_end += step_days

    print("\nWalk-Forward Mean Accuracy:", np.mean(accuracies))


# =========================================================
# TRAIN FINAL MODEL
# =========================================================
def train_long_model():

    df = load_and_preprocess()

    model_params = dict(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=40,
        max_depth=6,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1
    )

    print("\nStarting walk-forward validation...")
    # walk_forward_evaluate(df, model_params)

    print("\nTraining final model on full dataset...")

    final_model = LGBMClassifier(**model_params)
    final_model.fit(df[FEATURE_COLS], df["target"])

    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

    joblib.dump({
        "model": final_model,
        "feature_cols": FEATURE_COLS,
        "horizon": "22-63",
        "classes": {0: "Down", 1: "Neutral", 2: "Up"}
    }, MODEL_SAVE_PATH)

    print("\nModel saved to:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    train_long_model()
