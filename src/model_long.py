import sys
import os
from pathlib import Path

# --- 0. PATH FIX ---
ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

import joblib
import numpy as np
import pandas as pd
import glob
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURATION ---
DATA_PATH = Path(ROOT) / "data/raw/D1"
MODEL_SAVE_PATH = Path(ROOT) / "models/trend_classifier_model.pkl"
HORIZON = 21

# --- CRITICAL CHANGE: REMOVED drawdown_63 ---
# We removed the "value" metric to force the model to look at "trend/momentum"
FEATURE_COLS = [
    "ret_21",           # 1-month momentum
    "mom_126",          # 6-month structural trend
    "ma_ratio_21_63",   # Trend strength
    # "drawdown_63",    <-- DELETED: It was confusing the model
    "dist_sma_200",     # Long-term pivot distance
    "sma50_slope20",    # Trend velocity
    "mkt_ret_63"        # Market regime context
]

def load_and_preprocess():
    """
    Loads raw data, applies features, and creates the 'Strong Winner' target.
    """
    from src.config import UNIVERSE
    from src.features import add_features

    print(">>> Loading Data & Engineering Features...")
    files = glob.glob(str(DATA_PATH / "*.US_D1.csv"))

    available = {os.path.basename(f).split(".")[0] for f in files}
    tickers = [t for t in UNIVERSE if t in available]

    dfs = []
    for t in tickers:
        try:
            d = pd.read_csv(DATA_PATH / f"{t}.US_D1.csv")
            d["ticker"] = t
            dfs.append(d)
        except Exception as e:
            print(f"Warning: Could not load {t} ({e})")

    if not dfs:
        raise ValueError("No data loaded. Check your data/raw/D1 folder.")

    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    # Apply Features (matches src/features.py logic)
    df = add_features(df, normalize=True)

    # --- TARGET GENERATION ---
    df["fwd_ret"] = df.groupby("ticker")["close"].shift(-HORIZON) / df["close"] - 1
    df["mkt_fwd_ret"] = df.groupby("datetime")["fwd_ret"].transform("mean")

    # Target 1 = Winner (Outperforms market by > 2%)
    df["target"] = (df["fwd_ret"] > (df["mkt_fwd_ret"] + 0.02)).astype(int)

    # Drop rows where features are NaN
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    print(f"Data Ready: {len(df)} rows. Target Balance: {df['target'].mean():.2%}")
    return df

def train_robust_model():
    """
    Trains a Pure Momentum Classifier (No mean-reversion logic).
    """
    df = load_and_preprocess()
    dates = sorted(df["datetime"].unique())

    # Walk-Forward Split
    train_end_idx = int(len(dates) * 0.8)
    train_dates = dates[:train_end_idx]
    test_dates = dates[train_end_idx:]

    train_ds = df[df["datetime"].isin(train_dates)]
    test_ds = df[df["datetime"].isin(test_dates)]

    print(f"\n>>> Training on {len(train_ds)} samples. Validating on {len(test_ds)} samples.")

    # --- PURE MOMENTUM SETTINGS ---
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=40,         # Moderate complexity
        max_depth=6,           # Good depth for trend patterns
        min_child_samples=100, # Stable clusters
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=2.0,  # Aggressive weighting to find Winners
        random_state=42,
        importance_type='gain',
        n_jobs=-1
    )

    print(">>> Starting Momentum Training...")
    model.fit(
        train_ds[FEATURE_COLS],
        train_ds["target"],
        eval_set=[(test_ds[FEATURE_COLS], test_ds["target"])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0)
        ]
    )

    # --- CHECK RESULTS ---
    probs = model.predict_proba(test_ds[FEATURE_COLS])[:, 1]

    print("\n--- PROBABILITY SPREAD ---")
    print(pd.Series(probs).describe())

    # Top 10% Cutoff
    dynamic_threshold = np.percentile(probs, 90)
    print(f"Threshold (Top 10%): {dynamic_threshold:.4f}")

    preds = (probs >= dynamic_threshold).astype(int)

    print("\n" + "="*40)
    print("   MOMENTUM MODEL REPORT (ENGLISH)   ")
    print("="*40)
    print(classification_report(test_ds["target"], preds))

    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(test_ds["target"], preds))

    # Feature Importance (Should now be dominated by SLOPE or MOM_126)
    importances = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_})
    print("\nFEATURE IMPORTANCE (Gain):")
    print(importances.sort_values('importance', ascending=False))

    # Save
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_cols": FEATURE_COLS,
        "horizon": HORIZON,
        "threshold_percentile": 90
    }, MODEL_SAVE_PATH)
    print(f"\n>>> Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_robust_model()
