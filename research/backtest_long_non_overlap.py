import sys
import os
from pathlib import Path
import glob
import joblib
import numpy as np
import pandas as pd

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

DATA_PATH = Path(ROOT) / "data/raw/D1"
MODEL_PATH = Path(ROOT) / "models/long_model.pkl"

FWD_START = 22
FWD_END = 63

FEATURE_COLS = [
    "ret_21",
    "mom_126",
    "ma_ratio_21_63",
    "dist_sma_200",
    "sma50_slope20",
    "mkt_ret_63",
]


def load_data():
    from src.config import UNIVERSE
    from src.features import add_features

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

    df["price_fwd_start"] = df.groupby("ticker")["close"].shift(-FWD_START)
    df["price_fwd_end"] = df.groupby("ticker")["close"].shift(-FWD_END)

    df["fwd_return"] = (
        df["price_fwd_end"] - df["price_fwd_start"]
    ) / df["close"]

    df = df.dropna(subset=FEATURE_COLS + ["fwd_return"]).reset_index(drop=True)

    return df


def main():

    print("Loading model...")
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]

    df = load_data()

    dates = sorted(df["datetime"].unique())

    # Last 30% for test
    split = int(len(dates) * 0.7)
    test_dates = dates[split:]

    test_df = df[df["datetime"].isin(test_dates)].copy()

    equity = 1.0
    equity_curve = []

    rebalance_step = 63  # non-overlapping

    for i in range(0, len(test_dates) - FWD_END, rebalance_step):

        rebalance_date = test_dates[i]

        day_slice = test_df[test_df["datetime"] == rebalance_date].copy()

        if len(day_slice) < 10:
            continue

        proba = model.predict_proba(day_slice[FEATURE_COLS])
        day_slice["prob_up"] = proba[:, 2]

        day_slice = day_slice.sort_values("prob_up", ascending=False)

        top_n = int(len(day_slice) * 0.1)
        if top_n == 0:
            continue

        selected = day_slice.head(top_n)

        mean_ret = selected["fwd_return"].mean()

        equity *= (1 + mean_ret)
        equity_curve.append(equity)

        print(f"Rebalance {rebalance_date.date()} | Return: {mean_ret:.4f} | Equity: {equity:.4f}")

    print("\n===== FINAL RESULTS =====")
    print("Final Equity:", equity)
    print("Total Return:", equity - 1)


if __name__ == "__main__":
    main()
