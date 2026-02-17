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

    # Last 20% time split
    dates = sorted(df["datetime"].unique())
    split = int(len(dates) * 0.8)
    test_dates = dates[split:]

    test_df = df[df["datetime"].isin(test_dates)].copy()

    print("Test rows:", len(test_df))

    proba = model.predict_proba(test_df[FEATURE_COLS])
    test_df["prob_up"] = proba[:, 2]
    test_df["pred"] = model.predict(test_df[FEATURE_COLS])

    # ==============================
    # 1️⃣ UP CLASS PERFORMANCE
    # ==============================
    up_df = test_df[test_df["pred"] == 2]

    print("\n========== UP CLASS ==========")
    print("Signals:", len(up_df))

    if len(up_df) > 0:
        print("Mean return:", up_df["fwd_return"].mean())
        print("Median return:", up_df["fwd_return"].median())
        print("Hit rate:", (up_df["fwd_return"] > 0).mean())
    else:
        print("No UP signals.")

    # ==============================
    # 2️⃣ TOP 10% RANKING PERFORMANCE
    # ==============================
    print("\n========== TOP 10% RANKING ==========")

    top_decile_returns = []

    for date, group in test_df.groupby("datetime"):
        group = group.sort_values("prob_up", ascending=False)
        top_n = int(len(group) * 0.1)

        if top_n > 0:
            top_slice = group.head(top_n)
            top_decile_returns.append(top_slice["fwd_return"].mean())

    if len(top_decile_returns) > 0:
        print("Mean Top 10% return:", np.mean(top_decile_returns))
        print("Median Top 10% return:", np.median(top_decile_returns))

    # ==============================
    # 3️⃣ INFORMATION COEFFICIENT
    # ==============================
    print("\n========== IC ==========")

    ic_values = []

    for date, group in test_df.groupby("datetime"):
        if len(group) > 5:
            corr = group["prob_up"].corr(group["fwd_return"])
            if not np.isnan(corr):
                ic_values.append(corr)

    if len(ic_values) > 0:
        print("Mean IC:", np.mean(ic_values))
        print("Std IC:", np.std(ic_values))

    # ==============================
    # 4️⃣ SIMPLE LONG-ONLY EQUITY
    # ==============================
    print("\n========== SIMPLE EQUITY CURVE ==========")

    equity = 1.0
    equity_curve = []

    for date, group in test_df.groupby("datetime"):
        group = group.sort_values("prob_up", ascending=False)
        top_n = int(len(group) * 0.1)

        if top_n > 0:
            ret = group.head(top_n)["fwd_return"].mean()
            equity *= (1 + ret)
            equity_curve.append(equity)

    if equity_curve:
        print("Final Equity:", equity_curve[-1])
        print("Total Return:", equity_curve[-1] - 1)


if __name__ == "__main__":
    main()
