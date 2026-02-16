import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "datetime"])

    # === Base return ===
    df["ret_1"] = df.groupby("ticker")["close"].pct_change()

    # === Short horizon momentum ===
    df["ret_5"] = df.groupby("ticker")["close"].pct_change(5)
    df["ret_21"] = df.groupby("ticker")["close"].pct_change(21)

    # === Medium / long momentum ===
    df["mom_63"] = df.groupby("ticker")["close"].pct_change(63)
    df["mom_126"] = df.groupby("ticker")["close"].pct_change(126)

    # === Volatility ===
    df["vol_21"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(21)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["vol_63"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(63)
        .std()
        .reset_index(level=0, drop=True)
    )

    # === Moving averages ===
    df["ma_21"] = (
        df.groupby("ticker")["close"]
        .rolling(21)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["ma_63"] = (
        df.groupby("ticker")["close"]
        .rolling(63)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["ma_ratio_21_63"] = df["ma_21"] / df["ma_63"]

    # === Drawdown ===
    rolling_max_63 = (
        df.groupby("ticker")["close"]
        .rolling(63)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["drawdown_63"] = df["close"] / rolling_max_63 - 1

    # Drop rows with incomplete rolling windows
    df = df.dropna().reset_index(drop=True)

    return df
