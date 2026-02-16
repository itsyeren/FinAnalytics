import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "datetime"])

    # =============================
    # RETURNS
    # =============================
    df["ret_1"] = df.groupby("ticker")["close"].pct_change()
    df["ret_5"] = df.groupby("ticker")["close"].pct_change(5)
    df["ret_21"] = df.groupby("ticker")["close"].pct_change(21)

    df["mom_63"] = df.groupby("ticker")["close"].pct_change(63)
    df["mom_126"] = df.groupby("ticker")["close"].pct_change(126)

    # =============================
    # VOLATILITY
    # =============================
    df["vol_21"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(21).std()
        .reset_index(level=0, drop=True)
    )

    df["vol_63"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(63).std()
        .reset_index(level=0, drop=True)
    )

    # =============================
    # TREND STRUCTURE
    # =============================
    ma_21 = (
        df.groupby("ticker")["close"]
        .rolling(21).mean()
        .reset_index(level=0, drop=True)
    )

    ma_63 = (
        df.groupby("ticker")["close"]
        .rolling(63).mean()
        .reset_index(level=0, drop=True)
    )

    df["ma_ratio_21_63"] = ma_21 / ma_63

    rolling_max_63 = (
        df.groupby("ticker")["close"]
        .rolling(63).max()
        .reset_index(level=0, drop=True)
    )

    df["drawdown_63"] = df["close"] / rolling_max_63 - 1

    # =============================
    # OSCILLATORS (MEAN REVERSION CORE)
    # =============================

    # --- RSI (14)
    delta = df.groupby("ticker")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.groupby(df["ticker"])
        .rolling(14).mean()
        .reset_index(level=0, drop=True)
    )

    avg_loss = (
        loss.groupby(df["ticker"])
        .rolling(14).mean()
        .reset_index(level=0, drop=True)
    )

    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- Bollinger Position (20)
    bb_mid = (
        df.groupby("ticker")["close"]
        .rolling(20).mean()
        .reset_index(level=0, drop=True)
    )

    bb_std = (
        df.groupby("ticker")["close"]
        .rolling(20).std()
        .reset_index(level=0, drop=True)
    )

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)

    # --- Distance to SMA 50 (faster MR signal)
    sma_50 = (
        df.groupby("ticker")["close"]
        .rolling(50).mean()
        .reset_index(level=0, drop=True)
    )

    df["dist_sma_50"] = (df["close"] - sma_50) / (sma_50 + 1e-8)

    # =============================
    # CLEAN
    # =============================
    df = df.dropna().reset_index(drop=True)

    return df
