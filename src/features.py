import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "datetime"])

    # Returns
    df["ret_1"] = df.groupby("ticker")["close"].pct_change()
    df["ret_5"] = df.groupby("ticker")["close"].pct_change(5)
    df["ret_21"] = df.groupby("ticker")["close"].pct_change(21)

    df["mom_63"] = df.groupby("ticker")["close"].pct_change(63)
    df["mom_126"] = df.groupby("ticker")["close"].pct_change(126)

    # Volatility
    df["vol_21"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(21).std().reset_index(level=0, drop=True)
    )

    df["vol_63"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(63).std().reset_index(level=0, drop=True)
    )

    # Moving averages
    df["ma_21"] = (
        df.groupby("ticker")["close"]
        .rolling(21).mean().reset_index(level=0, drop=True)
    )

    df["ma_63"] = (
        df.groupby("ticker")["close"]
        .rolling(63).mean().reset_index(level=0, drop=True)
    )

    df["ma_ratio_21_63"] = df["ma_21"] / df["ma_63"]

    # Drawdown
    rolling_max_63 = (
        df.groupby("ticker")["close"]
        .rolling(63).max().reset_index(level=0, drop=True)
    )

    df["drawdown_63"] = df["close"] / rolling_max_63 - 1

    # =============================
    # NEW FEATURES
    # =============================

    # RSI
    delta = df.groupby("ticker")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(df["ticker"]).rolling(14).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby(df["ticker"]).rolling(14).mean().reset_index(level=0, drop=True)

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())

    macd_line = ema_12 - ema_26
    signal_line = macd_line.groupby(df["ticker"]).transform(lambda x: x.ewm(span=9, adjust=False).mean())

    df["macd_signal"] = macd_line - signal_line

    # Bollinger position
    bb_mid = df.groupby("ticker")["close"].rolling(20).mean().reset_index(level=0, drop=True)
    bb_std = df.groupby("ticker")["close"].rolling(20).std().reset_index(level=0, drop=True)

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    # Volume ratio
    if "volume" in df.columns:
        vol_ma = df.groupby("ticker")["volume"].rolling(50).mean().reset_index(level=0, drop=True)
        df["volume_ratio"] = df["volume"] / vol_ma

    # Distance to long SMAs
    sma_200 = df.groupby("ticker")["close"].rolling(200).mean().reset_index(level=0, drop=True)
    sma_50 = df.groupby("ticker")["close"].rolling(50).mean().reset_index(level=0, drop=True)

    df["dist_sma_200"] = (df["close"] - sma_200) / sma_200
    df["dist_sma_50"] = (df["close"] - sma_50) / sma_50

    df = df.dropna().reset_index(drop=True)

    return df
