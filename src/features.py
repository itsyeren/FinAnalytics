import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Calculates the simplified feature set for the Long-Term Model.
    Includes technical indicators and daily cross-sectional normalization.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    # --- 1. Momentum & Returns ---
    # ret_21: 1-month price change
    df["ret_21"] = df.groupby("ticker")["close"].pct_change(21)
    # mom_126: 6-month price change (Structural momentum)
    df["mom_126"] = df.groupby("ticker")["close"].pct_change(126)

    # --- 2. Trend Structure ---
    # ma_ratio_21_63: Ratio of short-term to medium-term trend
    sma_21 = df.groupby("ticker")["close"].rolling(21).mean().reset_index(level=0, drop=True)
    sma_63 = df.groupby("ticker")["close"].rolling(63).mean().reset_index(level=0, drop=True)
    df["ma_ratio_21_63"] = sma_21 / (sma_63 + 1e-8)

    # drawdown_63: Distance from the 3-month high
    rolling_max_63 = df.groupby("ticker")["close"].rolling(63).max().reset_index(level=0, drop=True)
    df["drawdown_63"] = df["close"] / (rolling_max_63 + 1e-8) - 1

    # dist_sma_200: Distance from the 200-day moving average (Major pivot)
    sma_200 = df.groupby("ticker")["close"].rolling(200).mean().reset_index(level=0, drop=True)
    df["dist_sma_200"] = (df["close"] - sma_200) / (sma_200 + 1e-8)

    # sma50_slope20: Velocity of the 50-day trend
    sma_50 = df.groupby("ticker")["close"].rolling(50).mean().reset_index(level=0, drop=True)
    sma_50_lag20 = sma_50.groupby(df["ticker"]).shift(20)
    df["sma50_slope20"] = (sma_50 - sma_50_lag20) / (np.abs(sma_50_lag20) + 1e-8)

    # --- 3. Market Regime ---
    # mkt_ret_63: 3-month performance of the equal-weighted universe
    mkt = df.pivot_table(index="datetime", columns="ticker", values="close").mean(axis=1).to_frame("mkt_close")
    mkt["mkt_ret_63"] = mkt["mkt_close"].pct_change(63)

    df = df.merge(mkt[["mkt_ret_63"]], left_on="datetime", right_index=True, how="left")

    # --- 4. Cross-Sectional Normalization (Z-Score) ---
    if normalize:
        cols_to_scale = [
            "ret_21", "mom_126", "ma_ratio_21_63",
            "drawdown_63", "dist_sma_200", "sma50_slope20", "mkt_ret_63"
        ]
        for col in cols_to_scale:
            # Group by date to compare stocks against each other on the same day
            df[col] = df.groupby("datetime")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

    return df
