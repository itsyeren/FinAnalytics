import pandas as pd


def add_long_score(df: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "datetime"])

    # Forward return per ticker
    df["fwd_close"] = df.groupby("ticker")["close"].shift(-horizon)
    df["fwd_ret"] = df["fwd_close"] / df["close"] - 1

    # Cross-sectional market forward return (same date)
    df["mkt_fwd_ret"] = df.groupby("datetime")["fwd_ret"].transform("mean")

    # Relative outperform label
    df["y"] = (df["fwd_ret"] > df["mkt_fwd_ret"]).astype(int)

    df = df.dropna(subset=["y"]).reset_index(drop=True)

    return df
