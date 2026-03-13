def add_long_score(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "datetime"])

    # Forward return calculation
    df["fwd_ret"] = df.groupby("ticker")["close"].shift(-horizon) / df["close"] - 1

    # Cross-sectional market average for that specific day
    df["mkt_fwd_ret"] = df.groupby("datetime")["fwd_ret"].transform("mean")

    # Binary Label: 1 if it beats the market, 0 otherwise
    df["target"] = (df["fwd_ret"] > df["mkt_fwd_ret"]).astype(int)

    # Drop rows where we don't have future data (the end of the dataset)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    return df
