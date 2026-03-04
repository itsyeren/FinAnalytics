"""
Split all_stocks.csv into individual ticker files
"""
import pandas as pd
from pathlib import Path

# Load combined data
df = pd.read_csv("data/all_stocks.csv")

# Create data dir if needed
Path("data/split").mkdir(exist_ok=True)

# Split by ticker
for ticker in df["Ticker"].unique():
    ticker_df = df[df["Ticker"] == ticker].copy()
    ticker_df = ticker_df.drop("Ticker", axis=1)
    ticker_df.to_csv(f"data/split/{ticker}.csv", index=False)
    print(f"✓ {ticker}: {len(ticker_df)} rows")

print(f"\n✓ {df['Ticker'].nunique()} hisse split edildi")
