import pandas as pd
from pathlib import Path
from src.config import UNIVERSE

DATA_DIR = Path("data/raw/D1")
START_DATE = "2010-01-01"

def available_tickers():
    files = DATA_DIR.glob("*.US_D1.csv")
    return sorted([f.name.split(".")[0] for f in files])

def load_data(tickers=UNIVERSE):
    avail = set(available_tickers())
    keep = [t for t in tickers if t in avail]
    missing = [t for t in tickers if t not in avail]

    if missing:
        print("Missing tickers (no file found):", missing)

    dfs = []
    for t in keep:
        fp = DATA_DIR / f"{t}.US_D1.csv"
        df = pd.read_csv(fp)
        df["ticker"] = t
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] >= START_DATE]
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    print("Loaded tickers:", sorted(df["ticker"].unique()))
    print("Shape:", df.shape)
    print("Date range:", df["datetime"].min(), "→", df["datetime"].max())
    return df
