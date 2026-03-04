"""
data_loader.py
==============
Kaggle'dan gelen "her hisse ayrı CSV" formatını okur.
İlk çalıştırmada parquet cache oluşturur (sonraki çalıştırmalar çok hızlı olur).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, ALL_TICKERS, MIN_ROWS


# ─── Olası kolon adı varyasyonları (Kaggle dataseti bazen farklı isimler kullanır)
COL_MAP = {
    "open":   "Open",  "Open":   "Open",  "OPEN":   "Open",
    "high":   "High",  "High":   "High",  "HIGH":   "High",
    "low":    "Low",   "Low":    "Low",   "LOW":    "Low",
    "close":  "Close", "Close":  "Close", "CLOSE":  "Close",
    "volume": "Volume","Volume": "Volume","VOLUME": "Volume",
    "adj close":"Close","Adj Close":"Close","adj_close":"Close",
}


def _find_csv(ticker: str, data_dir: Path) -> Path | None:
    """Ticker için CSV dosyasını çeşitli isim formatlarında arar."""
    candidates = [
        data_dir / f"{ticker}.csv",
        data_dir / f"{ticker.lower()}.csv",
        data_dir / f"{ticker.upper()}.csv",
        data_dir / f"{ticker}_daily.csv",
        data_dir / f"{ticker}_history.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Wildcard arama
    found = list(data_dir.glob(f"*{ticker}*.csv"))
    return found[0] if found else None


def load_single(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    """
    Tek bir ticker için CSV yükler, temizler ve döndürür.
    Döndürülen DataFrame index=DatetimeIndex, sütunlar=OHLCV
    """
    csv_path = _find_csv(ticker, data_dir)
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)

    # ── Kolon adlarını normalize et
    df.rename(columns={c: COL_MAP.get(c, c) for c in df.columns}, inplace=True)

    # ── Tarih sütununu bul ve index yap
    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "datetime", "timestamp", "time")),
        None
    )
    if date_col is None:
        # İlk sütunu dene
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "Date"

    # ── Zorunlu sütunlar
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None

    df = df[required].copy()

    # ── Tip dönüşümü
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Temizlik
    df = df.dropna()
    df = df[df["Close"] > 0]
    df = df[df["Volume"] > 0]

    # ── Duplikat tarih varsa son değeri al
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < MIN_ROWS:
        return None

    df["Ticker"] = ticker
    return df


def load_all(
    tickers: list[str] = ALL_TICKERS,
    data_dir: Path = DATA_DIR,
    cache_path: Path = Path("data/all_stocks.csv"),
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Tüm tickerları yükler, birleştirir, CSV cache üretir.
    
    Parameters
    ----------
    tickers      : Yüklenecek ticker listesi
    data_dir     : CSV dosyalarının bulunduğu klasör
    cache_path   : Cache dosyasının yolu
    force_reload : True ise cache'i görmezden gelip baştan yükler
    
    Returns
    -------
    pd.DataFrame  multi-index  (Ticker, Date)
    """

    cache_path = Path(cache_path)

    if cache_path.exists() and not force_reload:
        print(f"✓ Cache bulundu: {cache_path}  →  yükleniyor...")
        df = pd.read_csv(cache_path, index_col=["Ticker", "Date"], parse_dates=["Date"])
        print(f"  {len(df.index.get_level_values('Ticker').unique())} ticker, {len(df):,} satır")
        return df

    print(f"Veriler yükleniyor... ({len(tickers)} ticker)")
    frames = []
    missing = []

    for ticker in tqdm(tickers, desc="CSV okuma"):
        df = load_single(ticker, data_dir)
        if df is not None:
            frames.append(df)
        else:
            missing.append(ticker)

    if missing:
        print(f"\n⚠  Bulunamayan / yetersiz veri: {missing}")

    if not frames:
        raise FileNotFoundError(
            f"Hiç CSV okunamadı!\n"
            f"DATA_DIR = {data_dir.resolve()}\n"
            f"config.py içinde DATA_DIR'ı güncelleyin."
        )

    combined = pd.concat(frames, ignore_index=False)
    combined = combined.reset_index()
    combined = combined.set_index(["Ticker", "Date"])
    combined = combined.sort_index()

    # Cache kaydet
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path)
    print(f"\n✓ Cache oluşturuldu: {cache_path}")
    print(f"  {len(combined.index.get_level_values('Ticker').unique())} ticker, {len(combined):,} satır")

    return combined


def get_ticker_df(combined: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Combined DataFrame'den tek ticker'ı çıkarır, DatetimeIndex döner."""
    try:
        df = combined.loc[ticker].copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except KeyError:
        raise KeyError(f"'{ticker}' bulunamadı. Mevcut: {list(combined.index.get_level_values('Ticker').unique())}")


if __name__ == "__main__":
    # Hızlı test
    data = load_all(force_reload=True)
    print(data.head())
    print(data.dtypes)
