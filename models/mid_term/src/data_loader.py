"""
data_loader.py - Alpaca API'den toplu veri cekme
"""
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import client, ALL_TICKERS, TRAIN_START, TRAIN_END


def fetch_all_data():
    """Tum hisse ve ETF verilerini Alpaca'dan toplu olarak ceker."""
    print("[1/7] Tum veriler Alpaca'dan cekiliyor...")

    req = StockBarsRequest(
        symbol_or_symbols=ALL_TICKERS,
        timeframe=TimeFrame.Day,
        start=TRAIN_START,
        end=TRAIN_END
    )
    bars = client.get_stock_bars(req)
    df_all = bars.df.reset_index()

    print(f"       {len(df_all)} satir veri cekildi.\n")
    return df_all
