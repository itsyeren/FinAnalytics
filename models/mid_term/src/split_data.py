"""
split_data.py - Sektor bazli veri hazirlama ve train/test split
"""
import pandas as pd
from config import SECTORS, FEATURES, HORIZON_3M, SPLIT_DATE
from features import build_features


def prepare_sector_data(df_all):
    """
    Her sektor icin:
    - ETF verisini hazirla
    - Hisse bazli feature engineering
    - Train/test split (SPLIT_DATE ile)
    
    Returns: dict {sector_name: {X_train, X_test, y_train_dict, y_test_dict, full_train, mask_train, mask_test, stock_info}}
    """
    print("[2/7] Sektor bazli veri hazirlama ve split basliyor...\n")

    sector_data = {}
    sector_count = 0
    total_sectors = len(SECTORS)

    for sector_name, sector_info in SECTORS.items():
        sector_count += 1
        etf_ticker = sector_info["etf"]
        stock_tickers = sector_info["stocks"]

        print(f"--- [{sector_count}/{total_sectors}] {sector_name} (ETF: {etf_ticker}) ---")

        # Sektor ETF verisini hazirla
        df_etf = df_all[df_all["symbol"] == etf_ticker][["timestamp", "close"]].copy()
        if len(df_etf) < 300:
            print(f"    [!] {etf_ticker} icin yeterli veri yok, atlanyor.\n")
            continue
        df_etf = df_etf.rename(columns={"close": "sector_close"}).set_index("timestamp")

        sector_train_dfs = []
        sector_stock_info = {}

        for stock in stock_tickers:
            df_stock = df_all[df_all["symbol"] == stock].copy()
            if len(df_stock) < 500:
                print(f"    [!] {stock}: Yetersiz veri ({len(df_stock)} satir), atlanyor.")
                continue

            try:
                df_processed = build_features(df_stock, df_etf, is_training=True)

                if len(df_processed) > HORIZON_3M:
                    sector_stock_info[stock] = {
                        "last_features": df_processed.iloc[[-1]][FEATURES],
                        "last_price": df_processed["close"].iloc[-1]
                    }
                    df_train_part = df_processed.iloc[:-HORIZON_3M].copy()
                    df_train_part["symbol"] = stock
                    sector_train_dfs.append(df_train_part)
            except Exception as e:
                print(f"    [!] {stock}: Hata - {e}")

        if not sector_train_dfs:
            print(f"    [!] Bu sektor icin egitim verisi olusturulamadi.\n")
            continue

        full_train = pd.concat(sector_train_dfs)
        X = full_train[FEATURES]

        # Train/Test Split
        split_date = pd.Timestamp(SPLIT_DATE, tz="UTC")
        if X.index.tz is None:
            X.index = X.index.tz_localize("UTC")
            full_train.index = full_train.index.tz_localize("UTC") if full_train.index.tz is None else full_train.index

        mask_train = X.index < split_date
        mask_test = X.index >= split_date

        if mask_train.sum() < 100 or mask_test.sum() < 30:
            print(f"    [!] Yetersiz train/test verisi, atlanyor.\n")
            continue

        print(f"    Egitim: {mask_train.sum()} satir | Test: {mask_test.sum()} satir")

        sector_data[sector_name] = {
            "X": X,
            "full_train": full_train,
            "mask_train": mask_train,
            "mask_test": mask_test,
            "stock_info": sector_stock_info
        }

    return sector_data
