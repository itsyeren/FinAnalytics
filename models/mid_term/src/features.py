"""
features.py - Feature engineering (10 teknik ozellik + regression targets)
"""
from config import HORIZON_1M, HORIZON_3M


def build_features(df_stock, df_sector_etf, is_training=True):
    """
    Hisse verisine 10 teknik ozellik ekler.
    Regression target: 1 ay ve 3 ay sonraki yuzdesel getiri
    """
    df = df_stock.set_index("timestamp").join(df_sector_etf, how="inner")

    # 1. dist_sma_200
    sma_200 = df["close"].rolling(200).mean()
    df["dist_sma_200"] = (df["close"] - sma_200) / sma_200

    # 2. dist_sma_50
    sma_50 = df["close"].rolling(50).mean()
    df["dist_sma_50"] = (df["close"] - sma_50) / sma_50

    # 3. RSI (14 gunluk)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 4. Volatilite (20 gunluk)
    df["volatility"] = df["close"].pct_change().rolling(20).std()

    # 5. Relatif Guc (Hisse / Sektor ETF)
    df["rel_strength"] = df["close"] / df["sector_close"]

    # 6. Sektor Trendi (ETF > SMA200 mi?)
    sector_sma_200 = df["sector_close"].rolling(200).mean()
    df["is_sector_bullish"] = (df["sector_close"] > sector_sma_200).astype(int)

    # 7. Hacim Orani
    avg_vol = df["volume"].rolling(50).mean()
    df["volume_ratio"] = df["volume"] / avg_vol

    # 8. MACD Signal
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_signal"] = macd_line - signal_line

    # 9. Bollinger Band Pozisyonu
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    # 10. Momentum (20 gunluk getiri)
    df["momentum_20d"] = df["close"].pct_change(20)

    # TARGET: Yuzdesel getiri (regression)
    if is_training:
        df["target_1m"] = df["close"].shift(-HORIZON_1M) / df["close"] - 1
        df["target_3m"] = df["close"].shift(-HORIZON_3M) / df["close"] - 1

    return df.dropna()
