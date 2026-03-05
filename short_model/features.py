"""
features.py — Consumer Staples için optimize edilmiş feature seti
~30 yüksek kaliteli feature, Consumer Staples'ın stabil yapısına uygun.
"""
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

try:
    from short_model.config import HORIZONS, THRESHOLD
except ImportError:
    from config import HORIZONS, THRESHOLD


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in HORIZONS:
        future_ret = df["Close"].shift(-h) / df["Close"] - 1
        df[f"target_{h}d"] = (future_ret > THRESHOLD).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    out  = {}
    c    = df["Close"]
    h    = df["High"]
    lo   = df["Low"]
    v    = df["Volume"]
    ret1 = c.pct_change()

    # ── 1. Return & Momentum (Enhanced)
    out["ret_1d"]   = ret1
    out["ret_3d"]   = c.pct_change(3)
    out["ret_5d"]   = c.pct_change(5)
    out["ret_10d"]  = c.pct_change(10)
    out["ret_20d"]  = c.pct_change(20)

    # Return ivmesi (1 gün vs 3 gün — kısa vadeli momentum değişimi)
    out["accel_1_3"]  = out["ret_1d"] - out["ret_3d"] / 3
    # Return dönüşü sinyali (5 gün UP sonrası düşüş riski)
    out["mean_rev_5"] = -out["ret_5d"]   # ters sinyal

    # ── YENI: Autocorrelation features (mean reversion kuvveti)
    # Kısa vadeli autocorrelation = mean reversion marker (stables için güçlü)
    out["acf_lag1"]   = ret1.autocorr(lag=1)      # lag-1 correlation
    out["acf_lag2"]   = ret1.autocorr(lag=2)
    # Rolling autocorrelation (10 günlük)
    out["rolling_acf"] = ret1.rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x)>1 else np.nan, raw=False)

    # ── YENI: Price acceleration (2. türev — hız değişimi)
    out["price_acc_5"] = ret1.diff().rolling(5).mean()  # 5-gün acceleration
    out["price_jerk"] = out["price_acc_5"].diff()       # acceleration değişimi

    # Sharpe-like: getiri / volatilite
    std5 = ret1.rolling(5).std().replace(0, np.nan)
    std10= ret1.rolling(10).std().replace(0, np.nan)
    std20= ret1.rolling(20).std().replace(0, np.nan)
    out["sharpe_5d"]  = ret1.rolling(5).mean() / std5
    out["sharpe_10d"] = ret1.rolling(10).mean() / std10
    out["sharpe_20d"] = ret1.rolling(20).mean() / std20

    # Trend tutarlılığı
    out["up_days_5d"]  = (ret1 > 0).rolling(5).sum() / 5
    out["up_days_10d"] = (ret1 > 0).rolling(10).sum() / 10

    # ── YENI: Extreme move detection (sudden volatility spikes)
    out["extreme_move_5d"] = (np.abs(ret1) > ret1.rolling(20).std() * 2).rolling(5).sum()
    out["is_outlier"] = (np.abs(ret1 - ret1.rolling(20).mean()) > 3 * ret1.rolling(20).std()).astype(int)

    # ── 2. RSI — en güçlü teknik indikatör (Enhanced)
    out["rsi_14"]      = ta.momentum.RSIIndicator(c, window=14).rsi()
    out["rsi_7"]       = ta.momentum.RSIIndicator(c, window=7).rsi()
    out["rsi_21"]      = ta.momentum.RSIIndicator(c, window=21).rsi()
    # RSI momentum (eğim)
    out["rsi_slope"]   = out["rsi_14"] - out["rsi_14"].shift(5)
    out["rsi_accel"]   = out["rsi_slope"] - out["rsi_slope"].shift(3)  # RSI hızlanması
    # RSI normalize (0–1 arası)
    out["rsi_norm"]    = out["rsi_14"] / 100
    
    # YENI: RSI divergence (price vs RSI)
    price_up = (c > c.shift(5)).astype(int)
    rsi_up   = (out["rsi_14"] > out["rsi_14"].shift(5)).astype(int)
    out["rsi_divergence"] = (price_up * 2 - 1) - (rsi_up * 2 - 1)  # -2, 0, +2

    # ── 3. Trend göstergeleri (Enhanced)
    sma10  = c.rolling(10).mean()
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    ema10  = ta.trend.EMAIndicator(c, window=10).ema_indicator()
    ema20  = ta.trend.EMAIndicator(c, window=20).ema_indicator()

    out["c_sma10"]     = c / sma10 - 1
    out["c_sma20"]     = c / sma20 - 1
    out["c_sma50"]     = c / sma50 - 1
    out["sma10_20"]    = sma10 / sma20 - 1       # kısa-orta trend
    out["sma20_50"]    = sma20 / sma50 - 1       # orta-uzun trend
    out["ema10_20"]    = ema10 / ema20 - 1        # kısa-orta trend
    out["c_ema10"]     = c / ema10 - 1

    # YENI: SMA crossover signals (0=below, 1=above)
    out["c_above_sma20"] = (c > sma20).astype(int)
    out["sma10_above_20"] = (sma10 > sma20).astype(int)
    
    # MACD (Enhanced)
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    out["macd_diff"]   = macd.macd_diff()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"]   = out["macd_diff"] - out["macd_signal"]

    # ── 4. Volatilite (Enhanced)
    atr14  = ta.volatility.AverageTrueRange(h, lo, c, window=14).average_true_range()
    atr7   = ta.volatility.AverageTrueRange(h, lo, c, window=7).average_true_range()
    out["atr_pct"]    = atr14 / c
    out["atr_ratio"]  = atr7 / atr14.replace(0, np.nan)  # volatility regime change

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    out["bb_pct"]     = bb.bollinger_pband()      # 0=alt band, 1=üst band
    out["bb_width"]   = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    # YENI: BB squeeze (width < median width → düşük volatilite → breakout beklentisi)
    bb_width_ma = out["bb_width"].rolling(20).mean()
    out["bb_squeeze"] = (out["bb_width"] < bb_width_ma).astype(int)

    # Z-score (Enhanced mean reversion signals)
    std20 = c.rolling(20).std().replace(0, np.nan)
    std50 = c.rolling(50).std().replace(0, np.nan)
    out["zscore_20"]  = (c - sma20) / std20
    out["zscore_50"]  = (c - sma50) / std50
    
    # YENI: Z-score extremeness (how far from mean)
    out["zscore_extreme_20"] = np.abs(out["zscore_20"])
    out["zscore_extreme_50"] = np.abs(out["zscore_50"])

    # Gerçekleşen volatilite karşılaştırması
    hv5   = ret1.rolling(5).std()
    hv20  = ret1.rolling(20).std()
    out["vol_regime"] = hv5 / hv20.replace(0, np.nan)   # >1 = artan vol
    
    # YENI: Volatility mean reversion (vol normalı yüksekse düşme beklentisi)
    vol_ma = hv20.rolling(30).mean()
    out["vol_zscore"] = (hv20 - vol_ma) / vol_ma.rolling(30).std().replace(0, np.nan)

    # ── 5. Hacim (Enhanced)
    vma20 = v.rolling(20).mean()
    out["vol_ratio"]  = v / vma20.replace(0, np.nan)
    out["vol_z"]      = (v - vma20) / v.rolling(20).std().replace(0, np.nan)

    # YENI: Volume acceleration (hacim trend)
    out["vol_accel"]  = out["vol_ratio"].rolling(3).mean().diff()

    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    out["obv_ret5"]   = obv.pct_change(5)
    out["obv_ma"]     = obv.rolling(20).mean()
    out["obv_above_ma"] = (obv > out["obv_ma"]).astype(int)

    # ── 6. Fiyat yapısı (Enhanced)
    out["gap"]         = (df["Open"] - c.shift(1)) / c.shift(1)
    hl_range           = (h - lo).replace(0, np.nan)
    out["close_pos"]   = (c - lo) / hl_range     # 0=dip 1=tepe
    out["hl_pct"]      = hl_range / c             # gün içi volatilite
    
    # YENI: Intraday direction (close vs open)
    out["intraday_dir"] = ((df["Close"] - df["Open"]) / df["Open"]).rolling(3).mean()
    # YENI: Upper/lower shadow ratio (wick patterns)
    upper_wick = h - np.maximum(df["Open"], c)
    lower_wick = np.minimum(df["Open"], c) - lo
    out["wick_ratio"] = upper_wick / lower_wick.replace(0, np.nan)

    # ── 7. Stationarity features (differenced returns — impulse response)
    out["ret_momentum"] = ret1.rolling(5).sum()  # 5-day return momentum
    out["ret_norm"] = (ret1 - ret1.rolling(20).mean()) / ret1.rolling(20).std().replace(0, np.nan)  # standardized returns

    # ── 8. Takvim (Consumer Staples için önemli)
    idx = pd.to_datetime(df.index)
    out["dow"]         = idx.dayofweek             # 0=Pzt 4=Cum
    out["month_end"]   = ((idx + pd.offsets.MonthEnd(0) - idx).days <= 3).astype(int)
    out["month"]       = idx.month                 # mevsimsellik
    
    # YENI: Haftanın son günü (Friday effect)
    out["dow_friday"]  = (idx.dayofweek == 4).astype(int)

    feat_df   = pd.DataFrame(out, index=df.index)
    target_df = make_targets(df)
    tgt_cols  = [c for c in target_df.columns if c.startswith("target_")]
    return pd.concat([df, feat_df, target_df[tgt_cols]], axis=1)


def get_feature_cols(full_df: pd.DataFrame) -> list:
    skip = {"Open","High","Low","Close","Volume","Ticker"}
    skip |= {c for c in full_df.columns if c.startswith("target_")}
    return [c for c in full_df.columns if c not in skip]
