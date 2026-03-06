"""
config.py - Proje sabitleri, API ayarlari, sektor tanimlari
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient

# ==============================================================================
# API AYARLARI
# ==============================================================================
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# ==============================================================================
# EGITIM PARAMETRELERI
# ==============================================================================
TRAIN_START = "2018-01-01"
TRAIN_END = "2026-02-16"
SPLIT_DATE = "2024-01-01"

# Tahmin hedefleri (is gunu cinsinden)
HORIZON_1M = 20   # ~1 ay
HORIZON_3M = 63   # ~3 ay

# Sinyal esigi: Tahmin edilen getiri bu esigi asarsa AL sinyali
SIGNAL_THRESHOLD = 0.03  # %3

# ==============================================================================
# SEKTOR TANIMLARI (6 Sektor - 35 Hisse)
# ==============================================================================
SECTORS = {
    "Teknoloji": {
        "etf": "XLK",
        "stocks": ["AAPL", "MSFT", "NVDA"]
    },
    "Gida_Uretim": {
        "etf": "XLP",
        "stocks": ["CAG", "HSY", "CPB", "KDP", "TSN", "SJM", "KHC", "HRL", "GIS", "MKC", "MDLZ", "K", "LW"]
    },
    "Icecek": {
        "etf": "XLP",
        "stocks": ["PEP", "KO", "MNST", "STZ", "TAP"]
    },
    "Ev_Kisisel": {
        "etf": "XLP",
        "stocks": ["EL", "CL", "KMB", "CLX", "CHD"]
    },
    "Tutun": {
        "etf": "XLP",
        "stocks": ["PM", "MO"]
    },
    "Perakende_Temel": {
        "etf": "XLP",
        "stocks": ["WMT", "COST", "DG", "DLTR", "WBA", "KR", "SYY"]
    }
}

# Tum sembolleri topla
ALL_SYMBOLS = []
ALL_ETFS = []
for _info in SECTORS.values():
    ALL_SYMBOLS.extend(_info["stocks"])
    ALL_ETFS.append(_info["etf"])
ALL_TICKERS = list(set(ALL_SYMBOLS + ALL_ETFS))

# ==============================================================================
# FEATURE LISTESI (10 Ozellik)
# ==============================================================================
FEATURES = [
    "dist_sma_200", "dist_sma_50", "rsi", "volatility",
    "rel_strength", "is_sector_bullish", "volume_ratio",
    "macd_signal", "bb_position", "momentum_20d"
]

# ==============================================================================
# HYPERPARAMETER GRIDS (Regressor)
# ==============================================================================
RF_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_leaf': [5, 10, 20, 30],
    'min_samples_split': [5, 10, 20],
    'max_features': ['sqrt', 'log2', 0.5]
}

LGB_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [10, 20, 31, 50],
    'max_depth': [3, 5, 7, 10],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# ==============================================================================
# PROJE DIZINLERI (Proje kök dizinine göreceli)
# ==============================================================================
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parents[2]

DATA_DIR    = PROJECT_ROOT / "data" / "mid_term"
MODELS_DIR  = PROJECT_ROOT / "models" / "mid_term"
REPORTS_DIR = MODELS_DIR  # Raporları da aynı yere koyalım

for d in [DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
