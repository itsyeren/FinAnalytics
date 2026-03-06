"""
config.py — Consumer Staples Short Model
"""
from pathlib import Path

# ── Klasörler (Proje kök dizinine göreceli)
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parents[2]  # models/short_term/src -> models/short_term -> models -> root

DATA_DIR    = PROJECT_ROOT / "data" / "raw" / "D1"
MODELS_DIR  = PROJECT_ROOT / "models" / "short_term"

# ── Minimum satır sayısı (yeterli veri kontrolü)
MIN_ROWS = 200

# ── Reproducibility
RANDOM_SEED = 42

# ── Ticker listesi
# Kaggle'da ".US" eki olmadan geliyor — loader ikisini de dener
TICKERS = {
    "Apple":                           "AAPL",
    "Microsoft":                       "MSFT",
    "NVIDIA":                          "NVDA",
    "Conagra Brands":                  "CAG",
    "Hershey":                         "HSY",
    "Coca-Cola Europacific Partners":  "CCEP",
    "Kroger":                          "KR",
    "Sysco":                           "SYY",
    "Campbell Soup":                   "CPB",
    "Keurig Dr Pepper":                "KDP",
    "PepsiCo":                         "PEP",
    "Tyson Foods":                     "TSN",
    "JM Smucker":                      "SJM",
    "Kraft Heinz":                     "KHC",
    "Philip Morris":                   "PM",
    "Altria":                          "MO",
    "Hormel Foods":                    "HRL",
    "Estée Lauder":                    "EL",
    "Colgate-Palmolive":               "CL",
    "Kellogg":                         "K",
    "General Mills":                   "GIS",
    "Kimberly-Clark":                  "KMB",
    "Clorox":                          "CLX",
    "McCormick":                       "MKC",
    "Coca-Cola":                       "KO",
    "Walmart":                         "WMT",
    "Costco":                          "COST",
    "Dollar General":                  "DG",
    "Dollar Tree":                     "DLTR",
    "Walgreens":                       "WBA",
    "Monster Beverage":                "MNST",
    "Constellation Brands":            "STZ",
    "Mondelez":                        "MDLZ",
    "Molson Coors":                    "TAP",
    "Lamb Weston":                     "LW",
    "Church & Dwight":                 "CHD",
    "Brown-Forman":                    "BF.B",
}

# isim → ticker hızlı erişim
NAME_TO_TICKER = TICKERS
TICKER_TO_NAME = {v: k for k, v in TICKERS.items()}
ALL_TICKERS    = list(TICKERS.values())


# ── Yahoo Finance symbol converter
def to_yf_symbol(ticker: str) -> str:
    """
    Convert internal ticker format to Yahoo Finance format.
    AAPL → AAPL, KO.US → KO.US, BF.B → BF.B, etc.
    """
    # Most yfinance symbols are straightforward; add .US for clarity if needed
    if ticker.endswith(".US"):
        return ticker  # Already in YF format
    # For comp symbols like BF.B, KO.US, return as-is
    return ticker

# ── Model parametreleri
HORIZONS  = [1, 3, 5, 7]   # iş günü
THRESHOLD = 0.01           # UP/DOWN eşiği (0.01 = ≥%1 getiri gerekli → class dengesi)

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "num_leaves":       25,
    "learning_rate":    0.02,          # Daha düşük → daha smooth öğrenme
    "feature_fraction": 0.8,           # Feature sampling artırıldı
    "bagging_fraction": 0.85,          # Bagging artırıldı
    "bagging_freq":     5,
    "max_depth":        7,             # Biraz daha derin (overfitting kontrollü)
    "min_data_in_leaf": 100,
    "lambda_l1":        1.5,           # L1 cezası yükseltildi (feature sparsity)
    "lambda_l2":        1.5,           # L2 cezası yükseltildi
    "is_unbalance":     True,
    "verbose":         -1,
    "random_state":     RANDOM_SEED,
}

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        6,             # Daha derin
    "learning_rate":    0.02,          # Daha düşük rata
    "n_estimators":     1200,          # Daha fazla tree
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 60,            # İyileştirildi
    "reg_alpha":        1.5,
    "reg_lambda":       2.5,           # L2 artırıldı → smooth predictions
    "gamma":            0.1,           # Min loss reduction → overfitting prevented
    "verbosity":        0,
    "early_stopping_rounds": 50,
    "random_state":     RANDOM_SEED,
}

LOGREG_PARAMS = {
    "C":            0.02,              # Düşürüldü (daha strong regularization)
    "max_iter":     3000,              # Arttırıldı (convergence için)
    "solver":       "saga",
    "penalty":      "elasticnet",      # L1+L2 kombine
    "l1_ratio":     0.5,               # L1/L2 dengesi
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "multi_class":  "auto",
}

TRAIN_RATIO = 0.65              # Daha fazla eğitim verisi
VAL_RATIO   = 0.15             # Validation seti de biraz büyüt
