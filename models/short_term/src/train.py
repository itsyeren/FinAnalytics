"""
train.py — Consumer Staples Short Model Training
Çalıştırma:
    poetry run python train.py              # tümü
    poetry run python train.py --ticker KO  # tek ticker
"""

import argparse, json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

from config import (ALL_TICKERS, HORIZONS, MODELS_DIR,
                    TRAIN_RATIO, VAL_RATIO,
                    LGBM_PARAMS, XGB_PARAMS, LOGREG_PARAMS, DATA_DIR, RANDOM_SEED)
from features import build_features, get_feature_cols

# ── Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)


# ── Veri yükleme (mevcut data_loader yerine inline — bağımlılık azaltır)
def load_ticker(ticker: str) -> pd.DataFrame | None:
    """
    Kaggle D1 formatı: AAPL.US_D1.csv, KO.US_D1.csv, BF.B.US_D1.csv
    ticker = 'AAPL', 'KO.US', 'BF.B.US' vb. olabilir.
    """
    # Ticker'ın base kısmını al (KO.US → KO, AAPL → AAPL)
    base = ticker.replace(".US", "") if ticker.endswith(".US") else ticker

    candidates = [
        DATA_DIR / f"{base}.US_D1.csv",       # AAPL.US_D1.csv  ✓
        DATA_DIR / f"{ticker}_D1.csv",         # KO.US_D1.csv
        DATA_DIR / f"{base.upper()}.US_D1.csv",
        DATA_DIR / f"{ticker}.csv",            # fallback
        DATA_DIR / f"{base}.csv",
    ]

    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip().title() for c in df.columns]
            df.rename(columns={"Adj Close": "Close", "Adj_Close": "Close"}, inplace=True)
            date_col = next((c for c in df.columns
                             if c.lower() in ("date","datetime","time")), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            df.index.name = "Date"
            needed = ["Open","High","Low","Close","Volume"]
            if not all(c in df.columns for c in needed):
                continue
            df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
            df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
            if len(df) >= 400:
                return df
        except Exception:
            continue
    return None


# ── Train / val / test split
def time_split(df):
    n  = len(df)
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def xy(split_df, target_col, feat_cols):
    X = split_df[feat_cols].values.astype(np.float32)
    y = split_df[target_col].values.astype(int)
    return X, y


# ── Model eğitimi
def train_lgbm(X_tr, y_tr, X_val, y_val, feat_cols, horizon: int = 5):
    """
    Consumer Staples için ayarlı LGBM.
    Horizon arttıkça daha az regularizasyon — uzun vadede sinyal daha temiz.
    """
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    pos_weight   = float(n_neg / n_pos) if n_pos else 1.0

    # Horizon bazlı parametre ayarı
    depth   = 4 if horizon <= 2 else 5
    leaves  = 16 if horizon <= 2 else 24
    lr      = 0.02 if horizon <= 2 else 0.03

    params = {
        "objective":          "binary",
        "metric":             "auc",
        "num_leaves":         leaves,
        "max_depth":          depth,
        "learning_rate":      lr,
        "feature_fraction":   0.7,
        "bagging_fraction":   0.8,
        "bagging_freq":       5,
        "min_data_in_leaf":   40,
        "lambda_l1":          0.3,
        "lambda_l2":          0.3,
        "pos_neg_sampling_ratio": pos_weight,
        "is_unbalance":       True,
        "verbose":            -1,
    }
    tr  = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feat_cols)
    val = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=tr)
    m   = lgb.train(
        params, tr, num_boost_round=2000,
        valid_sets=[tr, val], valid_names=["train","val"],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )
    return m


def train_xgb(X_tr, y_tr, X_val, y_val, horizon: int = 5):
    """
    XGBoost eğitimi: CONFIG parametrelerini kullanır.
    Otomatik scale_pos_weight hesaplaması ile class balancing.
    """
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = float(n_neg / n_pos) if n_pos else 1.0
    
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = spw
    
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def find_best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Validation seti üzerinde F1'i maksimize eden threshold'u bulur."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.05):
        pred = (probs >= t).astype(int)
        # İki sınıf da tahmin edilmeli
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2)


# ── Tek ticker eğit
def train_one(ticker: str):
    raw = load_ticker(ticker)
    if raw is None:
        print(f"  ⚠  {ticker}: veri bulunamadı")
        return {}

    full = build_features(raw)
    feat_cols = get_feature_cols(full)
    
    # ── YENI: NaN features'ı filtrele (>30% NaN olan özellikleri kaldır)
    nan_pct = full[feat_cols].isna().sum() / len(full)
    feat_cols = [c for c in feat_cols if nan_pct[c] <= 0.30]
    
    if len(feat_cols) < 10:
        print(f"  ⚠  {ticker}: çok az geçerli feature ({len(feat_cols)} < 10)")
        return {}
    
    results = {}

    for h in HORIZONS:
        target_col = f"target_{h}d"
        subset = full[feat_cols + [target_col]].dropna()
        if len(subset) < 400:
            continue

        tr_df, val_df, te_df = time_split(subset)
        X_tr, y_tr   = xy(tr_df,  target_col, feat_cols)
        X_val, y_val = xy(val_df, target_col, feat_cols)
        X_te, y_te   = xy(te_df,  target_col, feat_cols)

        save_dir = MODELS_DIR / ticker
        save_dir.mkdir(parents=True, exist_ok=True)

        h_results = {}
        algo_probs = {}  # ── YENI: ensemble confidence için
        
        for algo in ["lgbm", "xgb", "logreg"]:
            try:
                if algo == "lgbm":
                    m = train_lgbm(X_tr, y_tr, X_val, y_val, feat_cols, horizon=h)
                    val_prob = m.predict(X_val)
                    prob     = m.predict(X_te)
                    bundle   = {"model": m, "feat_cols": feat_cols, "scaler": None}

                elif algo == "xgb":
                    m = train_xgb(X_tr, y_tr, X_val, y_val, horizon=h)
                    val_prob = m.predict_proba(X_val)[:, 1]
                    prob     = m.predict_proba(X_te)[:, 1]
                    bundle   = {"model": m, "feat_cols": feat_cols, "scaler": None}

                else:
                    m, sc = train_logreg(X_tr, y_tr)
                    val_prob = m.predict_proba(sc.transform(X_val))[:, 1]
                    prob     = m.predict_proba(sc.transform(X_te))[:, 1]
                    bundle   = {"model": m, "feat_cols": feat_cols, "scaler": sc}

                algo_probs[algo] = prob  # ── YENI: ensemble için sakla
                
                # Validation üzerinde en iyi threshold'u bul
                best_t = find_best_threshold(val_prob, y_val)
                pred   = (prob >= best_t).astype(int)
                bundle["threshold"] = best_t  # modelle birlikte kaydet

                h_results[algo] = {
                    "accuracy": round(float(accuracy_score(y_te, pred)),    4),
                    "f1":       round(float(f1_score(y_te, pred,
                                                     zero_division=0)),     4),
                    "roc_auc":  round(float(roc_auc_score(y_te, prob)),     4),
                    "n_test":   int(len(y_te)),
                    "test_start": str(te_df.index[0].date()),
                    "test_end":   str(te_df.index[-1].date()),
                }
                with open(save_dir / f"{algo}_{h}d.pkl", "wb") as f:
                    pickle.dump(bundle, f)

            except Exception as e:
                h_results[algo] = {"error": str(e)}

        # ── YENI: Ensemble confidence score (3 model uyum seviyesi)
        if len(algo_probs) == 3:
            ensemble_prob = np.mean(list(algo_probs.values()), axis=0)
            
            ensemble_roc = roc_auc_score(y_te, ensemble_prob)
            for algo in h_results:
                if "roc_auc" in h_results[algo]:
                    h_results[algo]["ensemble_auc"] = round(ensemble_roc, 4)
        
        results[f"{h}d"] = h_results

    return results


# ── Ana döngü
def run(tickers):
    all_results = {}
    for ticker in tqdm(tickers, desc="Eğitim"):
        all_results[ticker] = train_one(ticker)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Özet
    rows = []
    for ticker, hmap in all_results.items():
        for h_key, algos in hmap.items():
            for algo, m in algos.items():
                if "error" in m: continue
                rows.append({"Ticker": ticker, "Horizon": h_key,
                              "Algo": algo.upper(), **m})
    if rows:
        df = pd.DataFrame(rows)
        best = df.loc[df.groupby(["Ticker","Horizon"])["roc_auc"].idxmax()]
        print("\n── En iyi modeller (ROC-AUC sırası) ──")
        print(best.sort_values("roc_auc", ascending=False)
                  [["Ticker","Horizon","Algo","accuracy","f1","roc_auc"]]
                  .to_string(index=False))
        print(f"\nOrtalama ROC-AUC: {best['roc_auc'].mean():.4f}")

    print(f"\n✓ Modeller → models/")
    print(f"✓ Rapor    → reports/training_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()
    tickers = [args.ticker] if args.ticker else ALL_TICKERS
    run(tickers)
