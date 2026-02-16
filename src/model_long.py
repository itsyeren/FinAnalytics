"""
Long-term Stock Prediction Model
==================================
This module trains a LightGBM classifier to predict future stock performance
using cross-sectional ranking approach.

Key Features:
- Walk-forward validation with 5 folds
- Cross-sectional labeling (top 30% vs bottom 30%)
- Multiple evaluation metrics (AUC, Precision@10, IC)
- Feature importance analysis
- Model persistence
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import UNIVERSE
from src.features import add_features
from src.labels_long import add_long_score

# =========================
# CONFIGURATION
# =========================
class Config:
    """Model configuration parameters"""

    # Data parameters
    MIN_DATE = "2010-01-04"
    DATA_PATH = PROJECT_ROOT / "data/raw/D1"

    # Model parameters
    LGBM_PARAMS = {
        'n_estimators': 800,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Features to use
    FEATURE_COLS = [
        "ret_1",
        "ret_5",
        "ret_21",
        "mom_63",
        "mom_126",
        "vol_21",
        "vol_63",
        "ma_ratio_21_63",
        "drawdown_63"
    ]

    # Walk-forward validation parameters
    MIN_TRAIN_DAYS = 756   # ~3 years
    TEST_DAYS = 126        # ~6 months
    STEP_DAYS = 63         # ~3 months

    # Cross-sectional labeling
    TOP_PERCENTILE = 0.70   # Top 30% (above 0.70)
    BOTTOM_PERCENTILE = 0.30  # Bottom 30% (below 0.30)

    # Model output
    MODEL_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Strategy type - IMPORTANT: Choose one!
    STRATEGY_TYPE = "MOMENTUM"  # Options: "MOMENTUM" or "MEAN_REVERSION"

# =========================
# DATA LOADING
# =========================
def load_universe_data(universe: List[str], data_path: Path) -> pd.DataFrame:
    """
    Load and combine data for all tickers in universe

    Args:
        universe: List of ticker symbols
        data_path: Path to data directory

    Returns:
        Combined DataFrame with all tickers
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Find available files
    files = glob.glob(str(data_path / "*.US_D1.csv"))
    available = {
        os.path.basename(f).split(".")[0]
        for f in files
    }

    # Filter valid universe
    valid_universe = [t for t in universe if t in available]
    missing = [t for t in universe if t not in available]

    print(f"✓ Available tickers: {len(valid_universe)}")
    if missing:
        print(f"✗ Missing tickers: {len(missing)}")
        print(f"  First 20: {missing[:20]}")

    # Load data
    dfs = []
    for ticker in valid_universe:
        path = data_path / f"{ticker}.US_D1.csv"
        df_tmp = pd.read_csv(path)
        df_tmp["ticker"] = ticker
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    # Filter by date
    df = df[df["datetime"] >= Config.MIN_DATE].copy()

    print(f"✓ Total rows: {len(df):,}")
    print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"✓ Tickers: {df['ticker'].nunique()}")

    # Data quality checks
    duplicates = df.duplicated(subset=["ticker", "datetime"]).sum()
    if duplicates > 0:
        print(f"⚠ Found {duplicates} duplicate rows - removing...")
        df = df.drop_duplicates(subset=["ticker", "datetime"])

    return df

# =========================
# FEATURE ENGINEERING
# =========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical features and target labels

    Args:
        df: Raw price data

    Returns:
        DataFrame with features and labels
    """
    print("\n" + "=" * 60)
    print("ENGINEERING FEATURES")
    print("=" * 60)

    # Add technical features
    print("Adding technical features...")
    df = add_features(df)

    # Add forward returns (target)
    print("Adding forward return labels...")
    df = add_long_score(df)

    # Remove rows with missing features
    initial_rows = len(df)
    df = df.dropna(subset=Config.FEATURE_COLS + ["ret_long_norm"])
    dropped_rows = initial_rows - len(df)

    print(f"✓ Features added: {len(Config.FEATURE_COLS)}")
    print(f"✓ Rows after dropna: {len(df):,} (dropped {dropped_rows:,})")

    return df

# =========================
# CROSS-SECTIONAL LABELING
# =========================
def create_cross_sectional_labels(
    df: pd.DataFrame,
    strategy_type: str = "MOMENTUM"
) -> pd.DataFrame:
    """
    Create cross-sectional labels based on ranking

    Args:
        df: DataFrame with ret_long_norm column
        strategy_type: "MOMENTUM" or "MEAN_REVERSION"

    Returns:
        DataFrame with long_label column
    """
    df = df.copy()

    # Rank stocks by forward returns within each date
    df["rank_pct"] = df.groupby("datetime")["ret_long_norm"].rank(pct=True)

    # Initialize as Neutral
    df["long_label"] = "Neutral"

    if strategy_type == "MOMENTUM":
        # Buy winners (top 30%), sell losers (bottom 30%)
        df.loc[df["rank_pct"] >= Config.TOP_PERCENTILE, "long_label"] = "Up"
        df.loc[df["rank_pct"] <= Config.BOTTOM_PERCENTILE, "long_label"] = "Down"

    elif strategy_type == "MEAN_REVERSION":
        # Buy losers (bottom 30%), sell winners (top 30%)
        df.loc[df["rank_pct"] >= Config.TOP_PERCENTILE, "long_label"] = "Down"
        df.loc[df["rank_pct"] <= Config.BOTTOM_PERCENTILE, "long_label"] = "Up"

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return df

# =========================
# WALK-FORWARD SPLITS
# =========================
def create_walk_forward_splits(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Create walk-forward validation splits

    Args:
        df: DataFrame with datetime column

    Returns:
        List of (train_end, test_start, test_end) tuples
    """
    dates = pd.Index(df["datetime"].unique()).sort_values()

    splits = []
    for i in range(Config.MIN_TRAIN_DAYS, len(dates) - Config.TEST_DAYS, Config.STEP_DAYS):
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end = dates[i + Config.TEST_DAYS - 1]
        splits.append((train_end, test_start, test_end))

    return splits

# =========================
# EVALUATION METRICS
# =========================
def precision_at_topk(y_true: pd.Series, y_score: np.ndarray, topk_frac: float = 0.10) -> float:
    """
    Calculate precision at top K%

    Args:
        y_true: True binary labels
        y_score: Predicted scores
        topk_frac: Fraction of top predictions to consider

    Returns:
        Precision at top K%
    """
    n = len(y_true)
    k = max(1, int(n * topk_frac))
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true.iloc[idx].mean())

def ic_spearman(y_score: np.ndarray, fwd_ret: np.ndarray) -> float:
    """
    Calculate Information Coefficient (Spearman correlation)

    Args:
        y_score: Predicted scores
        fwd_ret: Forward returns

    Returns:
        Spearman correlation coefficient
    """
    corr, _ = spearmanr(y_score, fwd_ret, nan_policy="omit")
    return float(corr) if not np.isnan(corr) else 0.0

# =========================
# MODEL TRAINING
# =========================
def train_and_evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold_num: int,
    strategy_type: str
) -> Dict:
    """
    Train model on one fold and evaluate

    Args:
        train: Training data
        test: Test data
        fold_num: Fold number for logging
        strategy_type: MOMENTUM or MEAN_REVERSION

    Returns:
        Dictionary with evaluation metrics
    """
    # Create labels
    train = create_cross_sectional_labels(train, strategy_type)
    test = create_cross_sectional_labels(test, strategy_type)

    # Filter to binary classification (remove Neutral)
    train_bin = train[train["long_label"] != "Neutral"].copy()
    test_bin = test[test["long_label"] != "Neutral"].copy()

    if len(train_bin) == 0 or len(test_bin) == 0:
        print(f"⚠ Fold {fold_num}: No data after filtering neutral labels")
        return None

    # Create target (1 = Up, 0 = Down)
    y_train = (train_bin["long_label"] == "Up").astype(int)
    y_test = (test_bin["long_label"] == "Up").astype(int)

    # Features
    X_train = train_bin[Config.FEATURE_COLS]
    X_test = test_bin[Config.FEATURE_COLS]

    # Check for missing values
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        print(f"⚠ Fold {fold_num}: Missing values in features")
        return None

    # Train model
    model = LGBMClassifier(**Config.LGBM_PARAMS)
    model.fit(X_train, y_train)

    # Predict
    proba_up = model.predict_proba(X_test)[:, 1]

    # For momentum: high probability = buy
    # For mean reversion: would need different interpretation
    score_up = proba_up

    # Evaluate
    try:
        auc = roc_auc_score(y_test, score_up)
    except:
        auc = np.nan

    p10 = precision_at_topk(
        y_test.reset_index(drop=True),
        score_up,
        0.10
    )

    ic = ic_spearman(score_up, test_bin["ret_long_norm"].values)

    # Additional metrics
    y_pred = (score_up >= 0.5).astype(int)
    accuracy = (y_pred == y_test).mean()

    # Class balance
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()

    return {
        "fold": fold_num,
        "model": model,
        "train_samples": len(train_bin),
        "test_samples": len(test_bin),
        "train_pos_rate": train_pos_rate,
        "test_pos_rate": test_pos_rate,
        "auc": auc,
        "accuracy": accuracy,
        "p10": p10,
        "ic": ic,
        "test_start": test["datetime"].min(),
        "test_end": test["datetime"].max()
    }

# =========================
# MAIN TRAINING PIPELINE
# =========================
def train_model():
    """Main training pipeline"""

    print("\n" + "=" * 60)
    print("LONG-TERM STOCK PREDICTION MODEL")
    print("=" * 60)
    print(f"Strategy Type: {Config.STRATEGY_TYPE}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories
    Config.MODEL_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)

    # Load data
    df = load_universe_data(UNIVERSE, Config.DATA_PATH)

    # Engineer features
    df = engineer_features(df)

    # Create walk-forward splits
    print("\n" + "=" * 60)
    print("CREATING WALK-FORWARD SPLITS")
    print("=" * 60)

    splits = create_walk_forward_splits(df)
    print(f"✓ Number of folds: {len(splits)}")
    print(f"✓ First split: Train until {splits[0][0].date()}, Test {splits[0][1].date()} to {splits[0][2].date()}")
    print(f"✓ Last split: Train until {splits[-1][0].date()}, Test {splits[-1][1].date()} to {splits[-1][2].date()}")

    # Train on each fold
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION")
    print("=" * 60)

    results = []
    models = []

    for k, (train_end, test_start, test_end) in enumerate(splits):
        print(f"\n--- Fold {k} ---")
        print(f"Train: <= {train_end.date()}")
        print(f"Test: {test_start.date()} to {test_end.date()}")

        # Split data
        train = df[df["datetime"] <= train_end].copy()
        test = df[(df["datetime"] >= test_start) & (df["datetime"] <= test_end)].copy()

        print(f"Train samples: {len(train):,} | Test samples: {len(test):,}")

        # Train and evaluate
        result = train_and_evaluate_fold(train, test, k, Config.STRATEGY_TYPE)

        if result is not None:
            results.append(result)
            models.append(result["model"])

            print(f"✓ AUC: {result['auc']:.4f} | Accuracy: {result['accuracy']:.4f}")
            print(f"✓ P@10: {result['p10']:.4f} | IC: {result['ic']:.4f}")
            print(f"✓ Train pos rate: {result['train_pos_rate']:.3f} | Test pos rate: {result['test_pos_rate']:.3f}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    # Drop model column for display
    display_df = results_df.drop(columns=['model'], errors='ignore')
    print("\nFold-by-fold results:")
    print(display_df.to_string(index=False))

    print("\n" + "-" * 60)
    print("AVERAGE METRICS:")
    print("-" * 60)

    metrics = ['auc', 'accuracy', 'p10', 'ic']
    for metric in metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")

    # Train final model on all data
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 60)

    df_labeled = create_cross_sectional_labels(df, Config.STRATEGY_TYPE)
    df_final = df_labeled[df_labeled["long_label"] != "Neutral"].copy()

    y_final = (df_final["long_label"] == "Up").astype(int)
    X_final = df_final[Config.FEATURE_COLS]

    final_model = LGBMClassifier(**Config.LGBM_PARAMS)
    final_model.fit(X_final, y_final)

    print(f"✓ Final model trained on {len(X_final):,} samples")
    print(f"✓ Positive class rate: {y_final.mean():.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': Config.FEATURE_COLS,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))

    # Save model
    model_path = Config.MODEL_DIR / "long_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Save results
    results_path = Config.RESULTS_DIR / f"long_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    display_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")

    # Save feature importance
    fi_path = Config.RESULTS_DIR / "feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"✓ Feature importance saved to: {fi_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return final_model, results_df, feature_importance

# =========================
# PLOTTING FUNCTIONS
# =========================
def plot_results(results_df: pd.DataFrame):
    """Plot training results"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Across Folds', fontsize=16)

    metrics = ['auc', 'accuracy', 'p10', 'ic']
    titles = ['AUC', 'Accuracy', 'Precision@10%', 'Information Coefficient']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.plot(results_df['fold'], results_df[metric], marker='o', linewidth=2, markersize=8)
        ax.axhline(results_df[metric].mean(), color='red', linestyle='--',
                   label=f'Mean: {results_df[metric].mean():.4f}')
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = Config.RESULTS_DIR / f"performance_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance plot saved to: {plot_path}")

    plt.close()

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":

    # Run training
    final_model, results_df, feature_importance = train_model()

    # Plot results
    try:
        plot_results(results_df)
    except Exception as e:
        print(f"⚠ Could not create plots: {e}")

    print("\n✅ All done!")
