"""
train.py - Model egitimi, hyperparameter tuning, sinyal metrikleri
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)

from config import RF_PARAM_DIST, LGB_PARAM_DIST, FEATURES, SIGNAL_THRESHOLD


def train_sector_models(sector_data_dict):
    """
    Her sektor icin 1M ve 3M horizon modelleri egitir.
    RF + LGB karsilastirmasi, sinyal metrikleri hesaplar.
    
    Returns: (sector_results, all_stock_metrics)
    """
    print("[3/7] Model egitimi ve optimizasyon basliyor...\n")

    tscv = TimeSeriesSplit(n_splits=3)
    sector_results = {}
    all_stock_metrics = []

    for sector_name, sdata in sector_data_dict.items():
        X = sdata["X"]
        full_train = sdata["full_train"]
        mask_train = sdata["mask_train"]
        mask_test = sdata["mask_test"]

        horizon_models = {}
        for horizon_name, target_col in [("1m", "target_1m"), ("3m", "target_3m")]:
            y = full_train[target_col]
            if y.index.tz is None:
                y.index = y.index.tz_localize("UTC")

            X_train, y_train = X[mask_train], y[mask_train]
            X_test, y_test = X[mask_test], y[mask_test]

            # --- A. Random Forest Regressor ---
            rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf_search = RandomizedSearchCV(
                rf_base, RF_PARAM_DIST,
                n_iter=20, cv=tscv, scoring='neg_mean_absolute_error',
                random_state=42, n_jobs=-1, error_score=0
            )
            rf_search.fit(X_train, y_train)
            rf_best = rf_search.best_estimator_

            rf_preds = rf_best.predict(X_test)
            rf_mae = mean_absolute_error(y_test, rf_preds)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
            rf_r2 = r2_score(y_test, rf_preds)

            # --- B. LightGBM Regressor ---
            lgb_base = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
            lgb_search = RandomizedSearchCV(
                lgb_base, LGB_PARAM_DIST,
                n_iter=20, cv=tscv, scoring='neg_mean_absolute_error',
                random_state=42, n_jobs=-1, error_score=0
            )
            lgb_search.fit(X_train, y_train)
            lgb_best = lgb_search.best_estimator_

            lgb_preds_test = lgb_best.predict(X_test)
            lgb_mae = mean_absolute_error(y_test, lgb_preds_test)
            lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds_test))
            lgb_r2 = r2_score(y_test, lgb_preds_test)

            # --- C. En iyi modeli sec (MAE en dusuk) ---
            if rf_mae <= lgb_mae:
                best_model = rf_best
                best_name = "RandomForest"
                best_mae, best_rmse, best_r2 = rf_mae, rf_rmse, rf_r2
                best_params = rf_search.best_params_
                best_preds = rf_preds
            else:
                best_model = lgb_best
                best_name = "LightGBM"
                best_mae, best_rmse, best_r2 = lgb_mae, lgb_rmse, lgb_r2
                best_params = lgb_search.best_params_
                best_preds = lgb_preds_test

            # --- D. HIBRIT SINYAL METRIKLERI ---
            y_true_signal = (y_test > SIGNAL_THRESHOLD).astype(int)
            y_pred_signal = (best_preds > SIGNAL_THRESHOLD).astype(int)
            rf_pred_signal = (rf_preds > SIGNAL_THRESHOLD).astype(int)
            lgb_pred_signal = (lgb_preds_test > SIGNAL_THRESHOLD).astype(int)

            sig_acc = accuracy_score(y_true_signal, y_pred_signal)
            sig_prec = precision_score(y_true_signal, y_pred_signal, zero_division=0)
            sig_recall = recall_score(y_true_signal, y_pred_signal, zero_division=0)
            sig_f1 = f1_score(y_true_signal, y_pred_signal, zero_division=0)

            rf_sig_acc = accuracy_score(y_true_signal, rf_pred_signal)
            rf_sig_prec = precision_score(y_true_signal, rf_pred_signal, zero_division=0)
            rf_sig_recall = recall_score(y_true_signal, rf_pred_signal, zero_division=0)
            rf_sig_f1 = f1_score(y_true_signal, rf_pred_signal, zero_division=0)

            lgb_sig_acc = accuracy_score(y_true_signal, lgb_pred_signal)
            lgb_sig_prec = precision_score(y_true_signal, lgb_pred_signal, zero_division=0)
            lgb_sig_recall = recall_score(y_true_signal, lgb_pred_signal, zero_division=0)
            lgb_sig_f1 = f1_score(y_true_signal, lgb_pred_signal, zero_division=0)

            print(f"    [{horizon_name.upper()}] RF  -> MAE: {rf_mae:.4f} | RMSE: {rf_rmse:.4f} | R2: {rf_r2:.4f}")
            print(f"    [{horizon_name.upper()}] LGB -> MAE: {lgb_mae:.4f} | RMSE: {lgb_rmse:.4f} | R2: {lgb_r2:.4f}")
            print(f"    [{horizon_name.upper()}] >>> KAZANAN: {best_name} (MAE: {best_mae:.4f})")
            print(f"    [{horizon_name.upper()}] SINYAL (>{SIGNAL_THRESHOLD*100:.0f}%): Acc:%{sig_acc*100:.1f} Prec:%{sig_prec*100:.1f} Rec:%{sig_recall*100:.1f} F1:%{sig_f1*100:.1f}")

            # --- E. HISSE BAZLI SINYAL METRIKLERI ---
            symbols_test = full_train["symbol"][mask_test]
            for stk in symbols_test.unique():
                stk_mask = symbols_test == stk
                if stk_mask.sum() < 10:
                    continue
                y_true_stk = (y_test[stk_mask] > SIGNAL_THRESHOLD).astype(int)
                y_pred_stk = (best_preds[stk_mask.values] > SIGNAL_THRESHOLD).astype(int)

                if y_true_stk.sum() > 0 or y_pred_stk.sum() > 0:
                    stk_acc = accuracy_score(y_true_stk, y_pred_stk)
                    stk_prec = precision_score(y_true_stk, y_pred_stk, zero_division=0)
                    stk_recall = recall_score(y_true_stk, y_pred_stk, zero_division=0)
                    stk_f1 = f1_score(y_true_stk, y_pred_stk, zero_division=0)
                    stk_mae = mean_absolute_error(y_test[stk_mask], best_preds[stk_mask.values])

                    all_stock_metrics.append({
                        "Hisse": stk, "Sektor": sector_name,
                        "Horizon": horizon_name.upper(), "Model": best_name,
                        "Accuracy": stk_acc, "Precision": stk_prec,
                        "Recall": stk_recall, "F1": stk_f1,
                        "MAE": stk_mae, "N_Test": int(stk_mask.sum())
                    })

            horizon_models[horizon_name] = {
                "best_model": best_model, "best_name": best_name,
                "best_params": best_params,
                "rf_scores": {"mae": rf_mae, "rmse": rf_rmse, "r2": rf_r2},
                "lgb_scores": {"mae": lgb_mae, "rmse": lgb_rmse, "r2": lgb_r2},
                "best_scores": {"mae": best_mae, "rmse": best_rmse, "r2": best_r2},
                "signal_scores": {"acc": sig_acc, "prec": sig_prec, "recall": sig_recall, "f1": sig_f1},
                "rf_signal_scores": {"acc": rf_sig_acc, "prec": rf_sig_prec, "recall": rf_sig_recall, "f1": rf_sig_f1},
                "lgb_signal_scores": {"acc": lgb_sig_acc, "prec": lgb_sig_prec, "recall": lgb_sig_recall, "f1": lgb_sig_f1}
            }

        sector_results[sector_name] = {
            "models": horizon_models,
            "stock_info": sdata["stock_info"]
        }
        print()

    return sector_results, all_stock_metrics
