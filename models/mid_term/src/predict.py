"""
predict.py - Hisse bazli fiyat tahmini ve sinyal uretimi
"""
from config import SIGNAL_THRESHOLD


def predict_stocks(sector_results):
    """
    Her hisse icin 1M ve 3M fiyat tahmini yapar.
    AL / BEKLE / SAT sinyali uretir.
    
    Returns: list of prediction dicts
    """
    print("[4/7] Hisse bazli fiyat tahmini yapiliyor...\n")

    all_predictions = []

    for sector_name, res in sector_results.items():
        horizon_models = res["models"]
        stock_info = res["stock_info"]

        for stock, info in stock_info.items():
            try:
                pred_1m = horizon_models["1m"]["best_model"].predict(info["last_features"])[0]
                pred_3m = horizon_models["3m"]["best_model"].predict(info["last_features"])[0]

                last_price = info["last_price"]
                price_1m = last_price * (1 + pred_1m)
                price_3m = last_price * (1 + pred_3m)

                sinyal_1m = "AL" if pred_1m > SIGNAL_THRESHOLD else ("BEKLE" if pred_1m > 0 else "SAT")
                sinyal_3m = "AL" if pred_3m > SIGNAL_THRESHOLD else ("BEKLE" if pred_3m > 0 else "SAT")

                all_predictions.append({
                    "Sektor": sector_name,
                    "Hisse": stock,
                    "Son_Fiyat": last_price,
                    "Getiri_1Ay": pred_1m,
                    "Tahmin_1Ay": price_1m,
                    "Sinyal_1Ay": sinyal_1m,
                    "Getiri_3Ay": pred_3m,
                    "Tahmin_3Ay": price_3m,
                    "Sinyal_3Ay": sinyal_3m,
                    "Model_1M": horizon_models["1m"]["best_name"],
                    "Model_3M": horizon_models["3m"]["best_name"]
                })
            except Exception as e:
                print(f"    [!] {stock} tahmin hatasi: {e}")

    return all_predictions
