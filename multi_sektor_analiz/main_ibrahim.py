"""
main.py - Multi-Sektor Fiyat Tahmin Sistemi
Ana orkestrasyon dosyasi: tum modulleri sirasiyla calistirir.

Kullanim:
    python main.py
"""
import warnings
warnings.filterwarnings("ignore")

from config import SECTORS, ALL_TICKERS, TRAIN_START, TRAIN_END

# ==============================================================================
# BASLIK
# ==============================================================================
print("=" * 80)
print("    MULTI-SEKTOR FIYAT TAHMIN SISTEMI")
print(f"    {len(SECTORS)} Sektor | 35 Hisse | 1Ay + 3Ay Fiyat Tahmini")
print("    RandomForest + LightGBM Regressor + RandomizedSearchCV")
print("=" * 80)
print(f"Egitim Araligi: {TRAIN_START} -> {TRAIN_END}")
print(f"Toplam {len(ALL_TICKERS)} sembol icin veri cekilecek.\n")

# ==============================================================================
# PIPELINE
# ==============================================================================

# 1. Veri cekme
from data_loader import fetch_all_data
df_all = fetch_all_data()

# 2. Sektor bazli veri hazirlama + split
from split_data import prepare_sector_data
sector_data = prepare_sector_data(df_all)

# 3. Model egitimi
from train import train_sector_models
sector_results, all_stock_metrics = train_sector_models(sector_data)

# 4. Tahmin
from predict import predict_stocks
all_predictions = predict_stocks(sector_results)

# 5-6. Raporlar
from generate_report import (
    print_regression_metrics, print_signal_metrics, print_top7_stock_metrics,
    print_all_stock_rankings,
    print_price_predictions, print_optimized_params, print_signal_summary
)

print_regression_metrics(sector_results)
print_signal_metrics(sector_results)
print_top7_stock_metrics(all_stock_metrics)
print_all_stock_rankings(all_stock_metrics)
df_preds = print_price_predictions(all_predictions)
print_optimized_params(sector_results)

# 7. JSON kaydet
from report_summary import save_json_report
print("\n[Kayit] Optimize parametreler JSON dosyasina kaydediliyor...")
save_json_report(sector_results, all_predictions, all_stock_metrics)

# 8. Sinyal ozeti
print_signal_summary(df_preds)

# ==============================================================================
# TAMAMLANDI
# ==============================================================================
print("\n" + "=" * 90)
print("[OK] Analiz tamamlandi.")
print("=" * 90)
