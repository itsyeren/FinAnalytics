"""
generate_report.py - Terminal raporlari (tablolar, ozetler)
"""
import pandas as pd
from config import SIGNAL_THRESHOLD


def print_regression_metrics(sector_results):
    """Sektor bazli regression metrikleri tablosu."""
    print("\n[5/7] Regression ve sinyal metrik raporlari hazirlaniyor...\n")

    print("=" * 110)
    print("    SEKTOR BAZLI REGRESSION METRIKLERI")
    print("=" * 110)
    print(f"{'SEKTOR':<16} {'HORIZON':<8} {'KAZANAN':<14} {'MAE':<10} {'RMSE':<10} {'R2':<10} {'RF_MAE':<10} {'LGB_MAE':<10}")
    print("-" * 110)

    for sector_name, res in sector_results.items():
        for hz in ["1m", "3m"]:
            m = res["models"][hz]
            s = m["best_scores"]
            rf_m = m["rf_scores"]["mae"]
            lgb_m = m["lgb_scores"]["mae"]
            winner = "[RF]" if m["best_name"] == "RandomForest" else "[LGB]"
            print(f"{sector_name:<16} {hz.upper():<8} {winner:<14} {s['mae']:<10.4f} {s['rmse']:<10.4f} {s['r2']:<10.4f} {rf_m:<10.4f} {lgb_m:<10.4f}")

    print("-" * 110)


def print_signal_metrics(sector_results):
    """Sektor bazli sinyal kalitesi tablosu."""
    print(f"\n\n    SEKTOR BAZLI SINYAL KALITESI (Esik: >{SIGNAL_THRESHOLD*100:.0f}% getiri = AL sinyali)")
    print("=" * 130)
    print(f"{'SEKTOR':<16} {'HZ':<5} {'MODEL':<12} {'ACCURACY':<10} {'PRECISION':<11} {'RECALL':<10} {'F1':<10} {'RF_PREC':<10} {'RF_F1':<10} {'LGB_PREC':<10} {'LGB_F1':<10}")
    print("-" * 130)

    for sector_name, res in sector_results.items():
        for hz in ["1m", "3m"]:
            m = res["models"][hz]
            ss, rfs, lgbs = m["signal_scores"], m["rf_signal_scores"], m["lgb_signal_scores"]
            winner = "[RF]" if m["best_name"] == "RandomForest" else "[LGB]"
            print(f"{sector_name:<16} {hz.upper():<5} {winner:<12} %{ss['acc']*100:<8.1f} %{ss['prec']*100:<9.1f} %{ss['recall']*100:<8.1f} %{ss['f1']*100:<8.1f} %{rfs['prec']*100:<8.1f} %{rfs['f1']*100:<8.1f} %{lgbs['prec']*100:<8.1f} %{lgbs['f1']*100:<8.1f}")

    print("-" * 130)


def print_top7_stock_metrics(all_stock_metrics):
    """Top 7 hisse bazli sinyal kalitesi tablosu."""
    if not all_stock_metrics:
        return

    df_stk = pd.DataFrame(all_stock_metrics)

    print(f"\n\n    TOP 7 HISSE BAZLI SINYAL KALITESI (F1 Skoruna Gore)")
    print(f"    Regresyon tahmini sinyale donusturulup Accuracy/Precision/Recall/F1 hesaplandi")
    print("=" * 130)
    print(f"{'SIRA':<5} {'HISSE':<8} {'SEKTOR':<16} {'HZ':<5} {'MODEL':<12} {'ACCURACY':<10} {'PRECISION':<11} {'RECALL':<10} {'F1':<10} {'MAE':<10} {'N_TEST':<8}")
    print("-" * 130)

    top7 = df_stk.nlargest(7, "F1")
    for i, (_, row) in enumerate(top7.iterrows(), 1):
        print(f"{i:<5} {row['Hisse']:<8} {row['Sektor']:<16} {row['Horizon']:<5} {row['Model']:<12} %{row['Accuracy']*100:<8.1f} %{row['Precision']*100:<9.1f} %{row['Recall']*100:<8.1f} %{row['F1']*100:<8.1f} {row['MAE']:<10.4f} {row['N_Test']:<8}")

    print("-" * 130)

    worst3 = df_stk[df_stk["F1"] > 0].nsmallest(3, "F1")
    if not worst3.empty:
        print(f"\n    EN DUSUK SINYAL KALITESI (F1 Skoruna Gore, Alt 3):")
        for _, row in worst3.iterrows():
            print(f"    >> {row['Hisse']:<6} [{row['Sektor']}] {row['Horizon']} Prec:%{row['Precision']*100:.1f} Rec:%{row['Recall']*100:.1f} F1:%{row['F1']*100:.1f}")

    print(f"\nYORUM: Precision = 'AL' dediginde gercekten yukselme orani")
    print(f"        Recall   = Gercek yukselislerin ne kadari yakalandi")
    print(f"        F1       = Precision ve Recall'in harmonik ortalamasi")


def print_all_stock_rankings(all_stock_metrics):
    """Tum hisselerin F1 skoruna gore siralamasini yazdirir (bir seferlik)."""
    if not all_stock_metrics:
        return

    df_stk = pd.DataFrame(all_stock_metrics)
    df_sorted = df_stk.sort_values("F1", ascending=False).reset_index(drop=True)

    print(f"\n\n{'='*140}")
    print(f"    TUM HISSELER - SINYAL KALITESI SIRALAMASI (F1 Skoruna Gore)")
    print(f"    Toplam {len(df_sorted)} kayit (hisse x horizon)")
    print(f"{'='*140}")
    print(f"{'SIRA':<5} {'HISSE':<8} {'SEKTOR':<16} {'HZ':<5} {'MODEL':<12} {'ACCURACY':<10} {'PRECISION':<11} {'RECALL':<10} {'F1':<10} {'MAE':<10} {'N_TEST':<8}")
    print("-" * 140)

    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<5} {row['Hisse']:<8} {row['Sektor']:<16} {row['Horizon']:<5} {row['Model']:<12} %{row['Accuracy']*100:<8.1f} %{row['Precision']*100:<9.1f} %{row['Recall']*100:<8.1f} %{row['F1']*100:<8.1f} {row['MAE']:<10.4f} {row['N_Test']:<8}")

    print("-" * 140)

    # Ozet istatistikler
    print(f"\n    OZET ISTATISTIKLER:")
    print(f"    Ortalama F1:        %{df_sorted['F1'].mean()*100:.1f}")
    print(f"    Medyan F1:          %{df_sorted['F1'].median()*100:.1f}")
    print(f"    Ortalama Precision: %{df_sorted['Precision'].mean()*100:.1f}")
    print(f"    Ortalama Recall:    %{df_sorted['Recall'].mean()*100:.1f}")
    print(f"    Ortalama MAE:       {df_sorted['MAE'].mean():.4f}")

    # F1 = 0 olan hisseler
    zero_f1 = df_sorted[df_sorted["F1"] == 0]
    if not zero_f1.empty:
        print(f"\n    UYARI: {len(zero_f1)} kayit F1=0 (sinyal uretilemiyor veya hicbir sinyal dogru degil):")
        for _, row in zero_f1.iterrows():
            print(f"      >> {row['Hisse']:<6} [{row['Sektor']}] {row['Horizon']}")

    print(f"\n{'='*140}")


def print_price_predictions(all_predictions):
    """Hisse bazli fiyat tahmin tablosu."""
    print("\n\n[6/7] Hisse bazli fiyat tahmin tablosu...\n")

    print("=" * 120)
    print("    HISSE BAZLI 1AY + 3AY FIYAT TAHMINI")
    print("=" * 120)

    df_preds = pd.DataFrame(all_predictions)
    if df_preds.empty:
        return df_preds

    df_preds = df_preds.sort_values("Getiri_3Ay", ascending=False)

    print(f"{'HISSE':<8} {'SEKTOR':<16} {'SON FIYAT':<12} {'1AY TAHMIN':<14} {'1AY GETIRI':<12} {'3AY TAHMIN':<14} {'3AY GETIRI':<12} {'MODEL 1M':<12} {'MODEL 3M':<12}")
    print("-" * 120)

    for _, row in df_preds.iterrows():
        g1 = f"%{row['Getiri_1Ay']*100:+.1f}"
        g3 = f"%{row['Getiri_3Ay']*100:+.1f}"
        dir_1m = "[+]" if row['Getiri_1Ay'] > 0 else "[-]"
        dir_3m = "[+]" if row['Getiri_3Ay'] > 0 else "[-]"
        print(f"{row['Hisse']:<8} {row['Sektor']:<16} ${row['Son_Fiyat']:<11.2f} ${row['Tahmin_1Ay']:<13.2f} {dir_1m}{g1:<10} ${row['Tahmin_3Ay']:<13.2f} {dir_3m}{g3:<10} {row['Model_1M']:<12} {row['Model_3M']:<12}")

    print("-" * 120)
    print(f"\nToplam {len(df_preds)} hisse analiz edildi.")

    # En cok yukselis/dusus beklenenler
    for label, col, n, asc in [
        ("EN COK YUKSELIS BEKLENEN (1 AY)", "Getiri_1Ay", 5, False),
        ("EN COK YUKSELIS BEKLENEN (3 AY)", "Getiri_3Ay", 5, False),
        ("EN COK DUSUS BEKLENEN (1 AY)", "Getiri_1Ay", 5, True),
        ("EN COK DUSUS BEKLENEN (3 AY)", "Getiri_3Ay", 5, True),
    ]:
        tahmin_col = "Tahmin_1Ay" if "1 AY" in label else "Tahmin_3Ay"
        subset = df_preds.nsmallest(n, col) if asc else df_preds.nlargest(n, col)
        print(f"\n--- {label} ---")
        for _, row in subset.iterrows():
            print(f"  >> {row['Hisse']} (${row['Son_Fiyat']:.2f}) -> ${row[tahmin_col]:.2f} (%{row[col]*100:+.1f}) [{row['Sektor']}]")

    return df_preds


def print_optimized_params(sector_results):
    """Optimize edilmis parametreler raporu."""
    print("\n\n    SEKTOR BAZLI OPTIMIZE PARAMETRELER")
    print("=" * 90)

    for sector_name, res in sector_results.items():
        for hz in ["1m", "3m"]:
            m = res["models"][hz]
            print(f"\n>> {sector_name} [{hz.upper()}] ({m['best_name']}):")
            for k, v in sorted(m["best_params"].items()):
                print(f"   {k}: {v}")

    print("\n" + "=" * 90)


def print_signal_summary(df_preds):
    """Sinyal ozet tablosu."""
    print("\n[7/7] Sinyal ozet tablosu...\n")

    if df_preds.empty:
        return

    print("=" * 90)
    print("    SINYAL OZET TABLOSU")
    print("=" * 90)

    for hz_label, sinyal_col, getiri_col, tahmin_col in [
        ("1 AY", "Sinyal_1Ay", "Getiri_1Ay", "Tahmin_1Ay"),
        ("3 AY", "Sinyal_3Ay", "Getiri_3Ay", "Tahmin_3Ay")
    ]:
        al_count = len(df_preds[df_preds[sinyal_col] == "AL"])
        bekle_count = len(df_preds[df_preds[sinyal_col] == "BEKLE"])
        sat_count = len(df_preds[df_preds[sinyal_col] == "SAT"])
        print(f"\n  {hz_label}: AL={al_count} | BEKLE={bekle_count} | SAT={sat_count}")

        if al_count > 0:
            al_hisseler = df_preds[df_preds[sinyal_col] == "AL"].sort_values(getiri_col, ascending=False)
            print(f"  AL sinyali veren hisseler:")
            for _, row in al_hisseler.iterrows():
                print(f"    >> {row['Hisse']:<6} ${row['Son_Fiyat']:.2f} -> ${row[tahmin_col]:.2f} (%{row[getiri_col]*100:+.1f}) [{row['Sektor']}]")

    print("\n" + "=" * 90)
