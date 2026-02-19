# Multi-Sektor Fiyat Tahmin Sistemi

6 sektorde toplam 35 hisse icin **Regression modeli** ile 1 ay ve 3 ay sonraki tahmini fiyatlari hesaplayan sistem.

## Model Degisikligi

Classification (yukselir/yukselmez) yerine **Regression (fiyat tahmini)** kullanilir:
- **Target 1M:** 20 is gunu sonraki yuzdesel getiri
- **Target 3M:** 63 is gunu sonraki yuzdesel getiri
- **Metrikler:** MAE, RMSE, R2 (eski Accuracy/Precision/F1 yerine)
- **Modeller:** RandomForestRegressor + LGBMRegressor

## Sektor Tanimlari

| Sektor | ETF | Hisseler |
|---|---|---|
| Teknoloji | XLK | AAPL, MSFT, NVDA |
| Gida_Uretim | XLP | CAG, HSY, CPB, KDP, TSN, SJM, KHC, HRL, GIS, MKC, MDLZ, K, LW |
| Icecek | XLP | PEP, KO, MNST, STZ, TAP |
| Ev_Kisisel | XLP | EL, CL, KMB, CLX, CHD |
| Tutun | XLP | PM, MO |
| Perakende_Temel | XLP | WMT, COST, DG, DLTR, WBA, KR, SYY |

## Cikti

- **Terminal:** Hisse bazli 1AY + 3AY fiyat tahmin tablosu
- **JSON:** `sector_optimized_params.json` - model parametreleri + tahminler

## Verification

- `cd multi_sektor_analiz && python allah.py`
- `sector_optimized_params.json` kontrolu
