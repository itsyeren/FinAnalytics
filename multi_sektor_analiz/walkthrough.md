# Multi-Sektor Fiyat Tahmin Sistemi - Sonuc Raporu

## Ne Yapildi?

[allah.py](file:///c:/Users/funny/Desktop/hisse/multi_sektor_analiz/allah.py) dosyasinda **6 sektor, 35 hisse** icin:
- **Regression modeli** ile 1 ay ve 3 ay sonraki **tahmini fiyatlar** hesaplandi
- **Hibrit sinyal metrikleri**: Regresyon tahminleri >%3 esigine gore sinyale donusturulup **Accuracy, Precision, Recall, F1** hesaplandi
- Tum sonuclar [sector_optimized_params.json](file:///c:/Users/funny/Desktop/hisse/multi_sektor_analiz/sector_optimized_params.json) dosyasina kaydedildi

## Sinyal Kalitesi (Hibrit Metrikler)

Regresyon tahmini >%3 → AL sinyali olarak donusturuldu.

| Sektor | Hz | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Ev_Kisisel | 1M | LGB | %72.3 | **%83.0** | %6.6 | %12.2 |
| **Teknoloji** | **3M** | **RF** | **%57.3** | **%56.5** | **%96.0** | **%71.1** |
| Icecek | 1M | LGB | %72.5 | %51.4 | %3.4 | %6.4 |
| Icecek | 3M | LGB | %58.9 | %38.5 | %45.1 | %41.6 |
| Ev_Kisisel | 3M | RF | %63.7 | %34.9 | %22.9 | %27.7 |
| Gida_Uretim | 1M | LGB | %72.4 | %41.2 | %1.0 | %1.9 |
| Gida_Uretim | 3M | RF | %62.7 | %25.0 | %18.1 | %21.0 |
| Perakende | 3M | RF | %62.3 | %31.1 | %18.1 | %22.9 |
| Perakende | 1M | LGB | %59.3 | %20.6 | %2.7 | %4.8 |

## Sinyal Ozeti

- **1 Ay:** AL=0 | BEKLE=31 | SAT=4
- **3 Ay:** AL=22 | BEKLE=9 | SAT=4

## 3 Ay AL Sinyali Veren Hisseler (Top 10)

| Hisse | Son Fiyat | 3 Ay Tahmin | Getiri | Sektor |
|---|---|---|---|---|
| **COST** | $914 | $1,007 | +%10.2 | Perakende_Temel |
| **KMB** | $104 | $114 | +%9.6 | Ev_Kisisel |
| **NVDA** | $194 | $212 | +%9.2 | Teknoloji |
| **MSFT** | $511 | $554 | +%8.3 | Teknoloji |
| **AAPL** | $273 | $293 | +%7.2 | Teknoloji |
| **HRL** | $22 | $24 | +%6.4 | Gida_Uretim |
| **CAG** | $17 | $18 | +%6.1 | Gida_Uretim |
| **KDP** | $27 | $29 | +%6.0 | Gida_Uretim |
| **KHC** | $25 | $26 | +%5.7 | Gida_Uretim |
| **CLX** | $105 | $111 | +%5.6 | Ev_Kisisel |

## Dogrulama

- `python allah.py` - Exit code: 0
- 35/35 hisse basariyla analiz edildi
- `sector_optimized_params.json` basariyla guncellendi (regression + sinyal metrikleri)
