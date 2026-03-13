# Multi-Sektör Fiyat Tahmin Sistemi

**6 Sektör | 35 Hisse | 1Ay + 3Ay Fiyat Tahmini** — RandomForest + LightGBM ile hibrit regresyon+sinyal analizi.

## Proje Yapısı

```
multi_sektor_analiz/
├── data/                    # Çekilen veriler
├── models/                  # Eğitilmiş modeller
├── reports/                 # Üretilen raporlar (JSON)
│
├── config.py                # Sabitler, API, sektör tanımları, hyperparams
├── data_loader.py           # Alpaca API'den veri çekme
├── features.py              # 10 teknik özellik + regression targetlar
├── split_data.py            # Sektör bazlı veri hazırlama + train/test split
├── train.py                 # RF + LGB eğitimi + sinyal metrikleri
├── predict.py               # Fiyat tahmini + AL/BEKLE/SAT sinyalleri
├── generate_report.py       # Terminal tabloları ve raporlar
├── report_summary.py        # JSON çıktı üretimi
├── main.py                  # Ana orkestrasyon (python main.py)
└── README.md
```

## Kullanım

```bash
cd multi_sektor_analiz
python main.py
```

## Gereksinimler

- Python 3.10+
- `.env` dosyasında `ALPACA_API_KEY` ve `ALPACA_API_SECRET`

```
pip install pandas numpy scikit-learn lightgbm alpaca-py python-dotenv
```

## Pipeline Akışı

1. **data_loader** → Alpaca'dan 35 hisse + ETF verisi çeker
2. **features** → 10 teknik indikatör + 1Ay/3Ay getiri targetı hesaplar
3. **split_data** → Sektör bazlı birleştirme + 2024 öncesi/sonrası split
4. **train** → RandomizedSearchCV ile RF+LGB karşılaştırması
5. **predict** → Son fiyattan 1Ay+3Ay tahmin + sinyal üretimi
6. **generate_report** → Terminal tabloları (regression, sinyal, fiyat)
7. **report_summary** → `sector_optimized_params.json` kaydı

## Çıktılar

- **Terminal**: Sektör/hisse bazlı regression + sinyal kalitesi tabloları
- **JSON**: `sector_optimized_params.json` (modeller, parametreler, tahminler)
