# 📈 FinAnalytics

**FinAnalytics**, ABD borsasında işlem gören hisse senetleri için kısa, orta ve uzun vadeli makine öğrenmesi modelleri, güncel haber bülteni ve Türkçe finans asistanı sunan bir Streamlit uygulamasıdır.

---

## 🗂️ Proje Yapısı

```
FinAnalytics/
│
├── app/                          # Streamlit uygulama dosyaları
│   ├── Analiz.py                 # Ana analiz sayfası (giriş noktası)
│   ├── assets/
│   │   └── logos/                # Hisse senedi logoları (.png)
│   └── pages/
│       └── Portföy.py            # Portföy yönetimi ve model tahmin sayfası
│
├── src/                          # Uygulama mantığı (dashboard renderer'lar, entegrasyonlar)
│   ├── short.py                  # Kısa vadeli dashboard (1-7 gün)
│   ├── mid.py                    # Orta vadeli dashboard (1-3 ay)
│   ├── long.py                   # Uzun vadeli dashboard (3+ ay)
│   ├── config.py                 # Genel yapılandırma (uzun model evren, özellikler)
│   ├── data.py                   # Veri yükleme yardımcıları
│   ├── features.py               # Uzun model özellik mühendisliği
│   ├── model_long.py             # Uzun model eğitim scripti
│   ├── predict.py                # Uzun model tahmin yardımcıları
│   ├── integrations/
│   │   ├── gemini.py             # Google Gemini API client (RAG)
│   │   └── marketaux.py          # Marketaux API client (haberler)
│   ├── rag/
│   │   └── turkish_finance_sft_rag.py  # Semantik retrieval motoru
│   └── reports/
│       ├── pdf_builder.py        # PDF rapor oluşturma
│       └── news_prompt.py        # Haber özetleme prompt şablonları
│
├── models/
│   ├── short_term/               # Kısa vadeli modeller
│   │   ├── AAPL/ … WMT/          # Ticker başına LightGBM + XGBoost (.pkl)
│   │   └── src/                  # Eğitim kaynak kodları (config, train, features…)
│   ├── mid_term/                 # Orta vadeli modeller
│   │   ├── sector_optimized_params.json  # Sektör bazlı optimizasyon sonuçları
│   │   └── src/                  # Eğitim kaynak kodları
│   └── long_term/
│       └── long_model.pkl        # Uzun vadeli LightGBM modeli
│
├── data/
│   ├── all_stocks.csv            # Tüm hisselerin birleşik OHLCV verisi
│   ├── raw/D1/                   # Ham günlük OHLCV CSV'ler (ticker bazında)
│   ├── news_cache/               # Günlük haber önbelleği (JSON)
│   └── outputs/                  # Model eğitim çıktıları, grafikler, raporlar
│
├── research/                     # Ar-Ge, backtest ve validasyon scriptleri
├── pyproject.toml                # Bağımlılıklar (Poetry)
└── README.md
```

---

## ⚙️ Kurulum

### Gereksinimler

- Python ≥ 3.11
- [Poetry](https://python-poetry.org/) (önerilen) **veya** pip

### Poetry ile

```bash
git clone https://github.com/gunay-ozsoy/FinAnalytics.git
cd FinAnalytics
poetry install
```

### pip ile

```bash
pip install streamlit pandas numpy plotly yfinance lightgbm xgboost \
            joblib scikit-learn alpaca-py python-dotenv requests \
            reportlab google-genai sentence-transformers
```

---

## 🔑 API Anahtarları

Proje kök dizininde bir `.env` dosyası oluşturun:

```env
GEMINI_API_KEY=your_google_gemini_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
MARKETAUX_API_TOKEN=your_marketaux_token
```

| Servis | Amaç | Ücretsiz Katman |
|--------|------|-----------------|
| [Google Gemini](https://aistudio.google.com/) | FinAI sekmesi — RAG yanıt üretimi | ✅ |
| [Alpaca Markets](https://alpaca.markets/) | Orta vadeli model gerçek zamanlı veri | ✅ |
| [Marketaux](https://www.marketaux.com/) | Haber bülteni | ✅ (100 istek/gün) |

---

## 🚀 Uygulamayı Çalıştırma

```bash
# Ortam değişkenlerini yükle
set -a && source .env && set +a

# Streamlit'i başlat
streamlit run app/Analiz.py
```

Tarayıcınızda `http://localhost:8502` adresine gidin.

---

## 📊 Özellikler

### 🏢 Hakkında Sekmesi
Yahoo Finance üzerinden gerçek zamanlı şirket profili, finansal metrikler (F/K, PD/DD, piyasa değeri, temettü verimi), yönetim ekibi ve kurumsal yönetim risk skorları.

### 📉 Kısa Vadeli Model (1–7 Gün)
Her hisse için **LightGBM** ve **XGBoost** modelleri kullanılarak 1, 3, 5 ve 7 günlük fiyat yön tahmini. Teknik göstergeler (RSI, MACD, Bollinger Bantları, ATR) üzerinden özellik mühendisliği yapılmıştır.

### 📊 Orta Vadeli Model (1–3 Ay)
Sektör bazında optimize edilmiş parametrelerle **orta vadeli regresyon** tahminleri. `models/mid_term/sector_optimized_params.json` dosyasındaki sektör optimizasyon sonuçlarını kullanır.

### 📈 Uzun Vadeli Model (3+ Ay)
Geniş bir hisse evreni üzerinde eğitilmiş **LightGBM** modeli. Walk-forward validasyon ile değerlendirilmiştir.

### 📰 Haber Bülteni
Marketaux API üzerinden seçili hisse ve sektörüne ait güncel haberler. Günlük disk önbelleği sayesinde API kotası verimli kullanılır.

### 🤖 FinAI — Türkçe Finans Asistanı
**Retrieval-Augmented Generation (RAG)** mimarisi:
1. Kullanıcı sorusu → semantik benzerlik arama (Turkish Finance SFT Dataset)
2. En yakın örnekler bağlam olarak Gemini'ye iletilir
3. Gemini 2.0 Flash Lite → Türkçe yanıt üretir

### 📄 Raporlar
Kısa/orta/uzun vadeli model sinyallerini içeren **Finansal Rapor** ve haber özetlerini içeren **Gündem Raporu**'nu PDF olarak indirir.

### 💼 Portföy Yönetimi
Hisse ekle/çıkar, gerçek zamanlı kar-zarar hesabı, tahmini büyüme projeksiyonu (1 hafta / 1 ay / 3 ay), portföy ağırlık grafikleri ve temettü analizi.

---

## 🧠 Veri Kaynakları

### Turkish Finance SFT Dataset

FinAI sekmesindeki RAG sistemi, semantik retrieval kaynağı olarak aşağıdaki dataset'i kullanmaktadır:

> **Turkish Finance SFT Dataset**  
> Yazar: [Alican Kiraz](https://huggingface.co/AlicanKiraz0)  
> 🔗 https://huggingface.co/datasets/AlicanKiraz0/Turkish-Finance-SFT-Dataset

```bibtex
@dataset{kiraz2025turkishfinance,
  title     = {Turkish Finance SFT Dataset},
  author    = {Kiraz, Alican},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/AlicanKiraz0/Turkish-Finance-SFT-Dataset}
}
```

| Özellik | Detay |
|---------|-------|
| Kapsam | ~10 milyon token, Türkçe finans soru-cevap çiftleri |
| Kategoriler | Kripto para, Borsa & Hisse Senetleri, Teknik Analiz, Temel Analiz, Risk Yönetimi |
| Piyasalar | BIST (Türkiye) + Global (NASDAQ, S&P 500) |
| Lisans | MIT |

> ⚠️ Bu dataset ve uygulama **yalnızca eğitim/araştırma amaçlıdır**. Yatırım tavsiyesi niteliği taşımaz.

### Diğer Veri Kaynakları

| Kaynak | Kullanım |
|--------|----------|
| [Yahoo Finance](https://finance.yahoo.com/) | Hisse fiyatları, şirket profili, finansal metrikler |
| [Alpaca Markets](https://alpaca.markets/) | Gerçek zamanlı ve tarihsel OHLCV verisi |
| [Marketaux](https://www.marketaux.com/) | Finansal haberler |

---

## 🏗️ Model Eğitimi

### Kısa Vadeli Modeller

```bash
cd models/short_term/src
python train.py
```

### Orta Vadeli Modeller

```bash
cd models/mid_term/src
python train.py
```

### Uzun Vadeli Model

```bash
python src/model_long.py
```

---

## ⚠️ Sorumluluk Reddi

Bu uygulama **tamamen eğitim ve araştırma amaçlı** geliştirilmiştir. Model tahminleri ve yapay zeka yanıtları **kesinlikle yatırım tavsiyesi** değildir. Gerçek yatırım kararlarında kullanmayınız.

---

*FinAnalytics — Finansal Analiz & Tahmin Platformu*