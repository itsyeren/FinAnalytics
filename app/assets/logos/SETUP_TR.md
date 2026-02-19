# Şirket Logoları - Hızlı Setup

## En Kolay Yol: Hazır İndir ve Yapıştır

Aşağıdaki linkte **37 şirketin logolarının ZIP dosyası** var:
- SVG veya PNG formatında
- Kimseye dert etmeyecek boyutta (~5-10 MB total)

**Alternatif 1: Otomatik Script (Tek Komut)**

```bash
cd /workspaces/itsyeren/FinAnalytics
python3 scripts/download_logos.py
```

Script:
- Wikimedia Commons'ta arar
- Otomatik indir
- Dosyaları düzenle

**Alternatif 2: Manuel (5 dakika)**

Google'da ara → Indir → Yapıştır:

```bash
# Örnek (herbir şirket için):
1. Google: "Apple Inc logo PNG"
2. İndir (200x200 px+)
3. Dosyayı şu isimle kaydet: AAPL.png
4. Buraya koy: app/assets/logos/AAPL.png
```

**Alternatif 3: Batch İndir (Python, 10 komut)**

```python
# Bu script'i çalıştır (internet gerektir):
python3 << 'EOF'
import os
import requests
from pathlib import Path

logos_dir = Path("app/assets/logos")

# Her şirketin logosunu Wikipedia'dan çek
companies = {
    "AAPL": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Apple_logo_grey.svg",
    "MSFT": "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
    # ... (başka logoların direct URL'leri)
}

for ticker, url in companies.items():
    try:
        response = requests.get(url)
        with open(logos_dir / f"{ticker}.png", 'wb') as f:
            f.write(response.content)
        print(f"✓ {ticker}")
    except Exception as e:
        print(f"✗ {ticker}: {e}")
EOF
```

## Dosya Yapısı

```
app/assets/logos/
├── AAPL.png     (Apple)
├── MSFT.png     (Microsoft)
├── NVDA.png     (NVIDIA)
├── CAG.png      (Conagra)
├── HSY.png      (Hershey)
...
└── BF_B.png     (Brown-Forman - underscore dikkat!)
```

## Logo Özellikleri

- **Format**: PNG tercih, SVG da olur
- **Boyut**: 200x200 px veya daha büyük
- **Dosya adı**: `{TICKER}.png` (BF.B için `BF_B.png`)
- **Arkaplan**: Transparent veya beyaz (şeffaflık tercih)

## Linkler: Direkt Şirket Logoları

Aşağıdaki linklerden direkt indir (sağ tıkla → İndir):

| Ticker | Şirket | Logo Linki |
|--------|--------|-----------|
| AAPL | Apple | https://www.apple.com/brand/assets/downloads/art-direction/templates/logo.png |
| MSFT | Microsoft | https://c.s-microsoft.com/favicon.ico→MS_LOGO |
| NVDA | NVIDIA | https://s.nvidia.com/favicon.ico |
| CAG | Conagra | https://www.conagra.com favicon |
| ... | ... | ... |

**Daha hızlı**: [DuckDuckGo Images](https://duckduckgo.com/?q=company+logo&iax=images&ia=images) veya [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Company_logos) kullan.

---

⚠️ **Önemli**: Logolar olmadığında App boş placeholder gösterecek. Ama fonksiyon etmei bozulmaz.
