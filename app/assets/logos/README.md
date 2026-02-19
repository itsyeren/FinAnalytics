# Şirket Logoları

Bu klasöre 37 şirketin logolarını PNG formatında koymanız gerekiyor.

## Otomatik İndirme

Wikimedia Commons ve diğer kaynakların favicon'unu indirmek için bu script'i çalıştırın:

```bash
cd /workspaces/itsyeren/FinAnalytics
python3 scripts/download_logos.py
```

## Manuel İndirme

Aşağıdaki listedeki her şirketin logosunu PNG formatında indirin ve dosyayı `TICKER.png` olarak kaydedin:

| Ticker | Şirket | Logo Kaynağı |
|--------|--------|--------------|
| AAPL | Apple | https://www.apple.com |
| MSFT | Microsoft | https://www.microsoft.com |
| NVDA | NVIDIA | https://www.nvidia.com |
| CAG | Conagra Brands | https://www.conagra.com |
| HSY | Hershey | https://www.thehersheycompany.com |
| CCEP | Coca-Cola Europacific Partners | https://www.coca-colacompany.com |
| KR | Kroger | https://www.kroger.com |
| SYY | Sysco | https://www.sysco.com |
| CPB | Campbell Soup Company | https://www.campbellsoupcompany.com |
| KDP | Keurig Dr Pepper | https://www.keurigdrpepper.com |
| PEP | PepsiCo | https://www.pepsico.com |
| TSN | Tyson Foods | https://www.tysonfoods.com |
| SJM | JM Smucker | https://www.smuckercompany.com |
| KHC | Kraft Heinz | https://www.kraftheinzcompany.com |
| PM | Philip Morris International | https://www.philipmorrisinternational.com |
| MO | Altria | https://www.altria.com |
| HRL | Hormel Foods | https://www.hormelfoods.com |
| EL | Estée Lauder | https://www.esteelauder.com |
| CL | Colgate-Palmolive | https://www.colgatepalmolive.com |
| K | Kellogg | https://www.kelloggcompany.com |
| GIS | General Mills | https://www.generalmills.com |
| KMB | Kimberly-Clark | https://www.kimberly-clark.com |
| CLX | Clorox | https://www.thecloroxcompany.com |
| MKC | McCormick & Company | https://www.mccormick.com |
| KO | Coca-Cola | https://www.coca-cola.com |
| WMT | Walmart | https://www.walmart.com |
| COST | Costco | https://www.costco.com |
| DG | Dollar General | https://www.dollargeneral.com |
| DLTR | Dollar Tree | https://www.dollartree.com |
| WBA | Walgreens Boots Alliance | https://www.walgreensbootsalliance.com |
| MNST | Monster Beverage | https://www.monsterbevcorp.com |
| STZ | Constellation Brands | https://www.constellationbrandsusa.com |
| MDLZ | Mondelez International | https://www.mondelezinternational.com |
| TAP | Molson Coors | https://www.molsoncoors.com |
| LW | Lamb Weston | https://www.lambweston.com |
| CHD | Church & Dwight | https://www.churchdwight.com |
| BF.B | Brown-Forman | https://www.brownforman.com |

### Kolay Yol: Google "Company Logo PNG" Ara

Her şirket için:
1. Google'da `{Company Name} logo PNG` ara
2. Logoyu indir
3. `{TICKER}.png` olarak bu klasöre kaydet

## Dosya Adlandırması

Dosya adları şu formatta olmalı:
- `AAPL.png`
- `MSFT.png`
- `BF_B.png` (BF.B ticker'ı için underscore kullanır)

## Logo Özellikleri (İdeal)

- **Boyut**: 200x200 px veya daha büyük
- **Format**: PNG (transparent background tercih edilir)
- **Dosya boyutu**: 10-100 KB
- **Stil**: Tarafsız, şirketin resmi logosu

App.py otomatik olarak bu klasördeki PNG dosyalarını bulup gösterecektir.
