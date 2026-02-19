#!/usr/bin/env python3
"""
Şirket logolarını indirme script'i.
Wikimedia Commons'tan gerçek şirket logolarını arar ve indir.
"""

import os
import requests
from pathlib import Path
import json

# TICKERS ve şirket isimleri (Wikimedia Commons araması için)
COMPANY_INFO = {
    "AAPL": ("Apple Inc.", "Apple_Inc._logo.svg"),
    "MSFT": ("Microsoft", "Microsoft_logo.svg"),
    "NVDA": ("Nvidia", "Nvidia_logo.svg"),
    "CAG": ("Conagra Brands", "Conagra-Monogram.svg"),
    "HSY": ("Hershey Company", "Hershey_Company_logo.svg"),
    "CCEP": ("Coca-Cola Europacific Partners", None),
    "KR": ("Kroger", "Kroger_logo.svg"),
    "SYY": ("Sysco", "Sysco_wordmark.svg"),
    "CPB": ("Campbell Soup Company", "Campbell_Soup_Company_logo.svg"),
    "KDP": ("Keurig Dr Pepper", "KDP_-_Keurig_Dr_Pepper.svg"),
    "PEP": ("PepsiCo", "PepsiCo_2023_logo.svg"),
    "TSN": ("Tyson Foods", "Tyson_Foods_logo.svg"),
    "SJM": ("J. M. Smucker Company", "Smuckers_logo.svg"),
    "KHC": ("Kraft Heinz", "Kraft_Heinz_logo.svg"),
    "PM": ("Philip Morris International", "Philip_Morris_International_logo.svg"),
    "MO": ("Altria", "Altria_Group_logo.svg"),
    "HRL": ("Hormel Foods", "Hormel_Foods_logo.svg"),
    "EL": ("Estée Lauder", "Estee_Lauder_logo.svg"),
    "CL": ("Colgate-Palmolive", "Colgate-Palmolive_logo.svg"),
    "K": ("Kellogg Company", "Kellogg_Company_logo.svg"),
    "GIS": ("General Mills", "General_Mills_logo.svg"),
    "KMB": ("Kimberly-Clark", "Kimberly-Clark_Logo.svg"),
    "CLX": ("Clorox Company", "Clorox_logo.svg"),
    "MKC": ("McCormick & Company", "McCormick_Company_logo.svg"),
    "KO": ("Coca-Cola", "Coca-Cola_logo.svg"),
    "WMT": ("Walmart", "Walmart_logo.svg"),
    "COST": ("Costco", "Costco_logo.svg"),
    "DG": ("Dollar General", "Dollar_General_logo.svg"),
    "DLTR": ("Dollar Tree", "Dollar_Tree_logo.svg"),
    "WBA": ("Walgreens Boots Alliance", "Walgreens_Boots_Alliance_logo.svg"),
    "MNST": ("Monster Beverage", "Monster_Beverage_logo.svg"),
    "STZ": ("Constellation Brands", "Constellation_Brands_logo.svg"),
    "MDLZ": ("Mondelez International", "Mondelez_International_logo.svg"),
    "TAP": ("Molson Coors", "Molson_Coors_logo.svg"),
    "LW": ("Lamb Weston", "Lamb_Weston_logo.svg"),
    "CHD": ("Church & Dwight", "Church_and_Dwight_logo.svg"),
    "BF.B": ("Brown-Forman", "Brown-Forman_logo.svg"),
}

LOGOS_DIR = Path(__file__).parent.parent / "app" / "assets" / "logos"
LOGOS_DIR.mkdir(parents=True, exist_ok=True)

def search_wikimedia_logo(ticker: str, company_name: str) -> str:
    """Wikimedia Commons'ta logo arayıp URL'sini döndür."""
    # Wikimedia Commons API
    api_url = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": f"{company_name} logo",
        "srnamespace": "6",  # File namespace
    }

    try:
        response = requests.get(api_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("query", {}).get("search"):
                first_result = data["query"]["search"][0]
                filename = first_result["title"].replace("File:", "")
                return filename
    except Exception as e:
        print(f"  API hatası: {e}")

    return None

def download_logo_from_wikimedia(filename: str, logo_path: Path) -> bool:
    """Wikimedia Commons'tan dosyayı indir."""
    if not filename:
        return False

    # URL oluştur
    url = f"https://commons.wikimedia.org/wiki/File:{filename}"

    try:
        # API ile dosya URL'sini al
        api_url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": f"File:{filename}",
            "prop": "imageinfo",
            "iiprop": "url",
        }

        response = requests.get(api_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                image_info = page.get("imageinfo", [])
                if image_info:
                    image_url = image_info[0].get("url")
                    if image_url:
                        # Dosyayı indir
                        img_response = requests.get(image_url, timeout=10)
                        if img_response.status_code == 200:
                            with open(logo_path, 'wb') as f:
                                f.write(img_response.content)
                            return True
    except Exception as e:
        print(f"  İndirme hatası: {e}")

    return False

def filename_from_ticker(ticker: str) -> str:
    """Ticker'dan safe filename yap."""
    import re
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    return f"{safe}.png"

def main():
    print(f"\nLogolar {LOGOS_DIR} klasörüne indirilecek\n")
    print("=" * 60)

    success = 0
    skipped = 0
    failed = 0

    for ticker, (company_name, suggested_filename) in COMPANY_INFO.items():
        filename_out = filename_from_ticker(ticker)
        filepath = LOGOS_DIR / filename_out

        if filepath.exists():
            print(f"✓ {ticker:6} → {filename_out:20} (zaten var)")
            skipped += 1
            continue

        print(f"⟳ {ticker:6} → {company_name[:30]:30}", end=" ")

        # Önce önerilen filename'ı dene
        if suggested_filename:
            if download_logo_from_wikimedia(suggested_filename, filepath):
                print("✓ (indirildi)")
                success += 1
                continue

        # Sonra otomatik arama yap
        found_filename = search_wikimedia_logo(ticker, company_name)
        if found_filename and download_logo_from_wikimedia(found_filename, filepath):
            print(f"✓ (bulundu ve indirildi)")
            success += 1
        else:
            print("✗ (bulunamadı)")
            failed += 1

    print("=" * 60)
    print(f"\nSonuç: {success} başarılı, {skipped} zaten var, {failed} başarısız")
    print(f"Klasör: {LOGOS_DIR}")
    print("\nİpucu: Başarısız olanlar için manuel indir:")
    print("  app/assets/logos/README.md içindeki linkleri takip et")
    main()
