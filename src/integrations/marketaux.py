import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE = "https://api.marketaux.com/v1"
CACHE_PATH = Path(os.getenv("MARKETAUX_ENTITY_CACHE", ".cache/marketaux_entity_cache.json"))


def _token() -> str:
    t = os.getenv("MARKETAUX_API_TOKEN", "").strip()
    if not t:
        raise RuntimeError("MARKETAUX_API_TOKEN yok. .env veya environment variable set et.")
    return t


def _ensure_cache_dir() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_cache() -> Dict[str, Any]:
    _ensure_cache_dir()
    if not CACHE_PATH.exists():
        return {"entities": {}}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"entities": {}}


def _save_cache(cache: Dict[str, Any]) -> None:
    _ensure_cache_dir()
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = {"api_token": _token(), **params}
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Marketaux istek zaman aşımı: {path}")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Marketaux bağlantı hatası: {exc}")
    if r.status_code == 401:
        raise RuntimeError("Marketaux 401: API token geçersiz veya süresi dolmuş.")
    if r.status_code == 429:
        raise RuntimeError("Marketaux 429: API kota aşıldı. Bir süre bekleyin.")
    if r.status_code != 200:
        raise RuntimeError(f"Marketaux HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def _entity_search(
    *,
    search: Optional[str] = None,
    symbols: Optional[str] = None,
    countries: Optional[str] = None,
    types: str = "equity",
    page: int = 1,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"page": page, "types": types}
    if search:
        params["search"] = search
    if symbols:
        params["symbols"] = symbols
    if countries:
        params["countries"] = countries
    return _get("/entity/search", params).get("data", [])


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _variants(ticker_like: str) -> List[str]:
    raw = (ticker_like or "").strip().upper()
    vs = [raw]

    if raw.endswith(".US"):
        vs.append(raw[:-3])

    if "." in raw:
        vs.append(raw.replace(".", "-"))
        vs.append(raw.replace(".", ""))
    if "-" in raw:
        vs.append(raw.replace("-", "."))
        vs.append(raw.replace("-", ""))

    return _dedupe_keep_order(vs)


def _pick_best(cands: List[Dict[str, Any]], prefer_country: str = "us") -> Optional[Dict[str, Any]]:
    if not cands:
        return None

    pc = (prefer_country or "").lower()
    for it in cands:
        if (it.get("country") or "").lower() == pc:
            return it

    return cands[0]


def resolve_entity(
    ticker_like: str,
    *,
    company_name: Optional[str] = None,
    prefer_country: str = "us",
) -> Dict[str, Any]:
    """Entity'yi önce cache'ten, yoksa Marketaux'tan çözer.
    Fallback sırası: symbol/country → symbol (global) → name/country → name (global).
    Her adımda bulunca hemen döner ve cache'e yazar — gereksiz API çağrısı önler.
    """
    cache = _load_cache()
    entities = cache.get("entities", {})

    key = (ticker_like or "").strip().upper()
    if key in entities:
        return entities[key]

    variants = _variants(key)

    def _try_search(search_type: str, q: str, country: Optional[str]) -> Optional[Dict[str, Any]]:
        """Tek bir entity arama dener, bulursa döndürür."""
        try:
            params: Dict[str, Any] = {}
            if search_type == "symbol":
                params["symbols"] = q
            else:
                params["search"] = q
            if country:
                params["countries"] = country
            cands = _entity_search(**params)
            return _pick_best(cands, prefer_country=prefer_country)
        except Exception:
            return None

    def _save_and_return(best: Dict[str, Any]) -> Dict[str, Any]:
        ent = {
            "symbol": best.get("symbol"),
            "name": best.get("name"),
            "industry": best.get("industry"),
            "country": best.get("country"),
            "type": best.get("type"),
        }
        entities[key] = ent
        cache["entities"] = entities
        _save_cache(cache)
        return ent

    # Fallback zinciri: dört adım, ilk başarıda erken dönür
    fallbacks = [
        # (search_type, country_filter)
        ("symbol", prefer_country),
        ("symbol", None),
        ("name",   prefer_country),
        ("name",   None),
    ]

    for search_type, country in fallbacks:
        queries = variants if search_type == "symbol" else (
            _dedupe_keep_order([company_name.strip(), company_name.strip().replace("-", " ")])
            if company_name else variants
        )
        for q in queries:
            best = _try_search(search_type, q, country)
            if best:
                return _save_and_return(best)

    raise ValueError(f"Entity bulunamadı: {ticker_like} (company_name={company_name})")



def _news_page(params: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "filter_entities": "true",
        "must_have_entities": "true",
        "group_similar": "false",
        "language": "en",
        "sort": "published_at",
    }
    base.update(params)
    return _get("/news/all", base)


def get_last_n_news(params_key: str, params_val: str, n: int = 10, per_req: int = 10, max_pages: int = 3) -> List[Dict[str, Any]]:
    """Son n haberi çeker. max_pages ile kota aşımı önlenir."""
    collected: List[Dict[str, Any]] = []
    seen = set()
    page = 1

    while len(collected) < n and page <= max_pages:
        resp = _news_page({params_key: params_val, "limit": per_req, "page": page})
        items = resp.get("data", [])
        if not items:
            break

        for it in items:
            uid = it.get("uuid")
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            collected.append(it)
            if len(collected) >= n:
                break

        # Sayfa dönen eleman sayısı istenen limit'ten azsa daha fazla sayfa yok
        if len(items) < per_req:
            break

        page += 1

    return collected[:n]


def get_ticker_and_industry_news(
    ticker_like: str,
    *,
    company_name: Optional[str] = None,
    country: str = "us",
    n: int = 10,
    per_req: int = 3,
) -> Dict[str, Any]:
    ent = resolve_entity(ticker_like, company_name=company_name, prefer_country=country)

    symbol = (ent.get("symbol") or "").strip()
    industry = (ent.get("industry") or "").strip()

    if not symbol:
        raise ValueError(f"Symbol boş döndü: {ticker_like}")

    if not industry:
        cands = _entity_search(symbols=symbol)
        best = _pick_best(cands, prefer_country=country)
        industry = (best.get("industry") if best else "") or ""
        industry = industry.strip()

    ticker_news = get_last_n_news("symbols", symbol, n=n, per_req=per_req)
    industry_news = get_last_n_news("industries", industry, n=n, per_req=per_req) if industry else []

    return {"symbol": symbol, "industry": industry, "ticker_news": ticker_news, "industry_news": industry_news}
