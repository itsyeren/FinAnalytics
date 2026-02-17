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
    r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Marketaux HTTP {r.status_code}: {r.text}")
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
    cache = _load_cache()
    entities = cache.get("entities", {})

    key = (ticker_like or "").strip().upper()
    if key in entities:
        return entities[key]

    for q in _variants(key):
        cands = _entity_search(symbols=q, countries=prefer_country)
        best = _pick_best(cands, prefer_country=prefer_country)
        if best:
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

    for q in _variants(key):
        cands = _entity_search(search=q, countries=prefer_country)
        best = _pick_best(cands, prefer_country=prefer_country)
        if best:
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

    for q in _variants(key):
        cands = _entity_search(symbols=q)
        best = _pick_best(cands, prefer_country=prefer_country)
        if best:
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

    for q in _variants(key):
        cands = _entity_search(search=q)
        best = _pick_best(cands, prefer_country=prefer_country)
        if best:
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

    if company_name:
        name_qs = _dedupe_keep_order([company_name.strip(), company_name.strip().replace("-", " ")])
        for q in name_qs:
            cands = _entity_search(search=q, countries=prefer_country)
            best = _pick_best(cands, prefer_country=prefer_country)
            if best:
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

        for q in name_qs:
            cands = _entity_search(search=q)
            best = _pick_best(cands, prefer_country=prefer_country)
            if best:
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


def get_last_n_news(params_key: str, params_val: str, n: int = 10, per_req: int = 3) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    seen = set()
    page = 1

    while len(collected) < n:
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

        meta = resp.get("meta", {})
        returned = meta.get("returned")
        limit = meta.get("limit")
        if returned is not None and limit is not None and returned < limit:
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
