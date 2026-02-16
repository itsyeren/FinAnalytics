from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple


DEFAULT_MAX_ITEMS = 10
DEFAULT_MAX_SNIPPET_CHARS = 500


_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(s: str) -> str:
    s = s or ""
    s = _CTRL_RE.sub(" ", s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


def _truncate(s: str, max_chars: int) -> str:
    s = _clean_text(s)
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def _fmt_dt(published_at: str) -> str:
    p = _clean_text(published_at)
    if not p:
        return ""
    try:
        # örn: "2026-02-13T01:23:45Z" -> "+00:00"
        iso = p.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return p


def _pick_content(it: Dict[str, Any], max_snippet_chars: int) -> str:
    desc = _clean_text(str(it.get("description") or ""))
    snip = _clean_text(str(it.get("snippet") or ""))
    content = desc if desc else snip
    return _truncate(content, max_snippet_chars)


def _format_item(
    it: Dict[str, Any],
    *,
    idx: int,
    label: str,
    include_url: bool,
    max_snippet_chars: int,
) -> str:
    title = _clean_text(str(it.get("title") or ""))
    published_at = _fmt_dt(str(it.get("published_at") or ""))
    source = _clean_text(str(it.get("source") or ""))
    url = _clean_text(str(it.get("url") or ""))

    meta_parts = [p for p in [published_at, source] if p]
    meta = ", ".join(meta_parts)

    # 1. Şirket Haberi: Başlık, Tarih/Kaynak
    header = f"{idx}. {label}: {title}".strip()
    if meta:
        header = f"{header} ({meta})"

    content = _pick_content(it, max_snippet_chars=max_snippet_chars)

    parts = [header]
    if content:
        parts.append(content)
    if include_url and url:
        parts.append(url)

    return "\n".join(parts).strip()


def build_llm_context(
    *,
    symbol: str,
    industry: str,
    ticker_news: List[Dict[str, Any]],
    industry_news: List[Dict[str, Any]],
    include_url: bool = True,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
) -> Tuple[str, str]:
    """
    Çıktı:
      - ticker_context: LLM'e verilecek şirket haber bağlamı (son max_items)
      - industry_context: LLM'e verilecek sektör haber bağlamı (son max_items)
    """

    sym = _clean_text(symbol)
    ind = _clean_text(industry)

    # aynı url/title tekrarlarını azalt
    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for it in items or []:
            key = (_clean_text(str(it.get("url") or "")) or _clean_text(str(it.get("title") or ""))).lower()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
            if len(out) >= max_items:
                break
        return out

    t_items = _dedupe(ticker_news)[:max_items]
    i_items = _dedupe(industry_news)[:max_items]

    ticker_lines = []
    if sym:
        ticker_lines.append(f"SYMBOL: {sym}")
    ticker_lines.append("Şirket Haberleri:")
    if not t_items:
        ticker_lines.append("Yok.")
    else:
        for idx, it in enumerate(t_items, start=1):
            ticker_lines.append(
                _format_item(
                    it,
                    idx=idx,
                    label="Şirket Haberi",
                    include_url=include_url,
                    max_snippet_chars=max_snippet_chars,
                )
            )
            ticker_lines.append("")  # boş satır

    industry_lines = []
    if ind:
        industry_lines.append(f"INDUSTRY: {ind}")
    industry_lines.append("Sektör Haberleri:")
    if not i_items:
        industry_lines.append("Yok.")
    else:
        for idx, it in enumerate(i_items, start=1):
            industry_lines.append(
                _format_item(
                    it,
                    idx=idx,
                    label="Sektör Haberi",
                    include_url=include_url,
                    max_snippet_chars=max_snippet_chars,
                )
            )
            industry_lines.append("")

    ticker_context = "\n".join([x for x in ticker_lines if x is not None]).strip()
    industry_context = "\n".join([x for x in industry_lines if x is not None]).strip()

    return ticker_context, industry_context
