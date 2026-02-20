"""
src/reports/pdf_builder.py
==========================
Finansal Rapor ve Gündem Raporu PDF üretici.

Bağımlılık: fpdf2  (poetry add fpdf2)
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from fpdf import FPDF


# ── Renk paleti ────────────────────────────────────────────────────
_DARK   = (10,  16,  28)   # arka plan koyu başlık bandı
_ACCENT = (99,  179, 237)  # mavi vurgu
_TEXT   = (30,  30,  40)   # gövde metin
_LIGHT  = (245, 248, 255)  # açık satır arka planı
_WHITE  = (255, 255, 255)


# ═══════════════════════════════════════════════════════════════════
class _PDF(FPDF):
    """Tüm raporlar için ortak baz sınıf."""

    def __init__(self, title: str, ticker: str, label: str) -> None:
        super().__init__()
        self._report_title = title
        self._ticker = ticker
        self._label  = label
        self.set_auto_page_break(auto=True, margin=18)
        self.add_page()
        self._draw_cover_band()

    # ── Kapak bandı ──────────────────────────────────────────────
    def _draw_cover_band(self) -> None:
        self.set_fill_color(*_DARK)
        self.rect(0, 0, 210, 44, style="F")

        # Sol üst köşe: uygulama adı
        self.set_xy(12, 6)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_ACCENT)
        self.cell(0, 6, "FinAnalytics", ln=True)

        # Rapor başlığı
        self.set_xy(12, 14)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*_WHITE)
        self.cell(0, 8, self._report_title, ln=True)

        # Alt bilgi: ticker + tarih
        self.set_xy(12, 28)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_ACCENT)
        ts = datetime.now().strftime("%d %b %Y  %H:%M")
        self.cell(0, 5, f"{self._label}  ({self._ticker})    {ts}", ln=True)

        self.set_y(52)
        self.set_text_color(*_TEXT)

    # ── Bölüm başlığı ────────────────────────────────────────────
    def section_header(self, text: str) -> None:
        text = _safe(text)
        self.ln(4)
        self.set_fill_color(*_ACCENT)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, f"  {text}", ln=True, fill=True)
        self.ln(3)
        self.set_text_color(*_TEXT)

    # ── Anahtar-değer satırı ─────────────────────────────────────
    def kv_row(self, key: str, value: str, shade: bool = False) -> None:
        key, value = _safe(key), _safe(value)
        if shade:
            self.set_fill_color(*_LIGHT)
        self.set_font("Helvetica", "B", 9)
        self.set_x(12)
        self.cell(52, 6, key, fill=shade)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 6, value, fill=shade)
        self.set_x(12)

    # ── Düz metin paragrafı ──────────────────────────────────────
    def body_text(self, text: str, size: int = 9) -> None:
        self.set_font("Helvetica", "", size)
        self.set_x(12)
        self.multi_cell(186, 5, _safe(text))
        self.ln(2)

    # ── Haber kartı ──────────────────────────────────────────────
    def news_card(self, idx: int, title: str, source: str,
                  published_at: str, summary: str) -> None:
        # Tüm gelen metinleri güvene al
        title, source, published_at, summary = (
            _safe(title), _safe(source), _safe(published_at), _safe(summary)
        )
        # Numara bandı
        self.set_fill_color(*_DARK)
        self.set_text_color(*_ACCENT)
        self.set_font("Helvetica", "B", 8)
        self.set_x(12)
        self.cell(8, 5, f"{idx:02d}", fill=True, align="C")

        # Başlık
        self.set_fill_color(*_LIGHT)
        self.set_text_color(*_TEXT)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 5, _truncate(title, 95), ln=True, fill=True)

        # Meta: kaynak · tarih
        meta = "  " + "  .  ".join(p for p in [source, published_at] if p)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(110, 120, 140)
        self.set_x(12)
        self.cell(0, 4, meta, ln=True)

        # Özet paragrafı
        self.set_text_color(*_TEXT)
        self.set_font("Helvetica", "", 8.5)
        self.set_x(12)
        self.multi_cell(186, 4.5, summary if summary else "-")
        self.ln(3)

    # ── Footer ───────────────────────────────────────────────────
    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(160, 170, 190)
        self.cell(0, 5,
                  "Bu rapor egitim/arastirma amacidir. Yatirim tavsiyesi degildir.",
                  align="C")
        self.set_y(-10)
        self.set_font("Helvetica", "", 7)
        self.cell(0, 4, f"Sayfa {self.page_no()}", align="C")


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════

def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars - 1] + "..."


def _safe(text: Any) -> str:
    """None'ı boş string'e çevirir; latin-1 dışı karakterleri temizler."""
    s = str(text or "").strip()
    # fpdf2 Latin-1 çıktı kipi için Türkçe ı/İ → i/I
    replacements = {
        "ı": "i", "İ": "I",
        "ğ": "g", "Ğ": "G",
        "ş": "s", "Ş": "S",
        "ü": "u", "Ü": "U",
        "ö": "o", "Ö": "O",
        "ç": "c", "Ç": "C",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...",
    }
    for src, tgt in replacements.items():
        s = s.replace(src, tgt)
    # Hâlâ Latin-1 dışı kalan karakterleri "?" ile değiştir
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _fmt_dt(raw: str) -> str:
    if not raw:
        return ""
    try:
        import pandas as pd
        dt = pd.to_datetime(raw)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.tz_convert(None)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw


# ═══════════════════════════════════════════════════════════════════
# FİNANSAL RAPOR PDF
# ═══════════════════════════════════════════════════════════════════

def build_financial_pdf(
    ticker: str,
    label: str,
    short_data: dict | None = None,
    mid_data: dict | None = None,
    long_data: dict | None = None,
) -> bytes:
    """
    Kısa / Orta / Uzun vadeli model çıktılarını PDF'e yazar.

    Parametreler
    ------------
    short_data : {'signal_rows': [{'horizon', 'signal', 'confidence', 'prob_up'}, ...],
                  'anchor': str, 'price': float}
    mid_data   : {'son_fiyat', 'tahmin_1ay', 'tahmin_3ay',
                  'getiri_1ay_pct', 'getiri_3ay_pct',
                  'sinyal_1ay', 'sinyal_3ay', 'sektor': str}
    long_data  : {'signal': str, 'score': float, 'percentile': float,
                  'prob_up': float, 'prob_neutral': float, 'prob_down': float,
                  'momentum': float, 'slope': float, 'dist_sma200': float,
                  'price': float}
    """
    pdf = _PDF("Finansal Rapor", ticker, label)

    # ── Kısa Vadeli Model ───────────────────────────────────────
    pdf.section_header("KISA VADELI MODEL")

    if short_data:
        pdf.kv_row("Hisse", _safe(f"{label} ({ticker})"), shade=True)
        pdf.kv_row("Model Tarihi", _safe(short_data.get("anchor", "—")))
        price_str = f"${short_data.get('price', 0):.2f}" if short_data.get("price") else "—"
        pdf.kv_row("Son Fiyat", _safe(price_str), shade=True)
        pdf.ln(2)

        rows = short_data.get("signal_rows", [])
        if rows:
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_fill_color(*_DARK)
            pdf.set_text_color(*_WHITE)
            pdf.set_x(12)
            for hdr, w in [("Horizon", 30), ("Sinyal", 26), ("Güven", 30), ("P(UP)", 30)]:
                pdf.cell(w, 6, hdr, fill=True, align="C")
            pdf.ln()
            pdf.set_text_color(*_TEXT)
            for i, r in enumerate(rows):
                shade = (i % 2 == 0)
                if shade:
                    pdf.set_fill_color(*_LIGHT)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_x(12)
                pdf.cell(30, 5.5, _safe(r.get("horizon", "—")), fill=shade, align="C")
                sig   = _safe(r.get("signal", "—"))
                pdf.cell(26, 5.5, sig, fill=shade, align="C")
                pdf.cell(30, 5.5, f"%{r.get('confidence', 0)*100:.1f}", fill=shade, align="C")
                pdf.cell(30, 5.5, f"%{r.get('prob_up', 0)*100:.1f}", fill=shade, align="C")
                pdf.ln()
        pdf.ln(3)
    else:
        pdf.body_text("Kisa vadeli model verisi mevcut degil.")

    # ── Orta Vadeli Model ───────────────────────────────────────
    pdf.section_header("ORTA VADELI MODEL")

    if mid_data:
        rows_mid = [
            ("Hisse",          _safe(f"{label} ({ticker})")),
            ("Sektor",         _safe(mid_data.get("sektor", "—"))),
            ("Son Fiyat",      _safe(f"${mid_data.get('son_fiyat', 0):.2f}")),
            ("1 Ay Tahmin",    _safe(f"${mid_data.get('tahmin_1ay', 0):.2f}  ({mid_data.get('getiri_1ay_pct', 0):+.2f}%)")),
            ("1 Ay Sinyal",    _safe(mid_data.get("sinyal_1ay", "—"))),
            ("3 Ay Tahmin",    _safe(f"${mid_data.get('tahmin_3ay', 0):.2f}  ({mid_data.get('getiri_3ay_pct', 0):+.2f}%)")),
            ("3 Ay Sinyal",    _safe(mid_data.get("sinyal_3ay", "—"))),
        ]
        for i, (k, v) in enumerate(rows_mid):
            pdf.kv_row(k, v, shade=(i % 2 == 0))
    else:
        pdf.body_text("Orta vadeli model verisi mevcut degil.")
    pdf.ln(3)

    # ── Uzun Vadeli Model ───────────────────────────────────────
    pdf.section_header("UZUN VADELI MODEL")

    if long_data:
        rows_long = [
            ("Hisse",         _safe(f"{label} ({ticker})")),
            ("Son Fiyat",     _safe(f"${long_data.get('price', 0):.2f}")),
            ("Model Sinyali", _safe(long_data.get("signal", "—"))),
            ("Model Skoru",   _safe(f"{long_data.get('score', 0):.1f} / 100")),
            ("Persentil",     _safe(f"P{long_data.get('percentile', 0):.0f}")),
            ("P(UP)",         _safe(f"%{long_data.get('prob_up', 0)*100:.1f}")),
            ("P(NEUTRAL)",    _safe(f"%{long_data.get('prob_neutral', 0)*100:.1f}")),
            ("P(DOWN)",       _safe(f"%{long_data.get('prob_down', 0)*100:.1f}")),
            ("Momentum 6M",   _safe(f"{long_data.get('momentum', 0):+.3f}")),
            ("Trend Egimi",   _safe(f"{long_data.get('slope', 0):+.4f}")),
            ("SMA200 Uzakl.", _safe(f"{long_data.get('dist_sma200', 0):+.3f}")),
        ]
        for i, (k, v) in enumerate(rows_long):
            pdf.kv_row(k, v, shade=(i % 2 == 0))
    else:
        pdf.body_text("Uzun vadeli model verisi mevcut degil.")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# GÜNDEM RAPORU PDF
# ═══════════════════════════════════════════════════════════════════

def build_agenda_pdf(
    ticker: str,
    label: str,
    ticker_news: list[dict],
    industry_news: list[dict],
    symbol: str = "",
    industry: str = "",
) -> bytes:
    """
    Haber Bültenindeki ticker + sektör haberlerini 1'er paragraf özetle PDF'e yazar.
    Her haberin description/snippet'i özet olarak kullanılır.
    """
    pdf = _PDF("Gundem Raporu", ticker, label)

    meta_parts = [p for p in [_safe(symbol), _safe(industry)] if p]
    if meta_parts:
        pdf.body_text("  ".join(meta_parts), size=9)
    pdf.ln(2)

    # ── Şirket Haberleri ────────────────────────────────────────
    pdf.section_header(f"SIRKET HABERLERI  ({len(ticker_news)} haber)")

    if ticker_news:
        for i, it in enumerate(ticker_news, start=1):
            title   = _safe(it.get("title") or "")
            source  = _safe(it.get("source") or "")
            pub_at  = _safe(_fmt_dt((it.get("published_at") or "").strip()))
            desc    = _safe(it.get("description") or it.get("snippet") or "")
            # 1 paragraf: ilk 500 karakter
            summary = desc[:500] + ("..." if len(desc) > 500 else "")
            pdf.news_card(i, title, source, pub_at, summary)
    else:
        pdf.body_text("Bu hisse icin haber bulunamadi.")

    pdf.ln(4)

    # ── Sektör Haberleri ────────────────────────────────────────
    pdf.section_header(f"SEKTOR HABERLERI  ({len(industry_news)} haber)")

    if industry_news:
        for i, it in enumerate(industry_news, start=1):
            title   = _safe(it.get("title") or "")
            source  = _safe(it.get("source") or "")
            pub_at  = _safe(_fmt_dt((it.get("published_at") or "").strip()))
            desc    = _safe(it.get("description") or it.get("snippet") or "")
            summary = desc[:500] + ("..." if len(desc) > 500 else "")
            pdf.news_card(i, title, source, pub_at, summary)
    else:
        pdf.body_text("Bu sektor icin haber bulunamadi.")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
