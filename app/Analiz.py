
import hashlib
import json
import os
import re
import sys

from pathlib import Path

# Setup path before imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.integrations.gemini import generate_text
from src.integrations.marketaux import get_ticker_and_industry_news
from src.rag.turkish_finance_sft_rag import retrieve_examples
from src.reports.news_prompt import build_llm_context
from src.reports.pdf_builder import build_financial_pdf, build_agenda_pdf
from src.short import render_short_dashboard
from src.mid import render_mid_dashboard
from src.long import render_long_dashboard


TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Conagra Brands": "CAG",
    "Hershey": "HSY",
    "Coca-Cola Europacific Partners": "CCEP",
    "Kroger": "KR",
    "Sysco": "SYY",
    "Campbell Soup Company": "CPB",
    "Keurig Dr Pepper": "KDP",
    "PepsiCo": "PEP",
    "Tyson Foods": "TSN",
    "JM Smucker": "SJM",
    "Kraft Heinz": "KHC",
    "Philip Morris International": "PM",
    "Altria": "MO",
    "Hormel Foods": "HRL",
    "Estée Lauder": "EL",
    "Colgate-Palmolive": "CL",
    "Kellogg": "K",
    "General Mills": "GIS",
    "Kimberly-Clark": "KMB",
    "Clorox": "CLX",
    "McCormick & Company": "MKC",
    "Coca-Cola": "KO",
    "Walmart": "WMT",
    "Costco": "COST",
    "Dollar General": "DG",
    "Dollar Tree": "DLTR",
    "Walgreens Boots Alliance": "WBA",
    "Monster Beverage": "MNST",
    "Constellation Brands": "STZ",
    "Mondelez International": "MDLZ",
    "Molson Coors": "TAP",
    "Lamb Weston": "LW",
    "Church & Dwight": "CHD",
    "Brown-Forman": "BF.B",
}

LOGO_DIR = Path(__file__).resolve().parent / "assets" / "logos"

# RAG sabitleri
RAG_PREVIEW_CHARS = 1800
RAG_MAX_OUTPUT_TOKENS = 700
RAG_DEBUG_ENV_FLAG = "RAG_DEBUG"  # RAG_DEBUG=1 ise hata ayıklama göstergesi görünür

# Yanıt uzunluğu seçenekleri: etiket → (max_output_tokens, paragraf talimatı)
RAG_LENGTH_OPTIONS: dict[str, tuple[int, str]] = {
    "📝 Kısa":  (300,  "Yanıtın yalnızca 1 paragraf olsun; özlü ve doğrudan yaz."),
    "📄 Orta":  (600,  "Yanıtın 2 paragraftan oluşsun; açıklayıcı ama gereksiz uzatma."),
    "📃 Uzun":  (1024, "Yanıtın 3 paragraftan oluşsun; konuyu detaylı ve yapılandırılmış anlat."),
}


def _stable_id(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12]


def ticker_to_logo_filename(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    return f"{safe}.png"




def render_logo_or_placeholder(ticker: str) -> None:
    logo_path = LOGO_DIR / ticker_to_logo_filename(ticker)

    if logo_path.exists():
        st.image(str(logo_path), width=120)
        return

    st.markdown(
        """
        <div style="
            width: 100%;
            aspect-ratio: 16/10;
            border: 2px dashed rgba(255,255,255,0.25);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: rgba(255,255,255,0.55);
            font-size: 12px;
        ">
            Logo (PNG) eklenecek
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt_dt(published_at: str) -> str:
    if not published_at:
        return ""
    try:
        dt = pd.to_datetime(published_at)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.tz_convert(None)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return published_at


def render_news_item(idx: int, kind_label: str, it: dict) -> None:
    title = (it.get("title") or "").strip()
    published_at = _fmt_dt((it.get("published_at") or "").strip())
    source = (it.get("source") or "").strip()
    desc = (it.get("description") or "").strip()
    snippet = (it.get("snippet") or "").strip()
    url = (it.get("url") or "").strip()

    content = desc or snippet
    # Açıklamayı 220 karakterde kırp
    content_short = (content[:220] + "…") if len(content) > 220 else content

    title_html = (
        f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit;">{title}</a>'
        if url else title
    )
    meta_parts = [p for p in [published_at, source] if p]
    meta_html = " &nbsp;·&nbsp; ".join(
        f'<span style="background:rgba(100,120,180,0.15);border-radius:4px;padding:1px 7px;font-size:11px;">{p}</span>'
        for p in meta_parts
    )

    st.markdown(
        f"""<div style="
            border:1px solid rgba(128,128,128,0.18);
            border-radius:10px;
            padding:14px 16px 10px 16px;
            margin-bottom:10px;
            background:rgba(255,255,255,0.02);
        ">
            <div style="font-size:14px;font-weight:600;line-height:1.4;margin-bottom:6px;">
                {idx}. {title_html}
            </div>
            <div style="margin-bottom:8px;">{meta_html}</div>
            <div style="font-size:13px;color:rgba(200,200,200,0.85);line-height:1.5;">{content_short}</div>
        </div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    """yfinance üzerinden şirket bilgilerini çeker. Önbellek süresi: 1 saat."""
    try:
        import yfinance as yf
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def _fmt_large(val) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


def _fmt_pct(val, decimals: int = 1) -> str:
    if val is None:
        return "—"
    try:
        return f"%{float(val)*100:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ratio(val, decimals: int = 2) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}x"
    except (TypeError, ValueError):
        return "—"


def _rec_label(key: str) -> tuple:
    return {
        "strongBuy":  ("🟢 Güçlü Al",   "#00c47a"),
        "buy":        ("🟢 Al",          "#34d9a5"),
        "hold":       ("🟡 Tut",         "#f0c040"),
        "sell":       ("🔴 Sat",         "#ff7080"),
        "strongSell": ("🔴 Güçlü Sat",   "#ff4455"),
    }.get(key, ("—", "#888"))


def _chip(icon: str, text: str) -> str:
    return (
        f'<span style="background:rgba(99,179,237,0.12);border-radius:6px;'
        f'padding:3px 10px;font-size:12px;color:#a0b4cc;">{icon} {text}</span>'
    )




# ── Günlük haber cache dizini ────────────────────────────────────────────────
_NEWS_CACHE_DIR = Path(BASE_DIR) / "data" / "news_cache"


def _news_cache_path(ticker: str) -> Path:
    """Bugünün tarihine göre cache dosya yolunu döndürür."""
    from datetime import date
    today = date.today().strftime("%Y%m%d")
    return _NEWS_CACHE_DIR / f"{ticker}_{today}.json"


def fetch_marketaux_news(selected_ticker: str, selected_label: str) -> dict:
    """
    Günlük disk cache: aynı ticker gün içinde ilk kez çekildiğinde
    data/news_cache/{ticker}_{YYYYMMDD}.json dosyasına kaydedilir.
    Aynı gün içinde tekrar istenirse API'ye gitmeden dosyadan okunur.
    Ertesi gün otomatik olarak taze veri çekilir.
    """
    if not selected_ticker:
        return {"symbol": "", "industry": "", "ticker_news": [], "industry_news": []}

    cache_path = _news_cache_path(selected_ticker)

    # ── Bugüne ait cache varsa diskten oku ───────────────────────────────────
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if all(k in cached for k in ("ticker_news", "industry_news")):
                return cached
        except Exception:
            pass  # Bozuk cache → API'ye düş

    # ── API'den çek ──────────────────────────────────────────────────────────
    n = 10
    result = get_ticker_and_industry_news(
        selected_ticker,
        company_name=selected_label,
        country="us",
        n=n,
        per_req=n,
    )

    # ── Diske kaydet ─────────────────────────────────────────────────────────
    try:
        _NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(result, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass  # Kaydetme hatası kritik değil; veri zaten bellekte

    return result



def _build_system_instruction(hits: list[dict], answer_mode: str, length_hint: str = "") -> str:
    base_system = (hits[0].get("system") or "").strip() if hits else ""

    mode_hint = {
        "Resmi": "Resmi ve teknik bir dil kullan. Emoji kullanma. Kısa paragraflar ve net maddelerle yaz.",
        "Özet": "Sadece özeti ver: en fazla 8 madde veya 6-10 cümle. Gereksiz açıklama yapma.",
        "Anlaşılır 🙂": "Basit ve anlaşılır Türkçe kullan. Gerekirse az sayıda emoji kullanabilirsin. Adım adım anlat.",
    }.get(answer_mode, "Net ve yapılandırılmış yaz, tekrar etme.")

    guardrails_lines = [
        "Kurallar:",
        "- Yatırım tavsiyesi verme; al/sat/tut gibi kesin yönlendirme yapma.",
        "- Kesin getiri vaadi yok.",
        "- Güncel veri gerektiren noktada 'veri güncel olmayabilir' diye uyar.",
        "- Bilmediğin noktada uydurma; belirsizlikleri açıkça söyle.",
        f"- {mode_hint}",
    ]
    if length_hint:
        guardrails_lines.append(f"- Uzunluk: {length_hint}")

    guardrails = "\n".join(guardrails_lines) + "\n"

    if base_system:
        return f"{base_system}\n\n{guardrails}".strip()

    return (
        "Sen Türkçe finans asistanısın.\n"
        "Eğitim/araştırma amaçlı, ihtiyatlı ve anlaşılır yanıt ver.\n\n"
        f"{guardrails}"
    ).strip()


def _format_examples(hits: list[dict], k: int) -> str:
    blocks = []
    for i, h in enumerate(hits[:k], start=1):
        q = (h.get("user") or "").strip()
        a = (h.get("assistant") or "").strip()
        s = h.get("score", 0.0)
        if not q or not a:
            continue
        blocks.append(f"Örnek {i} (score={float(s):.3f})\nSoru: {q}\nCevap: {a}")
    return "\n\n".join(blocks).strip()


def _render_long_answer(full_text: str) -> None:
    """Chat mesajı içinde çağrılır; önizleme gösterir.
    Tam metin gösterimi chat bubble dışında _show_full_answer() ile yapılır."""
    full_text = (full_text or "").strip()
    if not full_text:
        st.markdown("Yanıt üretilemedi (boş döndü).")
        return
    if len(full_text) <= RAG_PREVIEW_CHARS:
        st.markdown(full_text)
    else:
        preview = full_text[:RAG_PREVIEW_CHARS].rstrip() + "…"
        st.markdown(preview)
        st.caption("Aşağıda ‘Tam Metin’ bölümünden devamını okuyabilirsin.")


def _show_full_answer_section(full_text: str, sid: str) -> None:
    """Chat bubble dışında çağrılır; tam metni expander içinde gösterir."""
    full_text = (full_text or "").strip()
    if not full_text:
        return
    with st.expander("📜 Tam Metni Göster", expanded=False):
        # Büyük metinlerde st.markdown render limitine takılmamak için
        # scrollable bir HTML div içinde göster
        escaped = (
            full_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        st.markdown(
            f'<div style="max-height:600px;overflow-y:auto;white-space:pre-wrap;'
            f'font-size:14px;line-height:1.6;padding:12px;'
            f'background:rgba(0,0,0,0.05);border-radius:8px;'
            f'border:1px solid rgba(128,128,128,0.2);">'
            f'{escaped}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "⇩ Yanıtı İndir (.txt)",
            data=full_text,
            file_name="finanalytics_rag_answer.txt",
            mime="text/plain",
            key=f"rag_dl_{sid}",
        )


st.set_page_config(
    page_title="FinAnalytics — Finansal Analiz",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="
        padding: 14px 0 18px 0;
        border-bottom: 1px solid rgba(99,179,237,0.2);
        margin-bottom: 18px;
    ">
        <div style="font-size:1.3rem;font-weight:800;letter-spacing:0.5px;
                    background:linear-gradient(90deg,#63b3ed,#90cdf4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            📈 FinAnalytics
        </div>
        <div style="font-size:11px;color:#556;margin-top:3px;letter-spacing:0.5px;">
            Finansal Analiz &amp; Tahmin Platformu
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_label = st.sidebar.selectbox(
    "💼 Hisse Seçin",
    [""] + list(TICKERS.keys()),
    help="Analiz etmek istediğiniz hisseyi seçin.",
)
selected_ticker = TICKERS.get(selected_label, "")

if not selected_ticker:
    st.markdown(
        """
        <div style="
            display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:60vh;text-align:center;gap:16px;
        ">
            <div style="font-size:3.5rem;">&#128202;</div>
            <div style="font-size:2rem;font-weight:700;color:#e8edf5;">
                FinAnalytics
            </div>
            <div style="font-size:1rem;color:#8892b0;max-width:460px;line-height:1.7;">
                Kısa / orta / uzun vadeli model çıktıları, haber bülteni ve
                yapay zeka destekli finansal analiz için sol panelden
                bir hisse seçin.
            </div>
            <div style="
                margin-top:8px;font-size:0.8rem;color:#3d4f6b;
                border:1px solid rgba(99,179,237,0.2);border-radius:8px;
                padding:8px 16px;
            ">
                Desteklenen: AAPL · MSFT · NVDA · CAG · HSY · KO · …
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Ana sayfa başlığı ─────────────────────────────────────────────
left, right = st.columns([5, 0.7], vertical_alignment="center")
with left:
    st.markdown(
        f'<h2 style="margin:0;font-weight:700;color:#e8edf5;">'
        f'{selected_label} '
        f'<span style="font-size:1rem;font-weight:400;color:#63b3ed;">({selected_ticker})</span>'
        f'</h2>',
        unsafe_allow_html=True,
    )
with right:
    render_logo_or_placeholder(selected_ticker)

tabs = st.tabs(["🏢 Hakkında", "📉 Kısa Vadeli", "📊 Orta Vadeli", "📈 Uzun Vadeli", "📰 Haber Bülteni", "🤖 FinAI", "📄 Raporlar"])

with tabs[0]:
    with st.spinner(f"{selected_label} bilgileri yükleniyor…"):
        info = fetch_ticker_info(selected_ticker)

    if not info:
        st.warning("⚠️ Yahoo Finance’ten veri alınamadı. İnternet bağlantınızı kontrol edin.")
    else:
        # ── Üst kimlik şeridi ──────────────────────────────────────────────────────
        long_name = info.get("longName") or selected_label
        sector    = info.get("sector", "—")
        industry  = info.get("industryDisp") or info.get("industry", "—")
        country   = info.get("country", "—")
        exchange  = info.get("exchange", "—")
        website   = info.get("website", "")
        employees = info.get("fullTimeEmployees")
        emp_str   = f"{employees:,}" if employees else "—"

        chips_html = " ".join([
            _chip("🏭", industry),
            _chip("🌍", f"{sector} · {country}"),
            _chip("🏛", exchange),
            _chip("👥", f"{emp_str} çalışan"),
        ])
        # Web sitesi linki
        link_html = (
            f'<a href="{website}" target="_blank" style="text-decoration:none;">'
            + _chip("🔗", "Web Sitesi") + "</a>"
        ) if website else ""  # noqa

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg,#0d1117 60%,#131c2e);
                border:1px solid rgba(99,179,237,0.18);
                border-radius:14px;
                padding:22px 26px;
                margin-bottom:18px;
            ">
                <div style="font-size:1.5rem;font-weight:700;color:#e8edf5;margin-bottom:4px;">
                    {long_name}
                    <span style="font-size:1rem;font-weight:400;color:#63b3ed;margin-left:10px;">
                        {selected_ticker}
                    </span>
                </div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
                    {chips_html} {link_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary = info.get("longBusinessSummary", "")
        if summary:
            with st.expander("📝 Şirket Özeti", expanded=True):
                st.markdown(summary)


        st.markdown("---")

        # ── Ortak CSS ──────────────────────────────────────────────────────────────
        st.markdown("""
        <style>
        .sc-grid{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:24px;}
        .sc-card{
            background:rgba(255,255,255,0.035);
            border:1px solid rgba(128,128,128,0.18);
            border-radius:12px;
            padding:16px 18px 14px;
            flex:1 1 140px;
            min-width:130px;
            position:relative;
            transition:border-color 0.2s,background 0.2s;
        }
        .sc-card:hover{border-color:rgba(99,179,237,0.4);background:rgba(99,179,237,0.07);}
        .sc-label{font-size:11px;color:#8892b0;display:flex;align-items:center;gap:5px;margin-bottom:8px;line-height:1.3;}
        .sc-value{font-size:1.22rem;font-weight:700;color:#e8edf5;line-height:1.2;}
        .tip{
            display:inline-flex;align-items:center;justify-content:center;
            width:15px;height:15px;
            background:rgba(99,179,237,0.14);
            border:1px solid rgba(99,179,237,0.30);
            border-radius:50%;font-size:9px;color:#63b3ed;
            cursor:help;position:relative;flex-shrink:0;
        }
        .tip::after{
            content:attr(data-tip);
            position:absolute;bottom:130%;left:50%;transform:translateX(-50%);
            background:#151d2e;color:#c8d3e8;
            padding:7px 11px;border-radius:7px;
            font-size:11px;white-space:normal;width:210px;
            z-index:9999;border:1px solid rgba(99,179,237,0.22);
            box-shadow:0 6px 20px rgba(0,0,0,0.5);
            pointer-events:none;opacity:0;transition:opacity 0.15s;line-height:1.5;
            font-weight:400;
        }
        .tip:hover::after{opacity:1;}
        </style>
        """, unsafe_allow_html=True)

        def _sc(icon, label, value, tooltip, value_color="#e8edf5", extra_html=""):
            return (
                f'<div class="sc-card">'
                f'<div class="sc-label">{icon} {label}'
                f'<span class="tip" data-tip="{tooltip}">?</span></div>'
                f'<div class="sc-value" style="color:{value_color};">{value}{extra_html}</div>'
                f'</div>'
            )

        def _sc_row(cards):
            st.markdown(f'<div class="sc-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

        # ── Piyasa & Fiyat ─────────────────────────────────────────────────────────
        st.markdown("#### 📊 Piyasa ve Fiyat Verileri")
        p52h = info.get("fiftyTwoWeekHigh")
        p52l = info.get("fiftyTwoWeekLow")
        _sc_row([
            _sc("💰", "Piyasa Değeri", _fmt_large(info.get("marketCap")),
                "Şirketi tamamen satın almak için gereken teorik toplam bedel."),
            _sc("📈", "52H Yüksek", f"${p52h:.2f}" if p52h else "—",
                "Son 52 haftada ulaşılan en yüksek hisse fiyatı."),
            _sc("📉", "52H Düşük", f"${p52l:.2f}" if p52l else "—",
                "Son 52 haftada görülen en düşük hisse fiyatı."),
            _sc("📊", "Beta", f"{info['beta']:.2f}" if info.get("beta") else "—",
                "Hissenin piyasaya göre oynanlık ölçüsü; 1’den büyükse daha volatil."),
            _sc("📀", "Ort. Hacim", f"{info['averageVolume']:,}" if info.get("averageVolume") else "—",
                "Son 3 aylık ortalama günlük işlem hacmi (lot sayısı)."),
        ])

        # ── Değerleme ───────────────────────────────────────────────────────────────
        st.markdown("#### 🔍 Değerleme Katsayıları")
        _sc_row([
            _sc("📐", "F/K (Trailing)", _fmt_ratio(info.get("trailingPE")),
                "Son 12 aylık gerçekleşen kazınca göre fiyat/kazanç oranı."),
            _sc("🔮", "F/K (Forward)",  _fmt_ratio(info.get("forwardPE")),
                "Önümüzdeki 12 ay için beklenen kazınca göre fiyat/kazanç oranı."),
            _sc("📦", "FİD/Satış",      _fmt_ratio(info.get("priceToSalesTrailing12Months")),
                "Piyasa değerinin son 12 aylık toplam satış gelirine oranı."),
            _sc("📋", "F/DD",           _fmt_ratio(info.get("priceToBook")),
                "Hisse fiyatının defter değerine (net varlık) oranı."),
            _sc("💵", "Temettü Verimi", _fmt_pct(info.get("dividendYield")),
                "Yıllık temettünün mevcut hisse fiyatına oranı."),
        ])

        # ── Büyüme & Karlılık ─────────────────────────────────────────────────────────
        st.markdown("#### 📈 Büyüme & Karlılık")
        _sc_row([
            _sc("🚀", "Gelir Büyümesi",     _fmt_pct(info.get("revenueGrowth")),
                "Son çeyrekte yıllık bazda (YoY) gelir büyüme oranı."),
            _sc("📊", "Kazanç Büyümesi",    _fmt_pct(info.get("earningsGrowth")),
                "Son çeyrekte yıllık bazda hisse başına kazanç (EPS) büyüme oranı."),
            _sc("🟦", "Brüt Marj",          _fmt_pct(info.get("grossMargins")),
                "Üretim maliyetleri düşüldükten sonra kalan kârın gelire oranı."),
            _sc("⚙️", "İşletme Marjı",      _fmt_pct(info.get("operatingMargins")),
                "Faaliyet giderleri sonrası işletme kârının gelire oranı."),
            _sc("💹", "Özkaynak Karlılığı", _fmt_pct(info.get("returnOnEquity")),
                "Net kârın ortalama özkaynakların oranı; verimlilik ölçüsü."),
        ])

        # ── Bilanço ─────────────────────────────────────────────────────────────────────────
        st.markdown("#### 🏦 Bilanço Özeti")
        de = info.get("debtToEquity")
        _sc_row([
            _sc("💼", "Toplam Gelir",       _fmt_large(info.get("totalRevenue")),
                "Son 12 ayda tüm faaliyetlerden elde edilen toplam gelir."),
            _sc("💰", "Net Kar",            _fmt_large(info.get("netIncomeToCommon")),
                "Tüm giderler ve vergiler düşüldükten sonra hissedarlara kalan net kâr."),
            _sc("🌊", "Serbest Nakit Akış", _fmt_large(info.get("freeCashflow")),
                "Sermaye harcamaları sonrası şirkette kalan nakit; büyüme kapasitesini gösterir."),
            _sc("⚖️", "Borç / Özkaynak",   f"{de:.1f}" if de else "—",
                "Toplam borcun özkaynakların oranı; düşük değer daha az finansal risk anlamına gelir."),
        ])

        st.markdown("---")

        # ── Yönetim Ekibi ─────────────────────────────────────────────────────────────
        officers = (info.get("companyOfficers") or [])[:6]
        if officers:
            st.markdown("#### 👤 Yönetim Ekibi")
            oc = st.columns(3)
            for idx, off in enumerate(officers):
                oname   = off.get("name", "—")
                otitle  = off.get("title", "—")
                opay    = off.get("totalPay")
                pay_str = f"Toplam Ücret: {_fmt_large(opay)}" if opay else ""
                with oc[idx % 3]:
                    st.markdown(
                        f'<div style="border:1px solid rgba(128,128,128,0.15);'
                        f'border-radius:10px;padding:12px 14px;margin-bottom:10px;'
                        f'background:rgba(255,255,255,0.02);">'
                        f'<div style="font-size:13px;font-weight:600;color:#e8edf5;">{oname}</div>'
                        f'<div style="font-size:11px;color:#63b3ed;margin:3px 0;">{otitle}</div>'
                        f'<div style="font-size:10px;color:#8892b0;">{pay_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Kurumsal Risk Skorları (score cards) ──────────────────────────────────
        audit   = info.get("auditRisk")
        board   = info.get("boardRisk")
        comp_r  = info.get("compensationRisk")
        overall = info.get("overallRisk")
        if any(x is not None for x in [audit, board, comp_r, overall]):
            st.markdown("---")
            st.markdown("#### 🛡️ Kurumsal Yönetim Risk Skorları")
            st.caption("ISP Yönetişim Skoru — Düşük değer = daha az risk. Maksimum 10.")

            def _risk_sc(icon, label, val, tooltip):
                if val is None:
                    return _sc(icon, label, "—", tooltip)
                v = int(val)
                color = "#00c47a" if v <= 3 else ("#f0c040" if v <= 6 else "#ff7080")
                sub = '<span style="font-size:0.72rem;color:#555;"> /10</span>'
                return _sc(icon, label, str(v), tooltip, value_color=color, extra_html=sub)

            _sc_row([
                _risk_sc("🔎", "Denetim",         audit,
                         "Denetim komitesi bağımsızlığı ve şeffaflık riski; 1=en iyi, 10=en kötü."),
                _risk_sc("🏛", "Yönetim Kurulu",  board,
                         "Yönetim kurulunun bağımsızlık ve çeşitlilik bakımından taşıdığı risk."),
                _risk_sc("💳", "Ücret",           comp_r,
                         "Yönetici ücret paketlerinin hissedar çıkarlarıyla uyumu riski."),
                _risk_sc("⚠️", "Genel Risk",      overall,
                         "Tüm kurumsal yönetim bileşenlerinin birleşik genel riski."),
            ])

        st.markdown("")
        st.caption(
            "⚠️ Veriler Yahoo Finance üzerinden çekilmektedir (önbellek süresi: 1 saat). "
            "Yatırım tavsiyesi niteliği taşımaz."
        )

with tabs[1]:
    render_short_dashboard(selected_ticker)

with tabs[2]:
    render_mid_dashboard(selected_ticker)

with tabs[3]:
    render_long_dashboard(selected_ticker)

with tabs[4]:
    # ── Önbellek durumu göstergesi ───────────────────────────────────────────
    _cp = _news_cache_path(selected_ticker) if selected_ticker else None
    _from_cache = bool(_cp and _cp.exists())

    if _from_cache:
        from datetime import date as _d
        st.success(
            f"📦 **{_d.today().strftime('%d %B %Y')}** tarihli önbellekten yükleniyor — "
            "gün içinde API'ye tekrar istek atılmadı.",
            icon="💾",
        )
    st.caption(
        f"**{selected_label} ({selected_ticker})** için güncel şirket ve sektör haberleri. "
        "Veriler Marketaux API üzerinden çekilir; aynı ticker gün içinde yalnızca **bir kez** sorgulanır."
    )

    try:
        _spinner_msg = "Önbellekten yükleniyor…" if _from_cache else "Haberler çekiliyor…"
        with st.spinner(_spinner_msg):
            result = fetch_marketaux_news(selected_ticker, selected_label)


        ticker_news = result.get("ticker_news", [])
        industry_news = result.get("industry_news", [])
        symbol = result.get("symbol", "")
        industry = result.get("industry", "")

        # ── Özet metrik kartları ──────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📌 Hisse Sembolü", symbol or "—")
        m2.metric("🏭 Sektör", (industry[:22] + "…") if len(industry) > 22 else (industry or "—"))
        m3.metric("📄 Şirket Haberi", len(ticker_news))
        m4.metric("🌐 Sektör Haberi", len(industry_news))

        st.markdown("---")

        # ── LLM bağlamı (FinAI sekmesinde kullanılır) ────────────────────
        include_links = True  # URL her zaman dahil (LLM bağlamı için)
        ticker_ctx, industry_ctx = build_llm_context(
            symbol=symbol,
            industry=industry,
            ticker_news=ticker_news,
            industry_news=industry_news,
            include_url=include_links,
            max_items=10,
            max_snippet_chars=500,
        )
        with st.expander("🔗 FinAI için Haber Bağlamını Göster", expanded=False):
            st.caption("Bu metin FinAI sekmesinde 'Haber bağlamını ekle' seçildiğinde Gemini'ye iletilir.")
            st.text_area("Şirket Bağlamı", ticker_ctx, height=200, key="llm_ticker_ctx")
            st.text_area("Sektör Bağlamı", industry_ctx, height=200, key="llm_industry_ctx")

        st.markdown("---")

        # ── İki sütunlu haber listesi ─────────────────────────────────────
        col_ticker, col_industry = st.columns(2, gap="large")

        with col_ticker:
            st.subheader(f"🏢 Şirket Haberleri ({len(ticker_news)})")
            if ticker_news:
                for i, it in enumerate(ticker_news, start=1):
                    render_news_item(i, "Şirket", it)
            else:
                st.info("Bu hisse için haber bulunamadı.")

        with col_industry:
            st.subheader(f"🌐 Sektör Haberleri ({len(industry_news)})")
            if industry_news:
                for i, it in enumerate(industry_news, start=1):
                    render_news_item(i, "Sektör", it)
            else:
                st.info("Bu sektör için haber bulunamadı.")

    except RuntimeError as e:
        err_str = str(e)
        if "401" in err_str:
            st.error("🔑 API token geçersiz veya süresi dolmuş.")
            st.info("`.env` dosyasındaki `MARKETAUX_API_TOKEN` değerini kontrol edin.")
        elif "429" in err_str:
            st.warning("⏳ Marketaux API kotası doldu. Birkaç dakika sonra tekrar deneyin.")
        else:
            st.error(f"Marketaux hata: {e}")
            st.info("Kontrol: `set -a && source .env && set +a` (MARKETAUX_API_TOKEN yüklü mü?)")
    except Exception as e:
        st.error(f"Beklenmedik hata: {e}")

with tabs[5]:
    # ── Başlık & açıklama ────────────────────────────────────────────────────
    st.markdown("## 🤖 FinAI — Finans Asistanı")
    st.caption(
        "SFT dataset'ten semantik olarak benzer soru/cevap çiftleri çekilir (retrieval), "
        "ardından Gemini bu örnekleri bağlam olarak kullanarak yanıt üretir. "
        "**Güncel piyasa verisi sağlamaz; yalnızca eğitim/araştırma amaçlıdır.**"
    )

    # ── Dataset atıf kartı ───────────────────────────────────────────────
    with st.expander("🌐 Veri Kaynağı — Turkish Finance SFT Dataset", expanded=False):
        st.markdown(
            """
**🇹🇷 Turkish Finance SFT Dataset** · [Hugging Face'te Görüntüle](https://huggingface.co/datasets/AlicanKiraz0/Turkish-Finance-SFT-Dataset)

Bu sekme, Retrieval-Augmented Generation (RAG) akışında semantik retrieval kaynağı olarak
[**Turkish Finance SFT Dataset**](https://huggingface.co/datasets/AlicanKiraz0/Turkish-Finance-SFT-Dataset)'i kullanmaktadır.

| Özellik | Detay |
|---------|-------|
| **Yazar** | [Alican Kiraz](https://huggingface.co/AlicanKiraz0) |
| **Kapsam** | ~10 milyon token, Türkçe finans soru-cevap çiftleri |
| **Kategoriler** | Kripto para, Borsa & Hisse Senetleri, Teknik Analiz, Temel Analiz, Risk Yönetimi |
| **Piyasalar** | BIST (Türkiye) + Global (NASDAQ, S&P 500, Kripto borsaları) |
| **Lisans** | MIT |
| **⚠️ Sorumluluk** | Yalnızca eğitim/araştırma amaçlıdır; yatırım tavsiyesi niteliği taşımaz. |

```bibtex
@dataset{kiraz2025turkishfinance,
  title     = {Turkish Finance SFT Dataset},
  author    = {Kiraz, Alican},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/AlicanKiraz0/Turkish-Finance-SFT-Dataset}
}
```
            """
        )

    # ── Bilgi kartları ───────────────────────────────────────────────────────
    info_c1, info_c2, info_c3, info_c4 = st.columns(4)
    info_c1.metric("🔍 Retrieval", "SFT Dataset", help="Eğitim dataseti üzerinden benzer örnekler çekilir.")
    info_c2.metric("🧠 Model", "Gemini 1.5 Flash", help="Google Gemini API üzerinden çalışır.")
    info_c3.metric("📈 Hisse", f"{selected_ticker}", help="Sohbet bağlamı bu hisse için özelleştirilir.")
    rag_session_key_info = f"rag_messages_{selected_ticker}"
    _msg_count = max(0, len(st.session_state.get(rag_session_key_info, [])) - 1)
    info_c4.metric("💬 Mesaj", _msg_count, help="Bu oturumda gönderilen mesaj sayısı.")

    st.markdown("---")

    # ── Ana düzen: sohbet (sol) + ayarlar (sağ) ─────────────────────────────
    rag_left, rag_right = st.columns([3, 1.6], vertical_alignment="top")

    # ── AYARLAR PANELİ ───────────────────────────────────────────────────────
    with rag_right:
        with st.container(border=True):
            st.markdown("#### ⚙️ Ayarlar")

            # Top-K sabit: 2 (UI'dan kaldırıldı)
            top_k = 2


            st.markdown("**✍️ Yanıt Stili**")
            answer_mode = st.selectbox(
                "Yanıt tarzı",
                ["Resmi", "Özet", "Anlaşılır 🙂"],
                index=0,
                key="rag_answer_mode",
                help=(
                    "**Resmi:** Teknik ve akademik dil, emoji yok. "
                    "**Özet:** Madde madde, kısa ve öz. "
                    "**Anlaşılır:** Sade Türkçe, adım adım açıklama."
                ),
            )

            st.markdown("**📊 Yanıt Uzunluğu**")
            length_label = st.radio(
                "Yanıt uzunluğu",
                options=list(RAG_LENGTH_OPTIONS.keys()),
                index=1,          # varsayılan: Orta
                key="rag_length_label",
                horizontal=True,
                help=(
                    "**Kısa:** 1 paragraf, özlü yanıt. "
                    "**Orta:** 2 paragraf, dengeli açıklama. "
                    "**Uzun:** 3 paragraf, detaylı analiz."
                ),
            )
            max_tokens, length_hint = RAG_LENGTH_OPTIONS[length_label]

            st.markdown("**🌡️ Yaratıcılık (Temperature)**")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                key="rag_temperature",
                help=(
                    "0'a yakın: tutarlı ve deterministik yanıtlar. "
                    "1'e yakın: daha yaratıcı ve çeşitli yanıtlar. "
                    "Finans soruları için 0.2–0.5 önerilir."
                ),
            )

            st.markdown("**🛡️ Guardrail & Bağlam**")
            enable_guardrail = st.toggle(
                "Yatırım uyarısı ekle",
                value=True,
                key="rag_guardrail",
                help=(
                    "Aktif olduğunda her yanıtın başına/sonuna 'yatırım tavsiyesi değildir' "
                    "uyarısı sistem talimatı olarak eklenir. Kapatmak araştırma bağlamında "
                    "daha akıcı yanıtlar sağlayabilir."
                ),
            )

            include_news_ctx = st.toggle(
                "Haber bağlamını ekle",
                value=False,
                key="rag_include_news_ctx",
                help=(
                    "Haber Bülteni sekmesinde çekilen şirket/sektör haberleri varsa "
                    "prompt bağlamına eklenir. Bu sayede model güncel gelişmeleri de "
                    "dikkate alarak yanıt üretir."
                ),
            )
            # 4. Haber Bağlamı Uyarısı
            if include_news_ctx:
                ticker_ctx = (st.session_state.get("llm_ticker_ctx") or "").strip()
                industry_ctx = (st.session_state.get("llm_industry_ctx") or "").strip()
                if not (ticker_ctx or industry_ctx):
                    st.info("ℹ️ Haber bağlamı boş. Lütfen önce **Haber Bülteni** sekmesine gidip güncel haberleri çekin.")

            st.markdown("**🔬 Geliştirici**")
            dev_mode = os.getenv(RAG_DEBUG_ENV_FLAG, "").strip() == "1"
            show_hits = False
            if dev_mode:
                show_hits = st.toggle(
                    "Retrieval debug",
                    value=False,
                    key="rag_show_hits",
                    help="Çekilen örnekleri, benzerlik skorlarını ve ham prompt'u gösterir.",
                )

            st.markdown("---")

            # Örnek sorular
            st.markdown("**💡 Örnek Sorular**")
            example_questions = [
                "RSI ve MACD birlikte nasıl yorumlanır?",
                "P/E oranı nedir, nasıl kullanılır?",
                "Bollinger Band sıkışması ne anlama gelir?",
                "Temettü verimi nasıl hesaplanır?",
            ]
            for eq in example_questions:
                if st.button(eq, key=f"eq_{hash(eq)}", use_container_width=True):
                    st.session_state[f"_rag_prefill_{selected_ticker}"] = eq
                    st.rerun()

            st.markdown("---")

            col_rst, col_info = st.columns([1, 1])
            with col_rst:
                if st.button("🗑️ Sıfırla", key="rag_reset", use_container_width=True,
                             help="Bu hisse için RAG konuşma geçmişini temizler."):
                    st.session_state.pop(f"rag_messages_{selected_ticker}", None)
                    st.rerun()
            with col_info:
                st.markdown(
                    f"<div style='font-size:11px; color: grey; padding-top:6px;'>"
                    f"🔵 {selected_label}<br>⚡ gemini-1.5-flash</div>",
                    unsafe_allow_html=True,
                )

    # ── SOHBETt ALANI ────────────────────────────────────────────────────────
    with rag_left:
        session_key = f"rag_messages_{selected_ticker}"
        if session_key not in st.session_state:
            st.session_state[session_key] = [
                {
                    "role": "assistant",
                    "content": (
                        f"👋 **{selected_label} ({selected_ticker})** hakkında finans sorularını yanıtlamaya hazırım.\n\n"
                        "Sormak istediğin teknik analiz göstergeleri, temel analiz kavramları veya "
                        "genel finans konularını yazabilirsin.\n\n"
                        "> ⚠️ Yanıtlar yatırım tavsiyesi değildir; yalnızca eğitim/araştırma amaçlıdır."
                    ),
                }
            ]

        # Prefill desteği (örnek soru butonlarından)
        prefill_key = f"_rag_prefill_{selected_ticker}"
        prefill_q = st.session_state.pop(prefill_key, None)

        for m in st.session_state[session_key]:
            role = m.get("role", "assistant")
            content = m.get("content", "")
            full = m.get("full", "")
            with st.chat_message(role):
                # Chat bubble içinde önizleme (veya tam metin kısaysa)
                display_text = full if (role == "assistant" and full) else content
                _render_long_answer(display_text)
            # Asistan yanıtları için her zaman "Tam Metni Göster" expander'ı göster
            # (ilk hoş geldin mesajını atla)
            if role == "assistant":
                answer_text = full if full else content
                welcome_content = st.session_state[session_key][0].get("content", "")
                if answer_text and answer_text.strip() != welcome_content.strip():
                    _show_full_answer_section(answer_text, sid=_stable_id(answer_text))

        # Chat input (prefill varsa göster)
        user_q = st.chat_input("Finans sorunuzu yazın…") or prefill_q
        if user_q:
            full_answer = None
            st.session_state[session_key].append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("🔍 Benzer örnekler aranıyor..."):
                        hits = retrieve_examples(user_q, k=top_k)

                    if not hits:
                        msg = "Benzer örnek bulunamadı. Soruyu biraz daha detaylandırabilir misin?"
                        st.warning(msg)
                        st.session_state[session_key].append({"role": "assistant", "content": msg, "full": msg})
                        # st.stop() chat bubble içinde çağrılmamalı — exception ile bloktan çık
                        raise StopIteration()

                    # Guardrail flag'i sistem talimatına aktar
                    _orig_guardrail = enable_guardrail
                    system_instruction = _build_system_instruction(
                        hits, answer_mode=answer_mode, length_hint=length_hint
                    )
                    if not _orig_guardrail:
                        # Guardrail devre dışı → uyarı satırlarını çıkar
                        system_instruction = "\n".join(
                            ln for ln in system_instruction.splitlines()
                            if "tavsiyesi" not in ln and "vaadi" not in ln
                        ).strip()

                    examples_text = _format_examples(hits, k=top_k)

                    news_ctx = ""
                    if include_news_ctx:
                        ticker_ctx = (st.session_state.get("llm_ticker_ctx") or "").strip()
                        industry_ctx = (st.session_state.get("llm_industry_ctx") or "").strip()
                        if ticker_ctx or industry_ctx:
                            news_ctx = (
                                "Ek bağlam (haber/sektör):\n\n"
                                f"Şirket bağlamı:\n{ticker_ctx}\n\n"
                                f"Sektör bağlamı:\n{industry_ctx}\n"
                            ).strip()

                    prompt_parts = [
                        "Aşağıdaki örnekler benzer soru/cevap örnekleridir. Kopyalamadan, aynı mantıkla yanıt üret.",
                        "",
                        examples_text,
                        "",
                    ]
                    if news_ctx:
                        prompt_parts += [news_ctx, ""]

                    prompt_parts += [
                        f"Kullanıcı sorusu: {user_q}",
                        "",
                        "Yanıt:",
                    ]
                    prompt = "\n".join([p for p in prompt_parts if p is not None]).strip()

                    with st.spinner("✨ Gemini yanıt üretiyor..."):
                        answer = generate_text(
                            prompt=prompt,
                            system_instruction=system_instruction,
                            model="gemini-1.5-flash",
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                        )

                    full_answer = (answer or "").strip()
                    if not full_answer:
                        full_answer = "Yanıt üretilemedi (boş döndü)."

                    # Chat bubble içinde önizleme göster
                    _render_long_answer(full_answer)

                    # Meta bilgi (bubble içinde)
                    meta_cols = st.columns(4)
                    meta_cols[0].caption(f"📝 {len(full_answer)} kar.")
                    meta_cols[1].caption(f"🔍 {len(hits)} örnek")
                    meta_cols[2].caption(f"🌡️ temp={temperature}")
                    meta_cols[3].caption(f"⚡ {length_label}")

                    # Session state'e kaydet — her zaman 'full' anahtarıyla
                    # (kısa yanıtta content == full, uzun yanıtta content önizleme olur)
                    if len(full_answer) > RAG_PREVIEW_CHARS:
                        preview_saved = full_answer[:RAG_PREVIEW_CHARS].rstrip() + "…"
                        st.session_state[session_key].append(
                            {"role": "assistant", "content": preview_saved, "full": full_answer}
                        )
                    else:
                        st.session_state[session_key].append(
                            {"role": "assistant", "content": full_answer, "full": full_answer}
                        )

                    if show_hits:
                        with st.expander("🔬 Retrieval debug"):
                            for i, h in enumerate(hits, 1):
                                st.markdown(f"**Örnek {i}** — score: `{h.get('score', 0):.4f}`")
                                st.markdown(f"> **Soru:** {h.get('user', '')}")
                                st.markdown(f"> **Cevap:** {h.get('assistant', '')[:300]}...")
                                st.markdown("---")

                except StopIteration:
                    pass  # hits boş — kullanıcıya uyarı gösterildi, devam et
                except Exception as e:
                    full_answer = None
                    st.error(f"❌ Hata: {e}")
                    st.info("Kontrol: `.env` içinde `GEMINI_API_KEY` var mı? (`set -a && source .env && set +a`)")

            # Tam metin expander'ı chat bubble DIŞINDA — her yanıt için göster
            if full_answer:
                _show_full_answer_section(full_answer, sid=_stable_id(full_answer))

with tabs[6]:
    st.header("📄 Raporlar")
    st.caption(
        "Model çıktılarını ve güncel haber özetlerini PDF olarak indirin. "
        "Finansal Rapor kısa/orta/uzun vadeli model sinyallerini, "
        "Gündem Raporu ise Haber Bülteni haberlerinin özetlerini içerir."
    )

    st.markdown("---")
    col_fin, col_agenda = st.columns(2, gap="large")

    # ── FİNANSAL RAPOR ───────────────────────────────────────────
    with col_fin:
        st.subheader("📊 Finansal Rapor")
        st.markdown(
            """
            İndirilen PDF içerir:
            - **Kısa Vadeli** (1/3/5/7 gün): Sinyal, güven skoru, olasılık
            - **Orta Vadeli** (1 Ay / 3 Ay): Fiyat tahmini, getiri, sinyal
            - **Uzun Vadeli** (22-63 gün): Model sinyali, skor, momentum metrikleri
            """
        )

        gen_fin = st.button(
            "💾 Finansal Raporu İndir (PDF)",
            key="btn_gen_fin_report",
            use_container_width=True,
        )

        if gen_fin:
            with st.spinner("Model verileri toplanıyor ve PDF oluşturuluyor…"):
                try:
                    # Kisa vadeli
                    from src.short import _run_prediction, _fetch_live
                    preds = _run_prediction(selected_ticker)
                    live_df = _fetch_live(selected_ticker)
                    short_price = (
                        float(live_df["Close"].iloc[-1])
                        if live_df is not None and not live_df.empty else None
                    )
                    short_rows = []
                    h_labels = {"1d": "1 Gün", "3d": "3 Gün", "5d": "5 Gün", "7d": "7 Gün"}
                    for hk in ["1d", "3d", "5d", "7d"]:
                        if hk in preds:
                            p = preds[hk]
                            short_rows.append({
                                "horizon":    h_labels[hk],
                                "signal":     p["signal"],
                                "confidence": p["confidence"],
                                "prob_up":    p["prob_up"],
                            })
                    short_data = {
                        "anchor": preds.get("anchor", "—"),
                        "price":  short_price,
                        "signal_rows": short_rows,
                    } if preds else None
                except Exception:
                    short_data = None

                try:
                    # Orta vadeli
                    from src.mid import _load_json_report, _get_stock_prediction
                    mid_report = _load_json_report()
                    _mid_result = _get_stock_prediction(mid_report, selected_ticker)
                    # _get_stock_prediction (pred, sector_name) 
                    mid_pred = None
                    mid_sector = None
                    if _mid_result and len(_mid_result) >= 2:
                        mid_pred   = _mid_result[0]
                        mid_sector = _mid_result[1]

                    mid_data = {
                        "son_fiyat":      mid_pred.get("son_fiyat", 0),
                        "tahmin_1ay":     mid_pred.get("tahmin_1ay", 0),
                        "tahmin_3ay":     mid_pred.get("tahmin_3ay", 0),
                        "getiri_1ay_pct": mid_pred.get("getiri_1ay_pct", 0),
                        "getiri_3ay_pct": mid_pred.get("getiri_3ay_pct", 0),
                        "sinyal_1ay":     mid_pred.get("sinyal_1ay", "—"),
                        "sinyal_3ay":     mid_pred.get("sinyal_3ay", "—"),
                        "sektor":         mid_sector or "—",
                    } if mid_pred else None
                except Exception as _e:
                    mid_data = None

                try:
                    # Uzun vadeli
                    from src.long import _load_data_and_predict
                    long_results, _, long_err = _load_data_and_predict()
                    long_data = None
                    if long_results is not None and selected_ticker in long_results["ticker"].tolist():
                        row = long_results[long_results["ticker"] == selected_ticker].iloc[0]
                        long_data = {
                            "price":        float(row["close"]),
                            "signal":       str(row["pred_label"]),
                            "score":        float(row["model_score"]),
                            "percentile":   float(row["percentile_rank"]),
                            "prob_up":      float(row["prob_up"]),
                            "prob_neutral": float(row["prob_neutral"]),
                            "prob_down":    float(row["prob_down"]),
                            "momentum":     float(row["mom_126"]),
                            "slope":        float(row["sma50_slope20"]),
                            "dist_sma200":  float(row["dist_sma_200"]),
                        }
                except Exception:
                    long_data = None

                pdf_bytes = build_financial_pdf(
                    ticker=selected_ticker,
                    label=selected_label,
                    short_data=short_data,
                    mid_data=mid_data,
                    long_data=long_data,
                )

            fname = f"finansal_rapor_{selected_ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                "⬇️ PDF’yi İndir",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                key="dl_fin_pdf",
                use_container_width=True,
            )
            st.success("✅ PDF hazır! Yukarıdaki butona tıkla.")

    # ── GÜNDEM RAPORU ─────────────────────────────────────────────
    with col_agenda:
        st.subheader("📰 Gündem Raporu")
        st.markdown(
            """
            İndirilen PDF içerir:
            - **10 Şirket Haberi**: Her biri 1 paragraf özet
            - **10 Sektör Haberi**: Her biri 1 paragraf özet
            - Kaynak ve tarih bilgisi
            """
        )

        gen_agenda = st.button(
            "💾 Gündem Raporunu İndir (PDF)",
            key="btn_gen_agenda_report",
            use_container_width=True,
        )

        if gen_agenda:
            try:
                with st.spinner("Haberler getiriliyor ve PDF oluşturuluyor…"):
                    result = fetch_marketaux_news(selected_ticker, selected_label)
                    ticker_news   = result.get("ticker_news", [])
                    industry_news = result.get("industry_news", [])
                    symbol   = result.get("symbol", "")
                    industry = result.get("industry", "")

                    pdf_bytes = build_agenda_pdf(
                        ticker=selected_ticker,
                        label=selected_label,
                        ticker_news=ticker_news,
                        industry_news=industry_news,
                        symbol=symbol,
                        industry=industry,
                    )

                fname = f"gundem_raporu_{selected_ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    "⬇️ PDF’yi İndir",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    key="dl_agenda_pdf",
                    use_container_width=True,
                )
                st.success("✅ PDF hazır! Yukarıdaki butona tıkla.")

            except RuntimeError as e:
                err_str = str(e)
                if "401" in err_str:
                    st.error("🔑 Marketaux API token geçersiz.")
                elif "429" in err_str:
                    st.warning("⏳ Marketaux kota doldu. Birkaç dakika sonra tekrar deneyin.")
                else:
                    st.error(f"Haber çekme hatası: {e}")
                    st.info("`.env` içinde `MARKETAUX_API_TOKEN` var mı?")
            except Exception as e:
                st.error(f"Beklenmedik hata: {e}")

