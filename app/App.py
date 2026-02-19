
import hashlib
import os
import re
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.integrations.gemini import generate_text
from src.integrations.marketaux import get_ticker_and_industry_news
from src.rag.turkish_finance_sft_rag import retrieve_examples
from src.reports.news_prompt import build_llm_context
from pages.short import render_short_dashboard
from pages.mid import render_mid_dashboard
from pages.long import render_long_dashboard

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

# RAG sabitleri (UI'dan kaldırıldı)
RAG_PREVIEW_CHARS = 1800
RAG_MAX_OUTPUT_TOKENS = 700
RAG_DEBUG_ENV_FLAG = "RAG_DEBUG"  # RAG_DEBUG=1 ise debug toggle görünür


def _stable_id(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12]


def ticker_to_logo_filename(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    return f"{safe}.png"


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def render_logo_or_placeholder(ticker: str) -> None:
    logo_path = LOGO_DIR / ticker_to_logo_filename(ticker)

    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
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

    header_parts = [f"{idx}. {kind_label}: {title}"]
    meta_parts = [p for p in [published_at, source] if p]
    if meta_parts:
        header_parts.append(f"({', '.join(meta_parts)})")
    st.write(" ".join(header_parts).strip())

    content = desc or snippet
    if content:
        st.write(content)

    if url:
        st.write(url)

    st.markdown("---")


@st.cache_data
def generate_dummy_price_series(ticker: str) -> pd.DataFrame:
    np.random.seed(hash(ticker) % 2**32)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    prices = np.cumsum(np.random.randn(len(dates)) * 0.5 + 0.2) + 100
    return pd.DataFrame({"Tarih": dates, "Fiyat": prices})


@st.cache_data
def run_dummy_models(ticker: str) -> dict:
    np.random.seed((hash(ticker) + 1337) % 2**32)
    metrics = {
        "expected_return": float(np.round(np.random.uniform(-5, 10), 2)),
        "volatility": float(np.round(np.random.uniform(2, 8), 2)),
        "confidence": float(np.round(np.random.uniform(50, 100), 2)),
    }
    scenario = pd.DataFrame(
        {"Senaryo": ["Ayı", "Baz", "Boğa"], "Getiri (%)": np.random.uniform(-15, 25, size=3).round(2)}
    )
    return {"metrics": metrics, "scenario": scenario}


@st.cache_data
def dummy_ticker_about(ticker: str) -> str:
    np.random.seed((hash(ticker) + 2026) % 2**32)
    profiller = [
        "istikrarlı nakit akışı üreten defansif bir şirket",
        "fiyatlama gücü dinamikleri olan olgun bir marka portföyü",
        "marj hassasiyeti yüksek, dağıtım odaklı bir operasyon",
        "kur riskine açık, global ölçekte tüketiciye dönük bir yapı",
        "mevsimsellik etkileri bulunan, talep dayanıklılığı yüksek bir şirket",
    ]
    riskler = [
        "girdi maliyeti oynaklığı",
        "kur dalgalanmaları",
        "rekabetçi fiyat baskısı",
        "dağıtım kısıtları",
        "regülasyon kaynaklı gündem riski",
    ]
    katalizorler = [
        "yönlendirme güncellemeleri",
        "beklenti üstü finansal sonuçlar",
        "fiyatlama aksiyonları",
        "maliyet azaltım programları",
        "kategori büyümesinde hızlanma",
    ]

    p = np.random.choice(profiller)
    r1, r2 = np.random.choice(riskler, size=2, replace=False)
    c1, c2 = np.random.choice(katalizorler, size=2, replace=False)

    return (
        f"{ticker} (sahte profil) bu şablonda {p} olarak kurgulanmıştır.\n\n"
        f"İzlenmesi gerekenler (sahte): {r1}, {r2}.\n\n"
        f"Olası katalizörler (sahte): {c1}, {c2}.\n\n"
        "TODO: Bu alanı gerçek şirket/sektör açıklaması, temel veriler ve model yorumlarıyla değiştir."
    )


@st.cache_data(ttl=900)
def fetch_marketaux_news(selected_ticker: str, selected_label: str) -> dict:
    """
    Marketaux haberlerini çekerken gereksiz çok istek atmamak için per_req'i n'e yaklaştırır.
    (per_req küçükse, n=10 için birden fazla istek atıp kotayı daha hızlı tüketebilir.)
    """
    if not selected_ticker:
        return {"symbol": "", "industry": "", "ticker_news": [], "industry_news": []}

    n = 10  # hem Haber Bülteni hem Raporlar için beklentin
    per_req = n  # pagination'ı azaltmak için: mümkünse tek seferde n kadar çek

    return get_ticker_and_industry_news(
        selected_ticker,
        company_name=selected_label,
        country="us",
        n=n,
        per_req=per_req,
    )


def _build_system_instruction(hits: list[dict], answer_mode: str) -> str:
    base_system = (hits[0].get("system") or "").strip() if hits else ""

    mode_hint = {
        "Resmi": "Resmi ve teknik bir dil kullan. Emoji kullanma. Kısa paragraflar ve net maddelerle yaz.",
        "Özet": "Sadece özeti ver: en fazla 8 madde veya 6-10 cümle. Gereksiz açıklama yapma.",
        "Anlaşılır 🙂": "Basit ve anlaşılır Türkçe kullan. Gerekirse az sayıda emoji kullanabilirsin. Adım adım anlat.",
    }.get(answer_mode, "Net ve yapılandırılmış yaz, tekrar etme.")

    guardrails = (
        "Kurallar:\n"
        "- Yatırım tavsiyesi verme; al/sat/tut gibi kesin yönlendirme yapma.\n"
        "- Kesin getiri vaadi yok.\n"
        "- Güncel veri gerektiren noktada 'veri güncel olmayabilir' diye uyar.\n"
        "- Bilmediğin noktada uydurma; belirsizlikleri açıkça söyle.\n"
        f"- {mode_hint}\n"
    )

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


def _render_long_answer(full_text: str, preview_chars: int = RAG_PREVIEW_CHARS) -> None:
    full_text = (full_text or "").strip()
    if not full_text:
        st.markdown("Yanıt üretilemedi (boş döndü).")
        return

    if len(full_text) <= preview_chars:
        st.markdown(full_text)
        return

    preview = full_text[:preview_chars].rstrip() + "\n\n...(devamı aşağıda)..."
    st.markdown(preview)

    sid = _stable_id(full_text)

    with st.expander("Tam yanıtı göster"):
        # Tek görünüm: stabil scroll + wrap + sabit yükseklik
        st.code(full_text, language="text", wrap_lines=True, height=650)
        st.download_button(
            "Yanıtı indir (.txt)",
            data=full_text,
            file_name="finanalytics_rag_answer.txt",
            mime="text/plain",
            key=f"rag_dl_{sid}",
        )


st.set_page_config(page_title="FinAnalytics", layout="wide")
st.title("FinAnalytics Dashboard")

st.sidebar.header("Kontroller")

selected_label = st.sidebar.selectbox("Hisse seçin", [""] + list(TICKERS.keys()))
selected_ticker = TICKERS.get(selected_label, "")

email = st.sidebar.text_input(
    "E-posta (raporlar için) — Mailiniz tarafınıza düzenli olarak rapor gönderilmesi için istenmektedir.",
    key="email_input",
)

if st.sidebar.button("E-postayı Kaydet"):
    if email and is_valid_email(email):
        st.session_state["saved_email"] = email
        st.sidebar.success("E-posta kaydedildi.")
    else:
        st.sidebar.error("Geçersiz e-posta formatı.")

saved_email = st.session_state.get("saved_email", "")

if not selected_ticker:
    st.write(
        "FinAnalytics, seçilen hisse için kısa/orta/uzun vadeli model çıktıları ve "
        "haber/sektör verilerini göstermeyi hedefleyen bir Streamlit dashboard şablonudur. "
        "Dashboard bölümlerini açmak için soldan bir hisse seçin."
    )
    st.stop()

left, right = st.columns([5, 1.5], vertical_alignment="center")
with left:
    st.markdown(f"## {selected_label} ({selected_ticker})")
with right:
    render_logo_or_placeholder(selected_ticker)

tabs = st.tabs(["Hakkında", "Model Çıktıları", "Kısa Vadeli Model", "Orta Vadeli Model", "Uzun Vadeli Model", "Haber Bülteni", "RAG (SFT + Gemini)", "Raporlar"])

with tabs[0]:
    st.header("Hakkında")
    st.write(dummy_ticker_about(selected_ticker))

with tabs[2]:
    render_short_dashboard(selected_ticker)

with tabs[3]:
    render_mid_dashboard(selected_ticker)

with tabs[4]:
    render_long_dashboard(selected_ticker)

with tabs[1]:
    st.header("Model Çıktıları (Sahte)")

    results = run_dummy_models(selected_ticker)
    metrics = results["metrics"]
    scenario = results["scenario"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Beklenen Getiri", f"{metrics['expected_return']}%")
    c2.metric("Volatilite", f"{metrics['volatility']}%")
    c3.metric("Güven", f"{metrics['confidence']}%")

    df_prices = generate_dummy_price_series(selected_ticker)
    fig = px.line(df_prices, x="Tarih", y="Fiyat", title=f"{selected_ticker} Fiyat (Sahte)")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Senaryo Çıktıları")
    st.dataframe(scenario, use_container_width=True)

with tabs[5]:
    st.header("Haber Bülteni")

    use_marketaux = st.toggle("Marketaux ile gerçek haberleri çek", value=True, key="use_marketaux")

    if use_marketaux:
        try:
            with st.spinner("Marketaux haberleri çekiliyor..."):
                result = fetch_marketaux_news(selected_ticker, selected_label)

            st.caption(f"Symbol: {result['symbol']} | Industry: {result['industry']}")

            include_links = st.toggle(
                "LLM prompt'a linkleri dahil et",
                value=True,
                key="llm_include_links",
            )

            ticker_ctx, industry_ctx = build_llm_context(
                symbol=result.get("symbol", ""),
                industry=result.get("industry", ""),
                ticker_news=result.get("ticker_news", []),
                industry_news=result.get("industry_news", []),
                include_url=include_links,
                max_items=10,
                max_snippet_chars=500,
            )

            with st.expander("LLM için ham bağlamı göster", expanded=False):
                st.text_area("Şirket Bağlamı (LLM input)", ticker_ctx, height=260, key="llm_ticker_ctx")
                st.text_area("Sektör Bağlamı (LLM input)", industry_ctx, height=260, key="llm_industry_ctx")

            st.subheader("Şirket Haberleri (Son 10)")
            for i, it in enumerate(result.get("ticker_news", []), start=1):
                render_news_item(i, "Şirket Haberi", it)

            st.subheader("Sektör Haberleri (Son 10)")
            for i, it in enumerate(result.get("industry_news", []), start=1):
                render_news_item(i, "Sektör Haberi", it)

        except Exception as e:
            st.error(f"Marketaux hata: {e}")
            st.info("Kontrol: set -a && source .env && set +a (MARKETAUX_API_TOKEN yüklü mü?)")
    else:
        st.info("Gerçek haberleri görmek için toggle'ı aç.")

with tabs[6]:
    st.header("RAG (SFT + Gemini)")
    st.caption(
        "SFT dataset’ten benzer soru/cevap örnekleri çekilir (retrieval), ardından Gemini ile yanıt üretilir. "
        "Bu yapı güncel piyasa verisi sağlamaz; eğitim/araştırma amaçlıdır."
    )

    rag_left, rag_right = st.columns([3, 2], vertical_alignment="top")

    with rag_right:
        st.subheader("Ayarlar")

        top_k = st.slider(
            "Top-K retrieval örnek",
            1,
            10,
            5,
            key="rag_top_k",
            help="Soruna en yakın kaç örnek (SFT soru/cevap) çekileceğini belirler.",
        )

        answer_mode = st.selectbox(
            "Yanıt tarzı",
            ["Resmi", "Özet", "Anlaşılır 🙂"],
            index=0,
            key="rag_answer_mode",
            help="Yanıtın dili ve formatı bu seçimle yönlendirilir.",
        )

        include_news_ctx = st.toggle(
            "Haber bağlamını prompt'a ekle (varsa)",
            value=False,
            key="rag_include_news_ctx",
            help="Haber Bülteni sekmesinde oluşan şirket/sektör özetleri varsa, yanıt üretirken bağlam olarak ekler.",
        )

        dev_mode = os.getenv(RAG_DEBUG_ENV_FLAG, "").strip() == "1"
        show_hits = False
        if dev_mode:
            show_hits = st.toggle(
                "Retrieval debug göster (geliştirici)",
                value=False,
                key="rag_show_hits",
                help="Çekilen örnekleri ve skorlarını gösterir (normal kullanıcı için gerekli değil).",
            )

        if st.button(
            "Sohbeti sıfırla",
            key="rag_reset",
            help="Bu hisse için RAG konuşma geçmişini temizler.",
        ):
            st.session_state.pop(f"rag_messages_{selected_ticker}", None)
            st.rerun()

        st.markdown("---")
        st.write("Model:", "`gemini-2.5-flash-lite`")
        st.write("Seçili hisse:", f"{selected_label} ({selected_ticker})")

    with rag_left:
        session_key = f"rag_messages_{selected_ticker}"
        if session_key not in st.session_state:
            st.session_state[session_key] = [
                {
                    "role": "assistant",
                    "content": "Sorunu yaz. Yanıtlar yatırım tavsiyesi değildir.\n\nÖrnek: 'RSI ve MACD birlikte nasıl yorumlanır?'",
                }
            ]

        for m in st.session_state[session_key]:
            role = m.get("role", "assistant")
            content = m.get("content", "")
            full = m.get("full", "")

            with st.chat_message(role):
                if role == "assistant" and full and len(full) > len(content):
                    _render_long_answer(full)
                else:
                    st.markdown(content)

        user_q = st.chat_input("Sorunu yaz…")
        if user_q:
            st.session_state[session_key].append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Benzer örnekler aranıyor..."):
                        hits = retrieve_examples(user_q, k=top_k)

                    if not hits:
                        msg = "Benzer örnek bulunamadı. Soruyu biraz daha detaylandır."
                        st.warning(msg)
                        st.session_state[session_key].append({"role": "assistant", "content": msg})
                        st.stop()

                    system_instruction = _build_system_instruction(hits, answer_mode=answer_mode)
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

                    with st.spinner("Gemini yanıt üretiyor..."):
                        answer = generate_text(
                            prompt=prompt,
                            system_instruction=system_instruction,
                            model="gemini-2.5-flash-lite",
                            max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
                        )

                    full_answer = (answer or "").strip()
                    if not full_answer:
                        full_answer = "Yanıt üretilemedi (boş döndü)."

                    _render_long_answer(full_answer)
                    st.caption(f"Yanıt uzunluğu: {len(full_answer)} karakter")

                    if len(full_answer) > RAG_PREVIEW_CHARS:
                        preview = full_answer[:RAG_PREVIEW_CHARS].rstrip() + "\n\n...(devamı var)..."
                        st.session_state[session_key].append(
                            {"role": "assistant", "content": preview, "full": full_answer}
                        )
                    else:
                        st.session_state[session_key].append({"role": "assistant", "content": full_answer})

                    if show_hits:
                        with st.expander("Retrieval debug"):
                            st.write(hits)

                except Exception as e:
                    st.error(str(e))
                    st.info("Kontrol: .env içinde GEMINI_API_KEY var mı? (set -a && source .env && set +a)")

with tabs[7]:
    st.header("Raporlar")

    st.caption("Mail tanımladıysan iki farklı rapor oluşturup mailine gönderebilirsin (şu an sadece şablon/önizleme).")

    st.write("Kayıtlı e-posta:")
    st.write(saved_email if saved_email else "Kayıtlı e-posta yok.")

    if not saved_email:
        st.warning(
            "Uyarı: E-postanı kaydetmeden rapor gönderilemez. Sol menüden e-postanı tanımla ve kaydet.",
            icon="⚠️",
        )

    st.markdown("---")

    left, right = st.columns(2, vertical_alignment="top")

    # Finansal rapor (şablon)
    with left:
        st.subheader("Finansal Rapor")
        st.caption("Seçili hisse için model çıktıları/özetleri (şu an şablon).")

        send_fin = st.button(
            "Finansal Raporu Gönder (sahte)",
            disabled=not bool(saved_email),
            help="E-posta tanımlıysa rapor gönderimi tetiklenmiş gibi davranır (şimdilik sadece önizleme).",
            key="btn_send_fin_report",
        )

        if send_fin:
            # Şablon/önizleme: mevcut dummy model çıktılarından bir metin üret
            results = run_dummy_models(selected_ticker)
            metrics = results["metrics"]
            scenario = results["scenario"]

            report_text = (
                f"FİNANSAL RAPOR — {selected_label} ({selected_ticker})\n"
                f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"Özet metrikler (sahte):\n"
                f"- Beklenen Getiri: {metrics['expected_return']}%\n"
                f"- Volatilite: {metrics['volatility']}%\n"
                f"- Güven: {metrics['confidence']}%\n\n"
                f"Senaryo (sahte):\n{scenario.to_string(index=False)}\n\n"
                "Not: Bu rapor eğitim/araştırma amaçlı şablondur, yatırım tavsiyesi değildir."
            )

            st.success("Finansal rapor gönderildi (sahte). Aşağıda önizleme var.")
            with st.expander("Finansal rapor önizleme", expanded=True):
                st.code(report_text, language="text", wrap_lines=True, height=520)
                st.download_button(
                    "Finansal raporu indir (.txt)",
                    data=report_text,
                    file_name="finanalytics_finansal_rapor.txt",
                    mime="text/plain",
                    key="dl_fin_report",
                )

    # Gündem raporu (Marketaux’dan haber + 2 cümle özet)
    with right:
        st.subheader("Gündem Raporu")
        st.caption("Marketaux haberlerinden: Şirket (son 10) + Sektör (son 10), her biri 2 cümle özet.")

        send_agenda = st.button(
            "Gündem Raporunu Gönder (sahte)",
            disabled=not bool(saved_email),
            help="E-posta tanımlıysa rapor gönderimi tetiklenmiş gibi davranır (şimdilik sadece önizleme).",
            key="btn_send_agenda_report",
        )

        if send_agenda:
            try:
                with st.spinner("Marketaux haberleri çekiliyor..."):
                    result = fetch_marketaux_news(selected_ticker, selected_label)

                agenda_text = _build_agenda_report_text(result, selected_label, selected_ticker)

                st.success("Gündem raporu gönderildi (sahte). Aşağıda önizleme var.")
                with st.expander("Gündem raporu önizleme", expanded=True):
                    st.code(agenda_text, language="text", wrap_lines=True, height=520)
                    st.download_button(
                        "Gündem raporunu indir (.txt)",
                        data=agenda_text,
                        file_name="finanalytics_gundem_raporu.txt",
                        mime="text/plain",
                        key="dl_agenda_report",
                    )

            except Exception as e:
                st.error(f"Gündem raporu oluşturulamadı: {e}")
                st.info("Kontrol: .env içinde MARKETAUX_API_TOKEN yüklü mü? (set -a && source .env && set +a)")
