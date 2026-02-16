import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.integrations.marketaux import get_ticker_and_industry_news
from src.reports.news_prompt import build_llm_context

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


@st.cache_data(ttl=600)
def fetch_marketaux_news(selected_ticker: str, selected_label: str) -> dict:
    return get_ticker_and_industry_news(
        selected_ticker,
        company_name=selected_label,
        country="us",
        n=10,
        per_req=3,
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

tabs = st.tabs(["Hakkında", "Model Çıktıları", "Haber Bülteni", "Raporlar"])

with tabs[0]:
    st.header("Hakkında")
    st.write(dummy_ticker_about(selected_ticker))

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

with tabs[2]:
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

with tabs[3]:
    st.header("Rapor Yönetimi")

    st.write("Kayıtlı e-posta:")
    st.write(saved_email if saved_email else "Kayıtlı e-posta yok.")

    if st.button("Test Raporu Gönder"):
        st.success("Test raporu gönderildi (sahte).")

    st.write("Rapor planla (sahte):")
    st.date_input("Rapor Tarihi")
