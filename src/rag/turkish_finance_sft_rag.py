from __future__ import annotations

import re
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_NAME = "AlicanKiraz0/Turkish-Finance-SFT-Dataset"

# Skor eşiği: bu değerin altındaki örnekler relevansız sayılır
MIN_SCORE_THRESHOLD = 0.05

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Dataset içindeki <think> bloklarını temizler."""
    return _THINK_RE.sub("", text or "").strip()


@st.cache_data(show_spinner=False)
def load_sft_df() -> pd.DataFrame:
    """HuggingFace'ten dataset yükler, temizler ve cache'ler."""
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()

    for col in ("system", "user", "assistant"):
        if col not in df.columns:
            raise ValueError(
                f"Dataset kolonları beklenenden farklı: {df.columns.tolist()}"
            )

    df = df.fillna("")
    df["assistant_clean"] = df["assistant"].map(_strip_think)

    # Çok kısa ya da boş asistan cevaplarını filtrele (gürültüyü azaltır)
    df = df[df["assistant_clean"].str.len() > 30].reset_index(drop=True)

    return df


@st.cache_resource(show_spinner=False)
def _build_tfidf_index() -> tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """
    TF-IDF vektörizer ve corpus matrisini oluşturur.
    cache_resource kullanılır: scipy sparse matrix ve TfidfVectorizer
    serialize edilemez veya maliyetlidir; cache_resource bunları belleğe
    bir kez yükler ve referans olarak paylaşır.
    Hem soru hem temizlenmiş cevap üzerinden retrieval yapılır.
    """
    df = load_sft_df()
    corpus = (
        df["user"].astype(str) + "\n" + df["assistant_clean"].astype(str)
    ).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=80_000,
        sublinear_tf=True,      # TF'yi log-scale'e çekerek sık kelimelerin ağırlığını dengeler
        min_df=2,               # en az 2 dokümanda geçen terimleri al (gürültü azalır)
    )
    X = vectorizer.fit_transform(corpus)

    # ndarray olarak döndürmeye gerek yok — sparse matrix cache_data ile çalışır
    return df, vectorizer, X


def retrieve_examples(query: str, k: int = 5) -> List[Dict]:
    """
    Verilen sorgu için SFT dataset'inden en benzer k örneği döndürür.

    - Skor eşiği (MIN_SCORE_THRESHOLD) altındaki sonuçlar elenir.
    - Tüm k slot doldurulamazsa daha az sonuç dönebilir.
    - Sıfır eşleşme durumunda boş liste döner (App.py bunu handle eder).
    """
    df, vectorizer, X = _build_tfidf_index()

    try:
        q_vec = vectorizer.transform([query])
    except Exception:
        return []

    sims = cosine_similarity(q_vec, X).ravel()

    # Eşik üstü indexleri al, skora göre sırala
    above_threshold = np.where(sims >= MIN_SCORE_THRESHOLD)[0]
    if len(above_threshold) == 0:
        return []

    # En iyi k tanesini al
    top_idx = above_threshold[np.argsort(-sims[above_threshold])][:k]

    results: List[Dict] = []
    for i in top_idx:
        row = df.iloc[int(i)]
        results.append(
            {
                "score": float(sims[int(i)]),
                "system": str(row["system"]),
                "user": str(row["user"]),
                "assistant": str(row["assistant_clean"]),
            }
        )

    return results
