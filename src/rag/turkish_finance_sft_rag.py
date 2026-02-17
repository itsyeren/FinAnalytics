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

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def _strip_think(text: str) -> str:
    # dataset içinde <think> blokları var; RAG örneğine taşımamak daha temiz
    return _THINK_RE.sub("", text or "").strip()

@st.cache_data(show_spinner=False)
def load_sft_df() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME, split="train")  # HF cache kullanır
    df = ds.to_pandas()
    # beklediğimiz kolonlar
    for col in ("system", "user", "assistant"):
        if col not in df.columns:
            raise ValueError(f"Dataset kolonları beklenenden farklı: {df.columns.tolist()}")
    df = df.fillna("")
    df["assistant_clean"] = df["assistant"].map(_strip_think)
    return df

@st.cache_resource(show_spinner=False)
def get_tfidf_index():
    df = load_sft_df()
    # Retrieval’i hem soru hem cevap üzerinden yapıyoruz
    corpus = (df["user"].astype(str) + "\n" + df["assistant_clean"].astype(str)).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=80000)
    X = vectorizer.fit_transform(corpus)
    return df, vectorizer, X

def retrieve_examples(query: str, k: int = 5) -> List[Dict]:
    df, vectorizer, X = get_tfidf_index()
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top_idx = np.argsort(-sims)[:k]

    out: List[Dict] = []
    for i in top_idx:
        row = df.iloc[int(i)]
        out.append(
            {
                "score": float(sims[int(i)]),
                "system": row["system"],
                "user": row["user"],
                "assistant": row["assistant_clean"],
            }
        )
    return out
