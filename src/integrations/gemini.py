from __future__ import annotations

import os
import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash-lite"

@st.cache_resource(show_spinner=False)
def get_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı. .env dosyanı kontrol et.")
    # Gemini Developer API
    return genai.Client(api_key=api_key)

def generate_text(
    prompt: str,
    system_instruction: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 500,
) -> str:
    client = get_client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_output_tokens,
            # İstersen: stop_sequences=["\n\n---"] gibi ekleyebilirsin
        ),
    )
    return (getattr(resp, "text", None) or "").strip()
