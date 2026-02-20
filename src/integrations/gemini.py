from __future__ import annotations

import os
import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Güvenli token aralıkları
_MIN_TOKENS = 64
_MAX_TOKENS = 8192


@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    """Gemini API client'ını oluşturur ve cache'ler (uygulama ömrü boyunca)."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY bulunamadı. "
            "`set -a && source .env && set +a` ile .env'ı yükle."
        )
    return genai.Client(api_key=api_key)


def generate_text(
    prompt: str,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 256,
    temperature: float = 0.4,
) -> str:
    """
    Gemini API'ye prompt gönderir ve metin yanıt döndürür.

    Args:
        prompt:             Kullanıcı / RAG promptu.
        system_instruction: Sistem talimatı (guardrail, yanıt stili vb.).
        model:              Kullanılacak Gemini model adı.
        max_output_tokens:  Üretilecek maksimum token sayısı.
        temperature:        0 (deterministik) → 1 (yaratıcı) arası örnekleme sıcaklığı.

    Returns:
        Modelin metin yanıtı. Hata durumunda RuntimeError fırlatır.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt boş olamaz.")

    # Token sınırlarını güvenli aralığa kısıt
    max_output_tokens = max(_MIN_TOKENS, min(max_output_tokens, _MAX_TOKENS))
    temperature = max(0.0, min(temperature, 1.0))

    client = get_client()

    config_kwargs: dict = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_instruction and system_instruction.strip():
        config_kwargs["system_instruction"] = system_instruction.strip()

    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:
        # API hatalarını daha anlaşılır bir mesajla yeniden fırlat
        raise RuntimeError(
            f"Gemini API hatası ({type(exc).__name__}): {exc}\n"
            "GEMINI_API_KEY geçerli mi? Kota aşıldı mı?"
        ) from exc

    text = (getattr(resp, "text", None) or "").strip()

    # Yanıt boşsa ve finish_reason varsa logla
    if not text:
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            # Candidate bir nesne; finish_reason attribute olarak erişilir
            finish_reason = str(getattr(candidates[0], "finish_reason", "UNKNOWN"))
        else:
            finish_reason = "UNKNOWN"
        raise RuntimeError(
            f"Gemini boş yanıt döndürdü (finish_reason={finish_reason}). "
            "İçerik filtresi devreye girmiş olabilir."
        )

    return text
