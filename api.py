"""
Boundary Test -- api.py
Единый клиент OpenRouter для всех моделей.
"""

import os
import time
import logging
from openai import OpenAI

from config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

log = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def get_client() -> OpenAI:
    """Возвращает OpenAI-совместимый клиент для OpenRouter."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY не установлен. "
            "export OPENROUTER_API_KEY=your_key_here"
        )
    return OpenAI(base_url=OPENROUTER_BASE, api_key=key)


def call_model(
    model_id: str,
    system_prompt: str | None,
    user_message: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, int]:
    """
    Вызов модели через OpenRouter.

    Returns:
        (response_text, latency_ms)
    """
    client = get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    t0 = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = int((time.time() - t0) * 1000)

    text = response.choices[0].message.content or ""
    return text, latency_ms
