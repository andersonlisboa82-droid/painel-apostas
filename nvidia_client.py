from __future__ import annotations

import os
import time

import requests


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")


def request_nvidia_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.35,
    top_p: float = 0.9,
    max_tokens: int = 1400,
) -> str:
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Defina a variavel de ambiente NVIDIA_API_KEY para usar a analise com IA.")

    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.post(
                NVIDIA_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.ReadTimeout as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2.0)
                continue
            raise TimeoutError("A NVIDIA demorou para responder. Tente novamente em alguns segundos.") from exc
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            if exc.response is not None and exc.response.status_code in (429, 500, 502, 503, 504):
                if attempt == 0:
                    time.sleep(3.0)
                    continue
            raise
        except Exception as exc:
            last_error = exc
            raise
    else:
        raise last_error if last_error else RuntimeError("Falha desconhecida ao consultar a NVIDIA.")

    choice = data["choices"][0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
        if parts:
            return "\n".join(parts)

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        cleaned_reasoning = reasoning.strip()
        markers = [
            "Estrutura da resposta:",
            "Restrições:",
            "Responda em no maximo",
            "Preciso explicar:",
            "Dados fornecidos:",
        ]
        if any(marker in cleaned_reasoning for marker in markers):
            raise ValueError(
                "A NVIDIA retornou apenas rascunho interno, sem resposta final. "
                "Tente novamente ou troque o modelo."
            )
        return cleaned_reasoning

    raise ValueError(f"Resposta da NVIDIA sem texto utilizavel: {data}")
