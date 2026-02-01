import time
import hashlib
from typing import Optional, List, Dict, Any

import requests

from src.api.interface import LLMClient, LLMResponse
from src.api.deepseek_config import DeepSeekConfig


class DeepSeekLLMClient(LLMClient):
    """
    Concrete LLM client for DeepSeek (OpenAI-compatible Chat Completions API).
    Implements the shared contract: generate() -> LLMResponse
    """

    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg
        base = cfg.base_url.rstrip("/")
        # DeepSeek uses OpenAI-compatible endpoints.
        self.endpoint = f"{base}/v1/chat/completions"

    def _prompt_hash(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        model: str,
    ) -> str:
        payload = f"{model}|{prompt}|temp={temperature}|max={max_tokens}|stop={stop}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        model = self.cfg.model
        prompt_hash = self._prompt_hash(prompt, temperature, max_tokens, stop, model)

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        # OpenAI-style messages format
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        t0 = time.time()
        r = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        latency = time.time() - t0

        r.raise_for_status()
        raw = r.json()

        # Typical OpenAI-compatible response parsing
        text = ""
        try:
            text = raw["choices"][0]["message"]["content"]
        except Exception:
            # Fallbacks if format differs
            text = raw.get("text") or raw.get("response") or ""

        return LLMResponse(
            text=text,
            raw=raw,
            model=model,
            prompt_hash=prompt_hash,
            latency_s=latency,
            cached=False,
        )