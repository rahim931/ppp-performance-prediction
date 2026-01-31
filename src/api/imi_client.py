import time
import hashlib
from typing import Optional, List, Dict, Any

import requests

from src.api.interface import LLMClient, LLMResponse
from src.api.config import IMIConfig


class IMILLMClient(LLMClient):
    """
    Concrete LLM client for the IMI API.
    Implements the shared contract: generate() -> LLMResponse
    """

    def __init__(self, cfg: IMIConfig):
        self.cfg = cfg
        self.endpoint = f"{cfg.base_url.rstrip('/')}/generate"

    def _prompt_hash(self, prompt: str, temperature: float, max_tokens: int, stop: Optional[List[str]]) -> str:
        payload = f"{prompt}|temp={temperature}|max={max_tokens}|stop={stop}"
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
        prompt_hash = self._prompt_hash(prompt, temperature, max_tokens, stop)
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        # Payload gemäß Aufgabenbeschreibung / IMI-Endpoint
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
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

        # Basic error handling (robustness kommt in Schritt 3)
        r.raise_for_status()
        raw = r.json()

        # Versuche typische Felder zu lesen (je nach API-Format)
        text = (
            raw.get("text")
            or raw.get("response")
            or raw.get("generated_text")
            or ""
        )

        return LLMResponse(
            text=text,
            raw=raw,
            model=self.cfg.model,
            prompt_hash=prompt_hash,
            latency_s=latency,
            cached=False,
        )
