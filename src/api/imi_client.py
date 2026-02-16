import time
import hashlib
import random
import re
from typing import Optional, List, Dict, Any

import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError

from src.api.interface import LLMClient, LLMResponse
from src.api.config import IMIConfig
from src.api.cache import JsonlCache
from src.api.errors import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTemporaryError,
    LLMBadRequestError,
)


class IMILLMClient(LLMClient):
    """
    Client for IMI-hosted local model.

    Request format per reference:
    {
      "prompt": "...",
      "model": "...",
      "stream": false,
      "options": {"num_ctx": 100000},
      "temperature": 0.2
    }

    Response key: "response"
    :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        cfg: IMIConfig,
        *,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        cache_path: str = "artifacts/cache/imi_cache.jsonl",
        num_ctx: int = 100000,
    ):
        self.cfg = cfg
        self.endpoint = cfg.base_url.rstrip("/")  # URL is already the full endpoint
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.num_ctx = num_ctx
        self.cache = JsonlCache(cache_path)

    def _prompt_hash(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        model: str,
    ) -> str:
        stop_norm = tuple(stop) if stop else tuple()
        key = (
            f"provider=imi\n"
            f"model={model}\n"
            f"temperature={temperature}\n"
            f"max_tokens={max_tokens}\n"
            f"stop={stop_norm}\n"
            f"num_ctx={self.num_ctx}\n"
            f"prompt={prompt}"
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _backoff(self, attempt: int) -> float:
        exp = self.backoff_base_s * (2 ** (attempt - 1))
        jitter = random.uniform(0, 0.3 * exp)
        return exp + jitter

    def _strip_think(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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

        cached = self.cache.get(prompt_hash)
        if cached is not None:
            return cached

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "prompt": prompt.strip(),
            "model": model,
            "stream": False,
            "options": {"num_ctx": self.num_ctx},
            "temperature": temperature,
        }

        attempt = 0
        last_error: Exception | None = None

        while attempt < self.max_retries:
            attempt += 1
            try:
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

                text = raw.get("response") or raw.get("text") or raw.get("generated_text") or ""
                text = self._strip_think(text)

                resp = LLMResponse(
                    text=text,
                    raw=raw,
                    model=model,
                    prompt_hash=prompt_hash,
                    latency_s=latency,
                    cached=False,
                )
                self.cache.put(resp)
                return resp

            except HTTPError as e:
                status = getattr(e.response, "status_code", None)

                if status in (400, 422):
                    raise LLMBadRequestError(f"IMI bad request ({status}).") from e
                if status in (401, 403):
                    raise LLMAuthError(f"IMI auth failed ({status}).") from e

                if status == 429:
                    last_error = LLMRateLimitError("IMI rate limit (429).")
                elif status is not None and 500 <= status <= 599:
                    last_error = LLMTemporaryError(f"IMI server error ({status}).")
                else:
                    last_error = LLMTemporaryError(f"IMI HTTP error ({status}).")

            except (Timeout, ConnectionError):
                last_error = LLMTemporaryError("Network timeout or connection error.")

            if attempt < self.max_retries:
                wait_s = self._backoff(attempt)
                print(
                    f"[IMILLMClient] retry {attempt}/{self.max_retries} "
                    f"in {wait_s:.2f}s due to {type(last_error).__name__}"
                )
                time.sleep(wait_s)
            else:
                assert last_error is not None
                raise last_error
