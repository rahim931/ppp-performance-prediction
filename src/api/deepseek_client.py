# src/api/deepseek_client.py

import time
import hashlib
import random
from typing import Optional, List, Dict, Any

import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError

from src.api.interface import LLMClient, LLMResponse
from src.api.deepseek_config import DeepSeekConfig
from src.api.cache import JsonlCache
from src.api.errors import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTemporaryError,
    LLMBadRequestError,
)


class DeepSeekLLMClient(LLMClient):
    """
    Concrete LLM client for DeepSeek (OpenAI-compatible Chat Completions API).

    Features:
    - Standardized LLMClient interface
    - Retry with exponential backoff + jitter
    - File-based caching to avoid duplicate calls
    """

    def __init__(
        self,
        cfg: DeepSeekConfig,
        *,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        cache_path: str = "artifacts/cache/deepseek_cache.jsonl",
        enable_cache: bool = True,
    ):
        self.cfg = cfg
        base = cfg.base_url.rstrip("/")
        self.endpoint = f"{base}/v1/chat/completions"

        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        
        self.enable_cache = enable_cache
        self.cache = JsonlCache(cache_path) if enable_cache else None

    def _prompt_hash(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        model: str,
    ) -> str:
        """
        Create a deterministic hash for caching.
        Includes all parameters that influence the model output.
        """
        stop_norm = tuple(stop) if stop else tuple()

        key = (
            f"model={model}\n"
            f"temperature={temperature}\n"
            f"max_tokens={max_tokens}\n"
            f"stop={stop_norm}\n"
            f"prompt={prompt}"
        )

        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _backoff(self, attempt: int) -> float:
        """
        Exponential backoff with jitter.
        attempt: 1..max_retries
        """
        exp = self.backoff_base_s * (2 ** (attempt - 1))
        jitter = random.uniform(0, 0.3 * exp)
        return exp + jitter

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
        prompt_hash = self._prompt_hash(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            model=model,
        )

        # 1) Cache check
        if self.cache is not None:
            cached_resp = self.cache.get(prompt_hash)
            if cached_resp is not None:
                return cached_resp

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

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

        attempt = 0
        last_error: Exception | None = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                t0 = time.time()
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout_s,
                )
                latency = time.time() - t0

                response.raise_for_status()
                raw = response.json()

                try:
                    text = raw["choices"][0]["message"]["content"]
                except Exception:
                    text = raw.get("text") or raw.get("response") or ""

                resp = LLMResponse(
                    text=text,
                    raw=raw,
                    model=model,
                    prompt_hash=prompt_hash,
                    latency_s=latency,
                    cached=False,
                )

                # 2) Store in cache
                if self.cache is not None:
                    self.cache.put(resp)

                return resp

            except HTTPError as e:
                status = getattr(e.response, "status_code", None)

                if status in (400, 422):
                    raise LLMBadRequestError(f"DeepSeek bad request ({status}).") from e
                if status in (401, 403):
                    raise LLMAuthError(f"DeepSeek auth failed ({status}).") from e
                if status == 402:
                    raise LLMAuthError(
                        "DeepSeek payment required (402). Check credits/billing."
                    ) from e

                if status == 429:
                    last_error = LLMRateLimitError("DeepSeek rate limit (429).")
                elif status is not None and 500 <= status <= 599:
                    last_error = LLMTemporaryError(
                        f"DeepSeek server error ({status})."
                    )
                else:
                    last_error = LLMTemporaryError(
                        f"DeepSeek HTTP error ({status})."
                    )

            except (Timeout, ConnectionError):
                last_error = LLMTemporaryError(
                    "Network timeout or connection error."
                )

            if attempt < self.max_retries:
                wait_s = self._backoff(attempt)
                print(
                    f"[DeepSeekLLMClient] retry {attempt}/{self.max_retries} "
                    f"in {wait_s:.2f}s due to {type(last_error).__name__}"
                )
                time.sleep(wait_s)
            else:
                assert last_error is not None
                raise last_error
