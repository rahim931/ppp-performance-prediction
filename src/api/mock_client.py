# src/api/mock_client.py
import time
import hashlib
from typing import Optional, Dict, Any, List

from src.api.interface import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """
    Minimal mock client for local testing.

    This mock exists ONLY to unblock runner and parser development.
    It must not mirror any real API behaviour.
    """

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        start = time.time()

        text = (
            "LLM prediction result:\n"
            "{\n"
            '  "prediction": 0.42,\n'
            '  "confidence": 0.85\n'
            "}\n"
        )

        latency = time.time() - start
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        return LLMResponse(
            text=text,
            raw={"mock": True},
            model="mock",
            prompt_hash=prompt_hash,
            latency_s=latency,
            cached=False,
        )
