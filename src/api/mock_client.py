# src/api/mock_client.py
import time
import hashlib
from typing import Optional, Dict, Any, List

from src.api.interface import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """
    Minimal mock client for local testing.

    This mock exists ONLY to unblock runner and parser development.
    It should return deterministic, parser-friendly JSON outputs for:
      - CAP stage: {"core_idea": "..."}
      - PPP stage: {"prediction": <float>, "confidence": <float>, "justification": "..."}
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
        meta = meta or {}
        stage = meta.get("stage", "ppp")  # default to ppp-style output

        # Deterministic prompt hash (also used for reproducibility/caching later)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if stage == "cap":
            # CAP output must contain core_idea
            text = (
                "CAP result:\n"
                "{\n"
                '  "core_idea": "This heuristic scores candidate actions using remaining capacity and penalty/bonus terms to prioritize placements that reduce waste and balance utilization."\n'
                "}\n"
            )
        else:
            # PPP output must contain prediction + confidence (+ optional justification)
            # NOTE: In your project: lower is better for both values.
            text = (
                "PPP prediction result:\n"
                "{\n"
                '  "prediction": 0.42,\n'
                '  "confidence": 0.15,\n'
                '  "justification": "The target core idea resembles the better reference heuristic more than the weaker one, so the predicted objective is closer to the low objective range."\n'
                "}\n"
            )

        latency = time.time() - start

        return LLMResponse(
            text=text,
            raw={"mock": True, "stage": stage},
            model="mock",
            prompt_hash=prompt_hash,
            latency_s=latency,
            cached=False,
        )
