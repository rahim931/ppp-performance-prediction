from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: Dict[str, Any]
    model: str
    prompt_hash: str
    latency_s: float
    cached: bool


class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        ...
