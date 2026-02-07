from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Union


@dataclass
class LLMResultRow:
    # ---- identification ----
    heuristic_id: str
    raw_app_type: str               # bin_greedy, cvrp_lns, ...
    strategy: Optional[str]         # e1 / e2 / m1

    # ---- ground truth ----
    objective: Optional[float]      # known objective (None for CAP rows)

    # ---- LLM output ----
    llm_value: Optional[Union[str, float]]
    llm_confidence: Optional[float]

    # ---- parsing quality ----
    parse_ok: bool
    parse_error: Optional[str]

    # ---- LLM metadata ----
    model: str
    prompt_hash: str
    latency_s: float
    cached: bool

    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def columns() -> List[str]:
        return [
            "heuristic_id",
            "raw_app_type",
            "strategy",
            "objective",
            "llm_value",
            "llm_confidence",
            "parse_ok",
            "parse_error",
            "model",
            "prompt_hash",
            "latency_s",
            "cached",
            "raw_text",
        ]
