from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any

from src.api.interface import LLMResponse


class JsonlCache:
    """
    Very simple JSONL cache.
    - Key: prompt_hash
    - Value: serialized LLMResponse

    Stored as one JSON object per line, so it's easy to inspect and append.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory index: prompt_hash -> file offset not needed; we just store full objects
        self._index: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            self._loaded = True
            return

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("prompt_hash")
                if key:
                    self._index[key] = obj

        self._loaded = True

    def get(self, prompt_hash: str) -> Optional[LLMResponse]:
        self._load()
        obj = self._index.get(prompt_hash)
        if not obj:
            return None

        # Reconstruct LLMResponse; mark cached=True
        return LLMResponse(
            text=obj["text"],
            raw=obj.get("raw", {}),
            model=obj.get("model", ""),
            prompt_hash=obj["prompt_hash"],
            latency_s=float(obj.get("latency_s", 0.0)),
            cached=True,
        )

    def put(self, resp: LLMResponse) -> None:
        self._load()

        obj = {
            "text": resp.text,
            "raw": resp.raw,
            "model": resp.model,
            "prompt_hash": resp.prompt_hash,
            "latency_s": resp.latency_s,
        }

        # Update in-memory
        self._index[resp.prompt_hash] = obj

        # Append to file (JSONL)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
