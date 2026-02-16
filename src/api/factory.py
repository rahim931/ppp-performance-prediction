import os

from src.api.deepseek_config import load_deepseek_config
from src.api.deepseek_client import DeepSeekLLMClient

from src.api.config import load_imi_config
from src.api.imi_client import IMILLMClient


def get_llm_client(
    *,
    provider: str | None = None,
    max_retries: int = 5,
    backoff_base_s: float = 1.0,
    cache_path: str | None = None,
):
    """
    Provider selection:
    - explicit arg `provider`
    - else env var LLM_PROVIDER (default: "imi")

    Supported:
    - "imi"
    - "deepseek"
    """
    prov = (provider or os.getenv("LLM_PROVIDER", "imi")).strip().lower()

    if prov in ("imi", "kit", "local"):
        cfg = load_imi_config()
        return IMILLMClient(
            cfg,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            cache_path=cache_path or "artifacts/cache/imi_cache.jsonl",
        )

    if prov in ("deepseek", "ds"):
        cfg = load_deepseek_config()
        return DeepSeekLLMClient(
            cfg,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            cache_path=cache_path or "artifacts/cache/deepseek_cache.jsonl",
        )

    raise ValueError(f"Unknown LLM provider: {prov}. Use 'imi' or 'deepseek'.")
