from src.api.deepseek_config import load_deepseek_config
from src.api.deepseek_client import DeepSeekLLMClient


def get_llm_client(
    *,
    max_retries: int = 5,
    backoff_base_s: float = 1.0,
    cache_path: str = "artifacts/cache/deepseek_cache.jsonl",
) -> DeepSeekLLMClient:
    """
    Factory function to create a ready-to-use LLM client.

    Person B should use only this function.
    It hides provider/config details and ensures consistent settings.
    """
    cfg = load_deepseek_config()
    return DeepSeekLLMClient(
        cfg,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        cache_path=cache_path,
    )
