from dataclasses import dataclass
import os

@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float = 60.0

def load_deepseek_config() -> DeepSeekConfig:
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")

    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY environment variable.")

    return DeepSeekConfig(base_url=base_url, model=model, api_key=api_key)