'''from dataclasses import dataclass
import os

@dataclass(frozen=True)
class IMIConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float = 60.0

def load_imi_config() -> IMIConfig:
    base_url = os.getenv("IMI_API_BASE_URL", "https://api.imi-services.imi.kit.edu")
    model = os.getenv("IMI_MODEL", "qwen3-coder:30b")
    api_key = os.getenv("IMI_API_KEY", "")

    if not api_key:
        raise ValueError("Missing IMI_API_KEY environment variable.")

    return IMIConfig(base_url=base_url, model=model, api_key=api_key)'''