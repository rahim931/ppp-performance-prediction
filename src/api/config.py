from dataclasses import dataclass
import os


@dataclass(frozen=True)
class IMIConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float = 60.0


def load_imi_config() -> IMIConfig:
    base_url = os.getenv(
        "IMI_API_BASE_URL",
        "https://api3.imi-services.imi.kit.edu/api/generate",
    )
    model = os.getenv("IMI_MODEL", "qwen2.5-coder:14b")
    api_key = os.getenv("IMI_API_KEY", "").strip()  # optional

    return IMIConfig(base_url=base_url, model=model, api_key=api_key)

