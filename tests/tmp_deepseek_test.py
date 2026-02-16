from dotenv import load_dotenv
load_dotenv()

from src.api.deepseek_config import load_deepseek_config
from src.api.deepseek_client import DeepSeekLLMClient

cfg = load_deepseek_config()
llm = DeepSeekLLMClient(cfg)

resp = llm.generate("Say hello in one short sentence.", temperature=0.0, max_tokens=50)
print(resp.text)
print("latency:", resp.latency_s, "cached:", resp.cached, "model:", resp.model)
