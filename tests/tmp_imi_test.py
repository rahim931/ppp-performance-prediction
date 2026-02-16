from dotenv import load_dotenv
load_dotenv()

from src.api.config import load_imi_config
from src.api.imi_client import IMILLMClient

cfg = load_imi_config()
llm = IMILLMClient(cfg)

resp = llm.generate("Say hello in one short sentence.", temperature=0.0, max_tokens=50)
print(resp.text)
print("latency:", resp.latency_s, "cached:", resp.cached, "model:", resp.model)
