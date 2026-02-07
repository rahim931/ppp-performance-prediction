from src.api.factory import get_llm_client

llm = get_llm_client()
resp = llm.generate("Say hello again.", temperature=0.0, max_tokens=30)
print(resp.text, resp.cached)
