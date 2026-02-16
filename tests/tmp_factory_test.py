from dotenv import load_dotenv
load_dotenv()

from src.api.factory import get_llm_client
import time

llm = get_llm_client()
print("client:", type(llm).__name__)

prompt = "Factory cache test " + time.strftime("%Y-%m-%d %H:%M:%S")

resp1 = llm.generate(prompt, temperature=0.0, max_tokens=30)
resp2 = llm.generate(prompt, temperature=0.0, max_tokens=30)

print("1:", resp1.text, "cached:", resp1.cached)
print("2:", resp2.text, "cached:", resp2.cached)
