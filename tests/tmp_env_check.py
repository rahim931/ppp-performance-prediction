import os
from dotenv import load_dotenv

load_dotenv()

print("LLM_PROVIDER =", os.getenv("LLM_PROVIDER"))
print("IMI_API_BASE_URL =", os.getenv("IMI_API_BASE_URL"))
print("IMI_MODEL =", os.getenv("IMI_MODEL"))
print("IMI_API_KEY set? =", bool(os.getenv("IMI_API_KEY")))

print("DEEPSEEK_BASE_URL =", os.getenv("DEEPSEEK_BASE_URL"))
print("DEEPSEEK_MODEL =", os.getenv("DEEPSEEK_MODEL"))
print("DEEPSEEK_API_KEY set? =", bool(os.getenv("DEEPSEEK_API_KEY")))
