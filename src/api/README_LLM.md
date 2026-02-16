# PPP LLM Clients (IMI + DeepSeek)

This project provides a small, unified wrapper to call an LLM via either **IMI (KIT-hosted)** or **DeepSeek**, with retries and a simple JSONL cache.

## Files (what they do)

- `src/api/interface.py`  
  Defines the shared contract: `LLMClient.generate(...) -> LLMResponse`.

- `src/api/errors.py`  
  Custom exception types used by both clients (`LLMAuthError`, `LLMTemporaryError`, etc.).

- `src/api/cache.py`  
  `JsonlCache`: very small file-based cache (`.jsonl`) keyed by `prompt_hash` to avoid repeated calls.

- `src/api/config.py`  
  IMI configuration loader (`load_imi_config()`): reads IMI settings from environment variables.

- `src/api/imi_client.py`  
  IMI client (`IMILLMClient`): calls the IMI endpoint, retries on temporary errors, writes/reads cache.

- `src/api/deepseek_config.py`  
  DeepSeek configuration loader (`load_deepseek_config()`): reads DeepSeek settings from environment variables.

- `src/api/deepseek_client.py`  
  DeepSeek client (`DeepSeekLLMClient`): OpenAI-style chat completions client with retries + cache.

- `src/api/factory.py`  
  `get_llm_client(...)`: returns either `IMILLMClient` or `DeepSeekLLMClient` based on `LLM_PROVIDER`.

## Environment variables

Create a `.env` in the project root:

```env
LLM_PROVIDER=imi

IMI_API_BASE_URL=https://api3.imi-services.imi.kit.edu/api/generate
IMI_MODEL=qwen2.5-coder:14b
IMI_API_KEY=PASTE_YOUR_IMI_KEY_HERE

DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_KEY=PASTE_YOUR_DEEPSEEK_KEY_HERE
```

## Quick tests

> Make sure `.env` is loaded (e.g. `from dotenv import load_dotenv; load_dotenv()` in your test scripts).

- **Check env loading**
  - `python tmp_env_check.py`

- **Test IMI directly**
  - `python tmp_imi_test.py`
  - run it twice; second run should show `cached: True`

- **Test via factory**
  - `python tmp_factory_test.py`
  - set `LLM_PROVIDER=imi` or `LLM_PROVIDER=deepseek` in `.env` to switch providers
