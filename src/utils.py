import os
from langchain_openai import ChatOpenAI
from datetime import datetime

# User agents for web scraping
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
]

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _get_llm_config():
    """Resolve LLM provider config from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    model = os.getenv("LLM_MODEL", "")

    if provider == "grok":
        api_key = os.getenv("XAI_API_KEY", "")
        base_url = "https://api.x.ai/v1"
        if not model:
            model = "grok-beta"
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        base_url = "https://api.groq.com/openai/v1"
        if not model:
            model = "llama-3.1-70b-versatile"
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        if not model:
            model = "gpt-4.1-mini"
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = "https://api.openai.com/v1"
        if not model:
            model = "gpt-4o-mini"
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        base_url = "https://api.anthropic.com/v1"
        if not model:
            model = "claude-3-haiku-20240307"
    else:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        if not model:
            model = "gpt-4.1-mini"

    return provider, api_key, base_url, model


async def ainvoke_llm(
    model,
    system_prompt,
    user_message,
    response_format=None,
    temperature=0.1
):
    provider, api_key, base_url, default_model = _get_llm_config()
    model_name = model or default_model

    if not api_key:
        raise ValueError(
            f"No API key found for provider '{provider}'. "
            f"Set the appropriate key in .env (e.g. XAI_API_KEY for grok, GROQ_API_KEY for groq)."
        )

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    if provider == "groq":
        # Groq (esp. qwen models) does not reliably support tool-based
        # structured output (with_structured_output) -> 400 "Failed to generate JSON".
        # Use plain text completion with JSON instructions instead, then parse.
        json_instruction = "\n\nIMPORTANT: Respond with ONLY valid JSON, no markdown fences, no extra text."
        messages[0]["content"] = system_prompt + json_instruction
        response = await llm.ainvoke(messages)
        content_text = response.content if hasattr(response, "content") else str(response)
        if response_format:
            import json as _json
            try:
                return _json.loads(content_text)
            except (_json.JSONDecodeError, ValueError):
                # Try to extract JSON from within the text
                import re
                m = re.search(r"\{.*\}", content_text, re.DOTALL)
                if m:
                    return _json.loads(m.group(0))
                raise ValueError(f"Could not parse JSON from Groq response: {content_text[:200]}")
        return content_text

    if response_format:
        llm = llm.with_structured_output(response_format)

    response = await llm.ainvoke(messages)
    return response if response_format else response.content
