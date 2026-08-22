"""Anthropic API wrapper. Needs ANTHROPIC_API_KEY set in the environment."""

import os
from anthropic import Anthropic

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_client = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = Anthropic(api_key=key)
    return _client


def run_prompt(system_prompt: str, user_input: str, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> str:
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
