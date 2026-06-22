"""LLM provider switch.

Pick which LLM backs the pipeline (extraction, recon reasoning, filtering) by
editing ``LLM_PROVIDER`` below, or override it at runtime with the ``LLM_PROVIDER``
environment variable. Both providers expose the same duck-typed surface
(``generate_json``, ``embed``, ``check_health``, ``model``, ``embed_model``), so
nothing at the call sites changes.

  - "claude"  -> AnthropicClient (Claude Haiku via the anthropic SDK; needs
                 ANTHROPIC_API_KEY). No embeddings.
  - "ollama"  -> OllamaClient (local llama3.2 + nomic-embed-text at :11434).
"""
from __future__ import annotations

import os
from typing import Any, Optional, Protocol, runtime_checkable

# ---- pick your LLM here ----------------------------------------------------
LLM_PROVIDER = "ollama"  # "claude" | "ollama"
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """The interface the pipeline depends on. Both clients satisfy it."""

    model: str
    embed_model: Optional[str]

    def generate_json(self, prompt: str, default: Optional[Any] = ..., system: Optional[str] = ...) -> Any: ...
    def embed(self, text: str) -> Optional[list[float]]: ...
    def check_health(self) -> bool: ...


def resolve_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or LLM_PROVIDER).strip().lower()


def get_llm_client() -> LLMClient:
    """Build the configured LLM client."""
    provider = resolve_provider()
    if provider == "claude":
        from agent.anthropic_client import AnthropicClient
        return AnthropicClient()
    if provider == "ollama":
        from agent.ollama_client import OllamaClient
        return OllamaClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}; expected 'claude' or 'ollama'."
    )
