"""Anthropic (Claude) client: drop-in alternative to OllamaClient.

Implements the same duck-typed surface the pipeline relies on — ``generate_json``,
``embed``, ``check_health``, and the ``model`` / ``embed_model`` attributes — so it
can be swapped in via :mod:`agent.llm` with no changes at the call sites.

Like OllamaClient, every public method degrades gracefully: it returns a typed
default instead of raising, so a flaky API or missing key can never crash the run.

Claude has no embeddings endpoint, so ``embed`` always returns None (``embed_model``
is None). Embeddings are unused today; when this client is active they are skipped.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agent.ollama_client import _safe_json_extract

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4096

_JSON_SYSTEM = (
    "You are a precise information-extraction engine. Respond with ONLY valid JSON "
    "matching the requested schema. No markdown fences, no prose, no explanation."
)


class AnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
    ):
        # Import lazily so the package only hard-depends on anthropic when this
        # provider is actually selected.
        import anthropic

        self.model = model
        self.embed_model = None  # Claude has no embeddings API
        self.max_tokens = max_tokens
        self._anthropic = anthropic
        # Anthropic() resolves ANTHROPIC_API_KEY from the env when api_key is None.
        # max_retries lets the SDK auto-retry 429/5xx with backoff.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def close(self) -> None:
        # The SDK manages its own connection pool; nothing to close explicitly.
        pass

    # ----------------------------------------------------------------- health
    def check_health(self) -> bool:
        """True iff an API key is present and the model is reachable."""
        if not (os.environ.get("ANTHROPIC_API_KEY") or self._client.api_key):
            logger.error("ANTHROPIC_API_KEY is not set")
            return False
        try:
            self._client.models.retrieve(self.model)
            return True
        except Exception as exc:
            logger.error("Anthropic not reachable / bad model %r: %s", self.model, exc)
            return False

    # ------------------------------------------------------------- generation
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return the model's text response, or "" if the call fails."""
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or _JSON_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception as exc:
            logger.warning("Anthropic generate failed: %s", exc)
            return ""

    def generate_json(
        self,
        prompt: str,
        default: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> Any:
        """Generate and safe-parse JSON; return ``default`` on any failure."""
        default = {} if default is None else default
        raw = self.generate(prompt, system=system)
        if not raw:
            return default
        parsed = _safe_json_extract(raw)
        if parsed is None:
            logger.warning("Anthropic JSON parse failed; raw head: %s", raw[:300])
            return default
        return parsed

    # ----------------------------------------------------------------- tool use
    def complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        """Low-level Messages API call used by the agentic ToolRunner (tool use).

        Returns the raw anthropic response (``.content`` blocks + ``.stop_reason``)
        so the runner can parse ``tool_use`` blocks. Unlike :meth:`generate_json`
        this does not swallow errors — the runner needs to see a failed turn — and
        it is only ever called by the standalone agent scripts, never the pipeline.
        Keeping the SDK call here means no anthropic SDK usage leaks outside this file.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return self._client.messages.create(**kwargs)

    # ------------------------------------------------------------- embeddings
    def embed(self, text: str) -> Optional[list[float]]:
        """Claude has no embeddings API; embeddings are skipped under this provider."""
        return None
