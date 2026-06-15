"""Thin Ollama client: text generation, JSON-mode extraction, embeddings, health.

Wraps http://localhost:11434. Every public method degrades gracefully — it returns
a typed default instead of raising, so a flaky or absent model can never crash the
pipeline. check_health() gates startup.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _safe_json_extract(raw: str) -> Optional[Any]:
    """Strip markdown fences, then json.loads; fall back to the outermost {...}."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # ----------------------------------------------------------------- health
    def available_models(self) -> list[str]:
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.error("Could not list Ollama models: %s", exc)
            return []

    def check_health(self) -> bool:
        """True iff Ollama responds and both required models are installed."""
        names = self.available_models()
        if not names:
            logger.error("Ollama not reachable at %s", self.base_url)
            return False

        def have(model: str) -> bool:
            return any(n == model or n.split(":")[0] == model for n in names)

        ok = have(self.model) and have(self.embed_model)
        if not ok:
            logger.error(
                "Ollama is missing required models. Need %r + %r; have %s",
                self.model, self.embed_model, sorted(names),
            )
        return ok

    # ------------------------------------------------------------- generation
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        format: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> str:
        """Return the model's text response, or "" if all attempts fail."""
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "") or ""
            except Exception as exc:
                logger.warning(
                    "Ollama generate failed (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        logger.error("Ollama generate gave up after %d attempts", self.max_retries)
        return ""

    def generate_json(
        self,
        prompt: str,
        default: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> Any:
        """Generate with JSON mode and safe-parse; return ``default`` on failure."""
        default = {} if default is None else default
        raw = self.generate(prompt, system=system, format="json")
        if not raw:
            return default
        parsed = _safe_json_extract(raw)
        if parsed is None:
            logger.warning("Ollama JSON parse failed; raw head: %s", raw[:300])
            return default
        return parsed

    # ------------------------------------------------------------- embeddings
    def embed(self, text: str) -> Optional[list[float]]:
        """Return an embedding vector, or None on empty input / failure."""
        if not text:
            return None
        payload = {"model": self.embed_model, "prompt": text}
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(f"{self.base_url}/api/embeddings", json=payload)
                resp.raise_for_status()
                emb = resp.json().get("embedding")
                if isinstance(emb, list) and emb:
                    return emb
                return None
            except Exception as exc:
                logger.warning(
                    "Ollama embed failed (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        return None
