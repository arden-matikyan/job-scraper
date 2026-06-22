"""Job extractor: raw job text -> structured record via Ollama (JSON mode).

The LLM only does the hard parsing (qualifications, summary fields...). Fields
the scraper already knows authoritatively are passed as ``hints`` and overwrite
the LLM output. ``description_full`` is always the raw text and is preserved
even when extraction fails. Never raises.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

LIST_FIELDS = (
    "locations_all",
    "required_qualifications",
    "preferred_qualifications",
)
SCALAR_FIELDS = (
    "title", "company", "location", "posted_date",
)

PROMPT_TEMPLATE = """Extract all available information from this job posting.
Return ONLY valid JSON matching this exact schema.
For any field you cannot find, use null (or [] for list fields).
Do not invent or infer information not present in the text.

Schema:
{{
  "title": "string or null",
  "company": "string or null",
  "location": "string or null (primary location)",
  "locations_all": ["all locations if multiple, else single or empty"],
  "required_qualifications": ["each Required/Minimum Qualifications item, full text"],
  "preferred_qualifications": ["each Preferred/Desired item, full text"],
  "posted_date": "as found on page or null"
}}

Rules:
- required_qualifications: each bullet/numbered item in a section whose header
  implies requirements. Recognized headers (case-insensitive, partial match ok):
  "Required", "Minimum Qualifications", "Basic Qualifications", "You Have",
  "You'll Need", "What You Bring", "Must Have", "Qualifications".
  Preserve the full text of each item.
  Include technical skills, tools, and languages found in required sections.
- preferred_qualifications: same pattern for sections implying preferred/optional
  items. Recognized headers: "Preferred", "Desired", "Nice to Have",
  "Nice If You Have", "Bonus", "Plus", "Good to Have", "What Would Be Nice".
- locations_all: if multiple locations are listed (e.g. "McLean, VA or Austin, TX"),
  capture all of them.

Job posting text:
{text}
"""


def default_record() -> dict[str, Any]:
    """A complete record with safe defaults (used as the extraction fallback)."""
    return {
        "job_id": None,
        "title": None,
        "company": None,
        "location": None,
        "locations_all": [],
        "description_full": None,
        "required_qualifications": [],
        "preferred_qualifications": [],
        "posted_date": None,
        "source_url": None,
    }


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif item not in (None, ""):
                out.append(str(item))
        return out
    return []


def _as_scalar(value: Any) -> Optional[str]:
    """Coerce an LLM value to a string scalar (it sometimes returns a list/dict)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if x not in (None, "")]
        return "; ".join(parts) if parts else None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class JobExtractor:
    def __init__(self, ollama, max_chars: int = 12000):
        self.ollama = ollama
        self.max_chars = max_chars

    def extract(
        self,
        raw_text: str,
        source_url: str,
        hints: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Return a structured record. Never raises; preserves description_full."""
        hints = hints or {}
        record = default_record()
        record["description_full"] = raw_text or None
        record["source_url"] = source_url

        text = (raw_text or "").strip()[: self.max_chars]
        if text:
            try:
                prompt = PROMPT_TEMPLATE.format(text=text)
                parsed = self.ollama.generate_json(prompt, default={})
                if isinstance(parsed, dict) and parsed:
                    self._merge_llm(record, parsed)
                else:
                    logger.warning("Extractor got empty/invalid JSON for %s", source_url)
            except Exception as exc:
                logger.warning("Extraction error for %s: %s", source_url, exc)

        self._apply_hints(record, hints)
        return record

    def _merge_llm(self, record: dict, parsed: dict) -> None:
        for key in SCALAR_FIELDS:
            if key in parsed and parsed[key] not in (None, ""):
                record[key] = _as_scalar(parsed[key])
        for key in LIST_FIELDS:
            if key in parsed:
                record[key] = _as_str_list(parsed[key])

    def _apply_hints(self, record: dict, hints: dict) -> None:
        """Scraper-known fields win over the LLM."""
        for key, value in hints.items():
            if key not in record:
                continue
            if value in (None, "", []):
                continue
            if key in LIST_FIELDS:
                record[key] = _as_str_list(value)
            else:
                record[key] = value
