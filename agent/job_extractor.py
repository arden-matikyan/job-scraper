"""Job extractor: raw job text -> structured record via Ollama (JSON mode).

The LLM only does the hard parsing (summary, qualifications, skills, verbatim
experience/education...). Fields the scraper already knows authoritatively are
passed as ``hints`` and overwrite the LLM output. ``description_full`` is always
the raw text and is preserved even when extraction fails. Never raises.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Fields the LLM is asked to fill (system-set fields like job_id/source_url/
# description_full/scraper_key/scraped_at/embedding are added elsewhere).
LIST_FIELDS = (
    "locations_all",
    "required_qualifications",
    "preferred_qualifications",
    "required_skills",
    "preferred_skills",
)
SCALAR_FIELDS = (
    "title", "company", "location", "remote_type", "employment_type",
    "description_summary", "experience_raw", "education_raw", "salary_raw",
    "posted_date",
)

_REMOTE_VALUES = {"remote", "hybrid", "onsite", "unspecified"}
_REMOTE_MAP = {
    "remote": "remote", "fully remote": "remote", "telework": "remote",
    "hybrid": "hybrid", "flexible": "hybrid",
    "onsite": "onsite", "on-site": "onsite", "on site": "onsite", "in office": "onsite",
    "in-office": "onsite", "in-person": "onsite",
}
_EMPLOYMENT_VALUES = {"full_time", "part_time", "contract", "internship", "unspecified"}
_EMPLOYMENT_MAP = {
    "full_time": "full_time", "full-time": "full_time", "full time": "full_time",
    "part_time": "part_time", "part-time": "part_time", "part time": "part_time",
    "contract": "contract", "contractor": "contract", "temporary": "contract",
    "internship": "internship", "intern": "internship",
}

PROMPT_TEMPLATE = """Extract all available information from this job posting.
Return ONLY valid JSON matching this exact schema.
For any field you cannot find, use null (or [] for list fields).
Do not invent or infer information not present in the text.
Do not normalize experience or education — copy verbatim phrases.

Schema:
{{
  "title": "string or null",
  "company": "string or null",
  "location": "string or null (primary location)",
  "locations_all": ["all locations if multiple, else single or empty"],
  "remote_type": "one of: remote, hybrid, onsite, unspecified",
  "employment_type": "one of: full_time, part_time, contract, internship, unspecified",
  "description_summary": "3 sentences in your own words: role, team context, key requirements",
  "required_qualifications": ["each Required/Minimum Qualifications item, full text"],
  "preferred_qualifications": ["each Preferred/Desired item, full text"],
  "required_skills": ["explicitly required technical skills, tools, languages, frameworks"],
  "preferred_skills": ["explicitly preferred technical skills"],
  "experience_raw": "verbatim experience phrase, e.g. '5+ years' or null",
  "education_raw": "verbatim education requirement or null",
  "salary_raw": "verbatim salary if present or null",
  "posted_date": "as found on page or null"
}}

Rules:
- required_qualifications: each bullet/numbered item in a "Required" or "Minimum
  Qualifications" section is one list entry. Preserve the full text of each item.
- preferred_qualifications: same pattern for "Preferred" or "Desired" sections.
- required_skills: extract only explicitly listed technical skills, tools,
  languages, frameworks. Do not infer from the job title.
- locations_all: if multiple locations are listed (e.g. "McLean, VA or Austin, TX"),
  capture all of them.
- experience_raw: copy the exact phrase as written, e.g. "5+ years",
  "3-5 years of experience", "Minimum 2 years". Do not convert to a number.
- description_summary: write exactly 3 sentences summarizing the role, team
  context, and key requirements.

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
        "remote_type": "unspecified",
        "employment_type": "unspecified",
        "description_full": None,
        "description_summary": None,
        "required_qualifications": [],
        "preferred_qualifications": [],
        "required_skills": [],
        "preferred_skills": [],
        "experience_raw": None,
        "education_raw": None,
        "salary_raw": None,
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


def _norm_enum(value: Any, mapping: dict, allowed: set, default: str) -> str:
    if not isinstance(value, str):
        return default
    key = value.strip().lower()
    if key in allowed:
        return key
    return mapping.get(key, default)


class JobExtractor:
    def __init__(self, ollama, max_chars: int = 5000):
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
            except Exception as exc:  # belt-and-suspenders; client already guards
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
        record["remote_type"] = _norm_enum(
            record.get("remote_type"), _REMOTE_MAP, _REMOTE_VALUES, "unspecified"
        )
        record["employment_type"] = _norm_enum(
            record.get("employment_type"), _EMPLOYMENT_MAP, _EMPLOYMENT_VALUES, "unspecified"
        )

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
