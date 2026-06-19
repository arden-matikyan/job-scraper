"""Tests for the job extractor and the Ollama JSON safe-parse.

A mock Ollama client returns canned responses so these run without a live model.
Covers: full-schema extraction, list coercion, hint override, and the
malformed-output fallback that must preserve description_full.
"""
from __future__ import annotations

from agent.job_extractor import JobExtractor, default_record
from agent.ollama_client import _safe_json_extract

SAMPLE_TEXT = (
    "Senior Backend Engineer at Acme.\n"
    "Minimum Qualifications:\n- 5+ years of experience\n- Bachelor's degree in CS\n"
    "Preferred: AWS, Kubernetes\n"
)

VALID_RESPONSE = {
    "title": "Senior Backend Engineer",
    "company": "Acme",
    "location": "McLean, VA",
    "locations_all": ["McLean, VA", "Austin, TX"],
    "required_qualifications": ["5+ years of experience", "Bachelor's degree in CS"],
    "preferred_qualifications": ["AWS experience"],
    "posted_date": "2026-01-01",
}


class MockOllama:
    def __init__(self, response):
        self.response = response

    def generate_json(self, prompt, default=None, system=None):
        return self.response if self.response is not None else (default or {})

    def embed(self, text):
        return [0.1, 0.2, 0.3]


def _extract(response, text=SAMPLE_TEXT, url="https://x/jobs/1", hints=None):
    return JobExtractor(MockOllama(response)).extract(text, url, hints=hints or {})


# --------------------------------------------------------------------------- #
def test_all_schema_keys_present():
    rec = _extract(VALID_RESPONSE)
    for key in default_record():
        assert key in rec, f"missing key {key}"


def test_full_extraction_values():
    rec = _extract(VALID_RESPONSE)
    assert rec["title"] == "Senior Backend Engineer"
    assert rec["company"] == "Acme"
    assert rec["locations_all"] == ["McLean, VA", "Austin, TX"]
    assert rec["required_qualifications"] == ["5+ years of experience", "Bachelor's degree in CS"]
    assert rec["preferred_qualifications"] == ["AWS experience"]
    assert rec["description_full"] == SAMPLE_TEXT  # always the raw text
    assert rec["source_url"] == "https://x/jobs/1"


def test_list_coercion_from_string():
    resp = dict(VALID_RESPONSE, required_qualifications="Must have Python")
    rec = _extract(resp)
    assert rec["required_qualifications"] == ["Must have Python"]


def test_null_handling():
    resp = dict(VALID_RESPONSE, posted_date=None)
    rec = _extract(resp)
    assert rec["posted_date"] is None


def test_hints_override_llm():
    hints = {"job_id": "REQ-9", "company": "RealCo", "title": "Override Title"}
    rec = _extract(VALID_RESPONSE, hints=hints)
    assert rec["job_id"] == "REQ-9"
    assert rec["company"] == "RealCo"
    assert rec["title"] == "Override Title"


def test_malformed_output_preserves_description_full():
    rec = _extract({})
    assert rec["description_full"] == SAMPLE_TEXT
    assert rec["title"] is None
    assert rec["required_qualifications"] == []


def test_extractor_never_raises_on_empty_text():
    rec = _extract(VALID_RESPONSE, text="")
    assert rec["description_full"] is None
    rec2 = _extract({}, text="", hints={"title": "T"})
    assert rec2["title"] == "T"


# --------------------------------------------------------------------------- #
# Ollama safe-parse (the JSON extraction layer)
# --------------------------------------------------------------------------- #
def test_safe_json_extract_plain():
    assert _safe_json_extract('{"a": 1}') == {"a": 1}


def test_safe_json_extract_fenced():
    assert _safe_json_extract('```json\n{"a": 1}\n```') == {"a": 1}


def test_safe_json_extract_embedded():
    assert _safe_json_extract('Here you go: {"b": 2} thanks!') == {"b": 2}


def test_safe_json_extract_garbage_returns_none():
    assert _safe_json_extract("not json at all") is None
    assert _safe_json_extract("") is None
