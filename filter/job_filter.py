"""Post-scrape qualification filter.

Reads jobs from the DB and classifies each as "qualified" or "not_qualified"
based on two disqualifying criteria:

  1. Active security clearance required — detected via Ollama because simple
     keyword matching produces false positives ("active state bar membership",
     "actively pursuing CPA", etc.).  Ollama is only called when clearance-
     related terms actually appear in required_qualifications.

  2. 3+ years of experience required in any skill/domain — detected by regex
     on each required_qualifications item.

Results are persisted to the jobs table (filter_status, filter_reason) so
re-runs skip already-evaluated jobs.

Note: title-relevance filtering (keyword match on the job title) is handled
upstream in runner.py before jobs are saved to the DB, so it is not repeated
here.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from rich.progress import track

from agent.llm import LLMClient
from storage.job_store import JobStore

logger = logging.getLogger(__name__)

# Matches an experience requirement and captures the LOWER bound of any range:
# "3+ years" -> 3, "3 plus years" -> 3, "3-5 years" -> 3, "1-3 years" -> 1 (kept),
# "5 or more years" -> 5, "1 to 3 years" -> 1, "three (3) years" -> 3. The first number
# before "years" is the lower bound, so capture it directly (don't require it to be >=3,
# which mis-read "1-3" as "3"). Tolerates "plus", a parenthesised number, and an
# upper-bound after a range connector.
_EXP_RE = re.compile(
    r'\(?\b(\d+(?:\.\d+)?)\b\)?'          # the lower-bound number, optionally in ( )
    r'\s*(?:\+|plus|[-–]|to)?\s*'         # optional connector: +, "plus", -, "to"
    r'\(?\d*\)?\+?'                       # optional upper bound (e.g. the 5 in "3-5")
    r'\s*(?:or\s+more\s+)?'               # optional "or more"
    r'years?\b',
    re.I,
)

# Unambiguous clearance phrases — these REQUIRE a clearance by definition, so we
# disqualify deterministically without bothering the LLM (no false positives, no
# dependence on the model answering / parsing correctly).
_CLEARANCE_HARD = re.compile(
    r'\b(polygraph|poly|ts/sci|ts\s+sci|top\s+secret|sci\s+clearance|'
    r'(?:active|current|interim|secret|security|dod)\s+clearance)\b',
    re.I,
)

# Soft pre-screen: ambiguous mentions ("clearance", "secret", "classified") that need
# the LLM to disambiguate (e.g. "active state bar membership" is NOT a clearance).
_CLEARANCE_TERMS = re.compile(
    r'\b(clearance|secret|ts/sci|ts\s+sci|classified|dod\s+clearance|top\s+secret)\b',
    re.I,
)

_CLEARANCE_PROMPT = """\
You are a job screener. Given the required qualifications listed below, answer:
Does this role REQUIRE an active or current security clearance (Secret, Top Secret, TS/SCI, Poly or similar)?

Rules:
- Only answer true if clearance is explicitly REQUIRED, not just preferred or mentioned.
- "Active membership in a state bar" is NOT a security clearance — answer false.
- "Actively pursuing CPA" is NOT a security clearance — answer false.

Required qualifications:
{quals}

Respond with JSON only, no explanation:
{{"requires_clearance": true_or_false, "evidence": "quoted line or null"}}"""


class JobFilter:
    def __init__(self, store: JobStore, ollama: LLMClient) -> None:
        self.store = store
        self.ollama = ollama

    # ---------------------------------------------------------------- public

    def run(
        self,
        company: Optional[str] = None,
        limit: Optional[int] = None,
        rerun: bool = False,
    ) -> dict[str, list[dict]]:
        """Classify jobs and return {"qualified": [...], "not_qualified": [...]}."""
        jobs = self.store.get_jobs_for_filter(company=company, limit=limit, rerun=rerun)
        if not jobs:
            return {"qualified": [], "not_qualified": []}

        qualified: list[dict] = []
        not_qualified: list[dict] = []

        for job in track(jobs, description="Filtering jobs…"):
            reasons: list[str] = []

            exp = self._check_experience(job)
            if exp:
                reasons.append(exp)

            clearance = self._check_clearance(job)
            if clearance:
                reasons.append(clearance)

            if reasons:
                status = "not_qualified"
                reason_str = "; ".join(reasons)
                not_qualified.append({**job, "_filter_reason": reason_str})
            else:
                status = "qualified"
                reason_str = ""
                qualified.append(job)

            self.store.update_filter_status(job["id"], status, reason_str)

        return {"qualified": qualified, "not_qualified": not_qualified}

    def delete_not_qualified(self) -> int:
        """Delete all jobs previously classified as not_qualified.

        Returns the number of rows removed.
        """
        deleted = self.store.delete_by_filter_status("not_qualified")
        logger.info("Deleted %d not_qualified jobs", deleted)
        return deleted

    # --------------------------------------------------------------- private

    def _check_experience(self, job: dict) -> Optional[str]:
        """Return a reason string if ≥3 years of experience is required, else None."""
        candidates: list[str] = []

        for item in job.get("required_qualifications") or []:
            if isinstance(item, str):
                candidates.append(item)

        for text in candidates:
            m = _EXP_RE.search(text)
            if m:
                try:
                    years = float(m.group(1))
                except ValueError:
                    continue
                if years >= 3:
                    return f"{years:g}+ years of experience required"

        return None

    def _check_clearance(self, job: dict) -> Optional[str]:
        """Return a reason string if active clearance is required, else None.

        Unambiguous phrases (polygraph, TS/SCI, "Secret clearance", …) disqualify
        deterministically. Only ambiguous mentions fall back to the LLM.
        """
        quals: list[str] = [
            item for item in (job.get("required_qualifications") or [])
            if isinstance(item, str)
        ]
        quals_text = "\n".join(quals)

        # Unambiguous clearance language — no LLM needed (and no risk of it answering wrong).
        hard = _CLEARANCE_HARD.search(quals_text)
        if hard:
            return f"Clearance required: {hard.group(0)}"

        if not _CLEARANCE_TERMS.search(quals_text):
            return None

        prompt = _CLEARANCE_PROMPT.format(quals=quals_text)
        result = self.ollama.generate_json(prompt, default={"requires_clearance": False, "evidence": None})

        if not isinstance(result, dict):
            logger.warning("Unexpected Ollama response for job %s: %r", job.get("id"), result)
            return None

        if result.get("requires_clearance"):
            evidence = result.get("evidence") or "security clearance"
            return f"Clearance required: {evidence}"

        return None
