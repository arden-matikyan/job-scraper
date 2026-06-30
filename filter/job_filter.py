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

Additive YAML rules
-------------------
If ``config/filter_rules.yaml`` exists (authored by ``filter_qa.py`` after human
approval) its rules are loaded on init and applied *in addition to* the hard-coded
logic above — never replacing it. ``type: regex`` rules disqualify deterministically
and run before LLM disambiguation; ``type: llm_disambiguate`` rules pre-screen with
their regex then confirm with the model. With no YAML file present the filter behaves
exactly as it always has (fully backward compatible).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import yaml
from rich.progress import track

from agent.llm import LLMClient
from scrapers.base import non_us_location
from storage.job_store import JobStore

logger = logging.getLogger(__name__)

# Where filter_qa.py writes approved rules; also read here.
FILTER_RULES_FILENAME = "filter_rules.yaml"
_RULE_CATEGORIES = ("experience_rules", "clearance_rules")


def default_filter_rules_path() -> str:
    """Absolute path to config/filter_rules.yaml (created by filter_qa, optional here)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "config", FILTER_RULES_FILENAME)


# --------------------------------------------------------------------------- #
# Shared rule semantics
#
# These helpers define how a YAML rule maps onto a job dict. filter_qa.py imports
# them for its regression test so its impact predictions match production exactly.
# --------------------------------------------------------------------------- #
def rule_field_text(job: dict, field: Optional[str]) -> str:
    """Return the job text a rule's ``field`` targets ('title' | 'required_qualifications'
    | 'full_text'). Unknown / missing field defaults to full_text."""
    def _quals(key: str) -> str:
        return "\n".join(
            s for s in (job.get(key) or []) if isinstance(s, str)
        )

    if field == "title":
        return str(job.get("title") or "")
    if field == "required_qualifications":
        return _quals("required_qualifications")
    # full_text: everything we have, so a rule can match anywhere in the listing.
    parts = [
        str(job.get("title") or ""),
        str(job.get("description_full") or ""),
        _quals("required_qualifications"),
        _quals("preferred_qualifications"),
    ]
    return "\n".join(p for p in parts if p)


def compile_rule(rule: dict) -> Optional[re.Pattern]:
    """Compile a rule's regex (case-insensitive). Returns None on a bad/empty pattern."""
    pattern = rule.get("pattern")
    if not pattern:
        return None
    try:
        return re.compile(pattern, re.I)
    except re.error as exc:
        logger.warning("Ignoring filter rule %s — bad regex %r: %s",
                       rule.get("id", "?"), pattern, exc)
        return None


def regex_rule_matches(job: dict, rule: dict) -> Optional[str]:
    """If the rule's regex hits its target field, return the matched substring, else None."""
    rx = compile_rule(rule)
    if rx is None:
        return None
    m = rx.search(rule_field_text(job, rule.get("field") or "full_text"))
    return m.group(0) if m else None


def load_filter_rules(path: Optional[str] = None) -> dict:
    """Load filter_rules.yaml as ``{'experience_rules': [...], 'clearance_rules': [...]}``.

    A missing or unreadable file yields empty lists (backward compatible — the
    filter then runs with hard-coded logic only).
    """
    path = path or default_filter_rules_path()
    if not os.path.exists(path):
        return {cat: [] for cat in _RULE_CATEGORIES}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Could not load %s: %s", path, exc)
        data = {}
    return {cat: list(data.get(cat) or []) for cat in _RULE_CATEGORIES}

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
    def __init__(
        self,
        store: JobStore,
        ollama: LLMClient,
        rules_path: Optional[str] = None,
    ) -> None:
        self.store = store
        self.ollama = ollama
        # Additive YAML rules (optional). Split by mechanism so regex rules can run
        # before LLM disambiguation, per the spec. Each rule keeps its category so a
        # disqualification reason can name it.
        self.regex_rules: list[dict] = []
        self.llm_rules: list[dict] = []
        rules = load_filter_rules(rules_path)
        for category in _RULE_CATEGORIES:
            for rule in rules.get(category) or []:
                tagged = {**rule, "category": category}
                if tagged.get("type") == "llm_disambiguate":
                    self.llm_rules.append(tagged)
                else:
                    self.regex_rules.append(tagged)
        if self.regex_rules or self.llm_rules:
            logger.info(
                "Loaded %d regex + %d llm_disambiguate YAML filter rules",
                len(self.regex_rules), len(self.llm_rules),
            )
        # Non-US location filter (same switch as the runner's pre-LLM stage). This
        # catches scrapers whose location is only known after LLM extraction (Avature
        # / iCIMS), at no extra model cost. Read from config/scraper_configs.yaml.
        self.us_only, self.location_deny_extra = self._load_location_config()

    @staticmethod
    def _load_location_config() -> tuple[bool, list[str]]:
        try:
            from config_io import config_path, load_yaml
            cfg = load_yaml(config_path("scraper_configs.yaml")) or {}
        except Exception as exc:
            logger.warning("Could not load scraper_configs for us_only: %s", exc)
            return False, []
        us_only = bool(cfg.get("us_only", False))
        extra = [str(t) for t in (cfg.get("location_deny_extra") or []) if str(t).strip()]
        return us_only, extra

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

            loc = self._check_location(job)
            if loc:
                reasons.append(loc)

            exp = self._check_experience(job)
            if exp:
                reasons.append(exp)

            # YAML regex rules run before LLM disambiguation (cheap, deterministic).
            yaml_regex = self._check_yaml_regex(job)
            if yaml_regex:
                reasons.append(yaml_regex)

            clearance = self._check_clearance(job)
            if clearance:
                reasons.append(clearance)

            # YAML llm_disambiguate rules: regex pre-screen, then model confirmation.
            yaml_llm = self._check_yaml_llm(job)
            if yaml_llm:
                reasons.append(yaml_llm)

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

    def _check_location(self, job: dict) -> Optional[str]:
        """Return a reason string if the job is recognizably non-US, else None.

        Gated on `us_only`; deny-list, fail-open (see non_us_location). Covers jobs
        whose location was only known after extraction (Avature / iCIMS).
        """
        if not self.us_only:
            return None
        location = job.get("location")
        locations_all = job.get("locations_all") or []
        if non_us_location(location, locations_all, self.location_deny_extra):
            return f"Non-US location: {location or (locations_all[0] if locations_all else '?')}"
        return None

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

    # ------------------------------------------------------------- YAML rules

    @staticmethod
    def _rule_label(rule: dict, detail: str) -> str:
        rid = rule.get("id", "?")
        desc = rule.get("description") or ""
        head = f"Rule {rid}" + (f" ({desc})" if desc else "")
        return f"{head}: {detail}"

    def _check_yaml_regex(self, job: dict) -> Optional[str]:
        """First matching ``type: regex`` disqualify rule, or None."""
        for rule in self.regex_rules:
            if rule.get("action") not in (None, "disqualify", "disqualify_if_confirmed"):
                continue
            matched = regex_rule_matches(job, rule)
            if matched:
                logger.info(
                    "Job %s disqualified by YAML rule %s: %r",
                    job.get("id"), rule.get("id", "?"), matched,
                )
                return self._rule_label(rule, matched)
        return None

    def _check_yaml_llm(self, job: dict) -> Optional[str]:
        """First ``type: llm_disambiguate`` rule whose regex hits AND the model confirms."""
        for rule in self.llm_rules:
            if not regex_rule_matches(job, rule):
                continue
            if self._llm_confirm_rule(job, rule):
                logger.info(
                    "Job %s disqualified by YAML llm rule %s",
                    job.get("id"), rule.get("id", "?"),
                )
                return self._rule_label(rule, "confirmed by model")
        return None

    def _llm_confirm_rule(self, job: dict, rule: dict) -> bool:
        """Ask the model whether ``job`` genuinely satisfies a disambiguation rule."""
        text = rule_field_text(job, rule.get("field") or "required_qualifications")
        prompt = (
            "You are a job screener. A candidate disqualification rule pre-matched this "
            "listing and needs confirmation.\n"
            f"Rule: {rule.get('description') or '(no description)'}\n"
            "Answer true ONLY if the job text genuinely satisfies the rule (an actual "
            "requirement, not a 'preferred' or 'nice to have' mention).\n\n"
            f"Job text:\n{text[:3000]}\n\n"
            'Respond with JSON only: {"confirmed": true_or_false, "evidence": "<quote or null>"}'
        )
        result = self.ollama.generate_json(prompt, default={"confirmed": False})
        return bool(isinstance(result, dict) and result.get("confirmed"))
