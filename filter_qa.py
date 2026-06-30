"""filter_qa.py — agentic quality-assurance pass over the job filter.

Finds jobs that slipped through ``filter/job_filter.py`` incorrectly (and jobs that
were disqualified incorrectly), then proposes precise new rules to catch the misses
in future runs. Every rule is written to ``config/filter_rules.yaml`` only after
explicit human approval in the session — nothing is applied silently.

    python filter_qa.py [--dry-run] [--limit N] [--debug]

Model routing (per the project plan):
  * batch auditing / false-positive auditing -> qwen2.5-coder:14b via Ollama
  * rule proposal                            -> Claude Haiku via the anthropic SDK
  * regression testing                       -> pure Python, no LLM

Orchestration note
------------------
``agent/tool_runner.py`` exists for genuinely open-ended LLM loops. This audit,
by contrast, is a fixed pipeline: three
single-shot model calls (batch audit, FP audit, rule proposal) wrapped around
deterministic Python (human review UI, regression test, rule writing, DB tagging).
Handing that fixed sequence to an LLM orchestrator would add cost and a real failure
mode — the model could skip the regression test or write rules without approval,
violating hard constraints ("no silent rule application", "no rule without explicit
human approval"). So the workflow is driven by an explicit ``AuditState`` machine
here; the model is used only inside the individual tools, each with its assigned
model. The tools are still standalone functions and are also exposed as ``ToolDef``
objects via :func:`build_tooldefs` so they can be wired into the runner if desired.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import yaml
from rich.console import Console

from agent.human_input import ask_human, yes_all_enabled
from agent.tool_runner import ToolDef
from filter.job_filter import (
    default_filter_rules_path,
    regex_rule_matches,
    rule_field_text,
)
from storage.job_store import JobStore

logger = logging.getLogger(__name__)
console = Console()

# --------------------------------------------------------------------------- #
# Status reconciliation.
#
# The plan describes filter_status values 'passed'/'disqualified', but the live
# pipeline (filter/job_filter.py) writes 'qualified'/'not_qualified'. We audit the
# real data, so map the plan's names onto the actual ones here in one place.
# --------------------------------------------------------------------------- #
PASSED_STATUS = "qualified"
DISQUALIFIED_STATUS = "not_qualified"
FLAGGED_STATUS = "flagged"          # a passed job confirmed as a miss
REVIEWED_OK_STATUS = "reviewed_ok"  # a job a human looked at and cleared

AUDIT_MODEL = "qwen2.5-coder:14b"
BATCH_SIZE = 20
SUSPECT_FLAGS = {"yes", "unclear"}


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
class AuditState(Enum):
    LOADING = "loading"
    AUDITING = "auditing"
    COLLECTING_SUSPECTS = "collecting_suspects"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    CONFIRMING_MISSES = "confirming_misses"
    PROPOSING_RULES = "proposing_rules"
    REGRESSION_TESTING = "regression_testing"
    AWAITING_RULE_APPROVAL = "awaiting_rule_approval"
    WRITING_RULES = "writing_rules"
    TAGGING_DB = "tagging_db"
    DONE = "done"


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def new_run_id() -> str:
    return "audit_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# JSON coercion helpers (model output is never fully trusted)
# --------------------------------------------------------------------------- #
def _coerce_list(parsed: Any) -> list:
    """Pull a list out of model output that may be a bare array or an object wrapper."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("results", "jobs", "audits", "data", "items", "rules"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return [parsed]  # single object -> one-element list
    return []


def _job_audit_text(job: dict) -> str:
    """Compact per-job text block for the auditor model."""
    quals = "\n".join(
        f"  - {q}" for q in (job.get("required_qualifications") or [])
        if isinstance(q, str)
    )
    desc = (job.get("description_full") or "")[:1500]
    return (
        f"id: {job.get('id')}\n"
        f"title: {job.get('title')}\n"
        f"company: {job.get('company')}\n"
        f"required_qualifications:\n{quals or '  (none listed)'}\n"
        f"description (truncated):\n{desc}"
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def load_passed_jobs(store: JobStore, limit: Optional[int] = None) -> dict:
    """All jobs the filter let through (filter_status == passed)."""
    jobs = store.get_jobs_by_filter_status(PASSED_STATUS, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


def load_disqualified_jobs(store: JobStore, limit: Optional[int] = None) -> dict:
    """All jobs the filter rejected — the false-positive search space."""
    jobs = store.get_jobs_by_filter_status(DISQUALIFIED_STATUS, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


_AUDIT_PROMPT = """\
You are auditing job listings for a filtering system. For each job below, determine:
1. Does this job require 3 or more years of experience? (yes/no/unclear)
   - Look for: explicit years, level indicators (Level 4+, P4+, Sr./Senior), grade levels
2. Does this job require an ACTIVE security clearance? (yes/no/unclear)
   - "Active" means currently held, not "ability to obtain"
   - Ignore: clearance sponsorship offered, "clearance preferred", "clearance eligible"

Respond ONLY with a JSON object of the form:
{{"results": [{{"id": <job_id>, "exp_flag": "yes|no|unclear", "clr_flag": "yes|no|unclear",
  "exp_reason": "<brief>", "clr_reason": "<brief>"}}]}}
One object per job, in the same order.

Jobs:
{jobs}"""


def audit_batch(jobs: list[dict], ollama) -> dict:
    """Audit passed jobs with qwen2.5-coder:14b in batches of 20.

    A job is a *suspect* if either flag comes back "yes" or "unclear".
    Returns {suspects, clean, flagged}. ``suspects`` carry the original job dict
    plus the audit flags/reasons so the human reviewer has full context.
    """
    by_id = {str(j.get("id")): j for j in jobs}
    suspects: list[dict] = []
    flagged = 0

    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        blocks = "\n\n".join(_job_audit_text(j) for j in batch)
        raw = ollama.generate_json(
            _AUDIT_PROMPT.format(jobs=blocks), default={"results": []}
        )
        for entry in _coerce_list(raw):
            if not isinstance(entry, dict):
                continue
            job = by_id.get(str(entry.get("id")))
            if job is None:
                continue
            exp = str(entry.get("exp_flag", "no")).lower()
            clr = str(entry.get("clr_flag", "no")).lower()
            if exp in SUSPECT_FLAGS or clr in SUSPECT_FLAGS:
                flagged += 1
                suspects.append({
                    "id": job.get("id"),
                    "title": job.get("title"),
                    "company": job.get("company"),
                    "exp_flag": exp,
                    "clr_flag": clr,
                    "exp_reason": entry.get("exp_reason", ""),
                    "clr_reason": entry.get("clr_reason", ""),
                    "_job": job,
                })
    return {"suspects": suspects, "clean": len(jobs) - flagged, "flagged": flagged}


_FP_PROMPT = """\
For each job below, determine if it was INCORRECTLY disqualified.
A job was correctly disqualified if it genuinely requires 3+ years experience
OR an active security clearance.
A job was INCORRECTLY disqualified if the requirement was misread
(e.g., clearance mentioned only in a "nice to have" section, or years
in a description of a past role, not a requirement).

Respond ONLY with a JSON object of the form:
{{"results": [{{"id": <job_id>, "incorrectly_disqualified": true|false, "reason": "<brief>"}}]}}

Jobs:
{jobs}"""


def audit_false_positives(jobs: list[dict], ollama) -> dict:
    """Check disqualified jobs for ones rejected by mistake. Returns {false_positives, count}."""
    by_id = {str(j.get("id")): j for j in jobs}
    false_positives: list[dict] = []
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        blocks = "\n\n".join(_job_audit_text(j) for j in batch)
        raw = ollama.generate_json(
            _FP_PROMPT.format(jobs=blocks), default={"results": []}
        )
        for entry in _coerce_list(raw):
            if not isinstance(entry, dict) or not entry.get("incorrectly_disqualified"):
                continue
            job = by_id.get(str(entry.get("id")))
            if job is None:
                continue
            false_positives.append({
                "id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company"),
                "reason": entry.get("reason", ""),
                "_job": job,
            })
    return {"false_positives": false_positives, "count": len(false_positives)}


_PROPOSE_PROMPT = """\
Given these job listings that were missed by the current filter, propose new filter rules.
Current rules already cover:
- Regex: explicit years >= 3 (e.g. "3+ years", "5 years of experience")
- Regex: active/current TS/SCI, secret, top secret patterns
- LLM disambiguation for ambiguous clearance mentions

Propose ONLY rules that cover NEW patterns not already handled.
For each rule, specify:
- category: "experience" or "clearance"
- pattern: the regex string (Python re syntax)
- type: "regex" or "llm_disambiguate"
- field: "required_qualifications" | "title" | "full_text"
- action: "disqualify" | "disqualify_if_confirmed"
- description: plain English explanation
- example_match: the exact text from the job that triggered this

Respond ONLY with a JSON object: {{"rules": [ <rule objects> ]}}

Missed jobs:
{jobs}"""


def propose_rules(confirmed_misses: list[dict], claude) -> dict:
    """Claude Haiku proposes regex/llm rules covering the confirmed misses."""
    if not confirmed_misses:
        return {"proposed_rules": []}
    blocks = []
    for miss in confirmed_misses:
        job = miss.get("_job", miss)
        snippet = (job.get("description_full") or "")[:800]
        quals = "; ".join(
            q for q in (job.get("required_qualifications") or []) if isinstance(q, str)
        )
        blocks.append(
            f"id: {job.get('id')}\ntitle: {job.get('title')}\n"
            f"flag_reason: {miss.get('reason') or miss.get('exp_reason') or miss.get('clr_reason') or ''}\n"
            f"required_qualifications: {quals}\n"
            f"text: {snippet}"
        )
    raw = claude.generate_json(
        _PROPOSE_PROMPT.format(jobs="\n\n".join(blocks)), default={"rules": []}
    )
    rules = [r for r in _coerce_list(raw) if isinstance(r, dict) and r.get("pattern")]
    # Validate each pattern compiles; drop the ones that don't.
    valid: list[dict] = []
    for rule in rules:
        try:
            re.compile(rule["pattern"])
            valid.append(rule)
        except re.error as exc:
            logger.warning("Discarding proposed rule with bad regex %r: %s",
                           rule.get("pattern"), exc)
    return {"proposed_rules": valid}


def _infer_category(rule: dict) -> str:
    """experience vs clearance — trust an explicit category, else sniff the text."""
    cat = str(rule.get("category", "")).lower()
    if cat.startswith("exp"):
        return "experience"
    if cat.startswith("clr") or cat.startswith("clear"):
        return "clearance"
    blob = f"{rule.get('pattern','')} {rule.get('description','')}".lower()
    if any(t in blob for t in ("clear", "secret", "ts/sci", "poly", "sci")):
        return "clearance"
    return "experience"


def regression_test(proposed_rules: list[dict], all_jobs: list[dict]) -> dict:
    """Pure-Python impact analysis of proposed rules over every job in the DB.

    For each *regex* rule:
      - newly_disqualified: currently-passed jobs the rule would newly disqualify
        (the real risk metric the human weighs before approving).
      - flipped_correct: currently-disqualified jobs the rule also matches — i.e. it
        corroborates an existing disqualification rather than contradicting it. (A
        disqualify rule can never turn a job into a pass, so this is the faithful,
        decision-useful reading of the plan's "flipped" counter.)
      - sample_matches: up to 5 "title :: matched-text" examples from passed jobs.
    ``llm_disambiguate`` rules can't be evaluated deterministically and are reported
    with a null count so the human knows they were not regression-tested.
    """
    passed = [j for j in all_jobs if j.get("filter_status") == PASSED_STATUS]
    disqualified = [j for j in all_jobs if j.get("filter_status") == DISQUALIFIED_STATUS]

    per_rule: list[dict] = []
    for idx, rule in enumerate(proposed_rules):
        rid = rule.get("id") or f"proposed_{idx + 1}"
        if rule.get("type") == "llm_disambiguate":
            per_rule.append({
                "rule_id": rid,
                "newly_disqualified": None,
                "flipped_correct": None,
                "sample_matches": [],
                "note": "llm_disambiguate rule — not regression-tested (needs model)",
            })
            continue

        newly = 0
        samples: list[str] = []
        for job in passed:
            matched = regex_rule_matches(job, rule)
            if matched:
                newly += 1
                if len(samples) < 5:
                    samples.append(f"{job.get('title')} :: {matched}")
        corroborates = sum(1 for job in disqualified if regex_rule_matches(job, rule))
        per_rule.append({
            "rule_id": rid,
            "newly_disqualified": newly,
            "flipped_correct": corroborates,
            "sample_matches": samples,
        })
    return {"per_rule": per_rule}


def write_rules(
    approved_rules: list[dict],
    path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Append approved rules to config/filter_rules.yaml (created if missing).

    Rules are split into experience_rules / clearance_rules with auto-incrementing
    ``exp_NNN`` / ``clr_NNN`` ids. With ``dry_run`` nothing is written — the would-be
    result is still returned so the caller can report it.
    """
    path = path or default_filter_rules_path()
    data: dict = {"version": 1, "experience_rules": [], "clearance_rules": []}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            data.update({k: v for k, v in loaded.items() if v is not None})
        except Exception as exc:
            logger.error("Could not read existing %s: %s", path, exc)
    data.setdefault("experience_rules", [])
    data.setdefault("clearance_rules", [])

    def _next_id(category_key: str, prefix: str) -> Callable[[], str]:
        existing = data.get(category_key) or []
        max_n = 0
        for r in existing:
            m = re.match(rf"{prefix}_(\d+)$", str(r.get("id", "")))
            if m:
                max_n = max(max_n, int(m.group(1)))
        counter = {"n": max_n}

        def _gen() -> str:
            counter["n"] += 1
            return f"{prefix}_{counter['n']:03d}"

        return _gen

    next_exp = _next_id("experience_rules", "exp")
    next_clr = _next_id("clearance_rules", "clr")

    written = 0
    for rule in approved_rules:
        category = _infer_category(rule)
        if category == "clearance":
            key, rid = "clearance_rules", next_clr()
        else:
            key, rid = "experience_rules", next_exp()
        entry = {
            "id": rid,
            "description": rule.get("description", ""),
            "type": rule.get("type", "regex"),
            "pattern": rule.get("pattern", ""),
            "field": rule.get("field", "full_text"),
            "action": rule.get("action", "disqualify"),
            "added": _today(),
            "approved_by": "human",
        }
        if rule.get("example_match"):
            entry["example_match"] = rule["example_match"]
        data[key].append(entry)
        written += 1

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return {"path": path, "rules_written": written, "dry_run": dry_run}


def tag_jobs(
    store: JobStore,
    job_ids: list[int],
    status: str,
    reason: str,
    run_id: str,
    dry_run: bool = False,
) -> dict:
    """Set filter_status / filter_flag_reason / audit_run_id for the given jobs."""
    if dry_run:
        return {"updated": 0, "dry_run": True, "would_update": len(job_ids or [])}
    updated = store.tag_audit(job_ids or [], status, reason, run_id)
    return {"updated": updated, "dry_run": False}


def build_tooldefs(store: JobStore, audit_client, rule_client) -> list[ToolDef]:
    """Expose the tools as ToolDefs (for ToolRunner wiring / parity with the plan)."""
    return [
        ToolDef("load_passed_jobs", "Load jobs that passed the filter.",
                {"type": "object", "properties": {}},
                lambda: load_passed_jobs(store)),
        ToolDef("load_disqualified_jobs", "Load jobs the filter disqualified.",
                {"type": "object", "properties": {}},
                lambda: load_disqualified_jobs(store)),
        ToolDef("audit_batch", "Audit passed jobs for missed disqualifiers.",
                {"type": "object", "properties": {"jobs": {"type": "array"}}},
                lambda jobs: audit_batch(jobs, audit_client)),
        ToolDef("audit_false_positives", "Audit disqualified jobs for mistakes.",
                {"type": "object", "properties": {"jobs": {"type": "array"}}},
                lambda jobs: audit_false_positives(jobs, audit_client)),
        ToolDef("propose_rules", "Propose new filter rules for confirmed misses.",
                {"type": "object", "properties": {"confirmed_misses": {"type": "array"}}},
                lambda confirmed_misses: propose_rules(confirmed_misses, rule_client)),
        ToolDef("regression_test", "Measure impact of proposed rules over all jobs.",
                {"type": "object", "properties": {"proposed_rules": {"type": "array"}}},
                lambda proposed_rules: regression_test(
                    proposed_rules, store.get_jobs())),
        ToolDef("write_rules", "Write approved rules to filter_rules.yaml.",
                {"type": "object", "properties": {"approved_rules": {"type": "array"}}},
                lambda approved_rules: write_rules(approved_rules)),
        ToolDef("tag_jobs", "Tag jobs with an audit status.",
                {"type": "object", "properties": {
                    "job_ids": {"type": "array"}, "status": {"type": "string"},
                    "reason": {"type": "string"}, "run_id": {"type": "string"}}},
                lambda job_ids, status, reason, run_id: tag_jobs(
                    store, job_ids, status, reason, run_id)),
        ToolDef("ask_human", "Pause and ask the human a question.",
                {"type": "object", "properties": {
                    "question": {"type": "string"}, "context": {"type": "object"}}},
                lambda question, context=None: {"response": ask_human(question, context)}),
    ]


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class AuditReport:
    run_id: str
    dry_run: bool
    passed_count: int = 0
    disqualified_count: int = 0
    suspects: int = 0
    confirmed_misses: int = 0
    kept_ok: int = 0
    false_positives_confirmed: int = 0
    proposed_rules: int = 0
    approved_rules: int = 0
    rules_written: int = 0
    jobs_tagged: int = 0
    state_history: list[str] = field(default_factory=list)


class FilterQA:
    def __init__(
        self,
        store: JobStore,
        audit_client,
        rule_client,
        dry_run: bool = False,
        limit: Optional[int] = None,
        rules_path: Optional[str] = None,
    ):
        self.store = store
        self.audit = audit_client
        self.rules = rule_client
        self.dry_run = dry_run
        self.limit = limit
        self.rules_path = rules_path or default_filter_rules_path()
        self.run_id = new_run_id()
        self.state = AuditState.LOADING
        self.report = AuditReport(run_id=self.run_id, dry_run=dry_run)

    # ------------------------------------------------------------- transitions
    def _to(self, state: AuditState) -> None:
        self.state = state
        line = f"[{_now_stamp()}] STATE -> {state.value}"
        self.report.state_history.append(state.value)
        console.print(f"[dim]{line}[/]")
        logger.info(line)

    # --------------------------------------------------------------------- run
    def run(self) -> AuditReport:
        self._to(AuditState.LOADING)
        passed = load_passed_jobs(self.store, limit=self.limit)["jobs"]
        disqualified = load_disqualified_jobs(self.store)["jobs"]
        self.report.passed_count = len(passed)
        self.report.disqualified_count = len(disqualified)
        console.print(
            f"Loaded [bold]{len(passed)}[/] passed and "
            f"[bold]{len(disqualified)}[/] disqualified jobs."
        )
        if not passed and not disqualified:
            console.print("[yellow]No jobs to audit. Has the filter run yet?[/]")
            self._to(AuditState.DONE)
            return self.report

        # --- audit -------------------------------------------------------
        self._to(AuditState.AUDITING)
        audit = audit_batch(passed, self.audit)
        fp = audit_false_positives(disqualified, self.audit)
        console.print(
            f"Audit: {audit['flagged']} suspect(s), {audit['clean']} clean; "
            f"{fp['count']} possible false-positive(s)."
        )

        self._to(AuditState.COLLECTING_SUSPECTS)
        suspects = audit["suspects"]
        self.report.suspects = len(suspects)

        # --- human review of suspects -----------------------------------
        self._to(AuditState.AWAITING_HUMAN_REVIEW)
        confirmed_misses, kept_ok = self._review_suspects(suspects)
        self.report.confirmed_misses = len(confirmed_misses)
        self.report.kept_ok = len(kept_ok)

        # --- confirm false positives ------------------------------------
        self._to(AuditState.CONFIRMING_MISSES)
        confirmed_fps = self._review_false_positives(fp["false_positives"])
        self.report.false_positives_confirmed = len(confirmed_fps)

        # --- propose + regression + approve -----------------------------
        approved_rules: list[dict] = []
        if confirmed_misses:
            self._to(AuditState.PROPOSING_RULES)
            proposed = propose_rules(confirmed_misses, self.rules)["proposed_rules"]
            self.report.proposed_rules = len(proposed)
            console.print(f"Proposed [bold]{len(proposed)}[/] new rule(s).")

            if proposed:
                self._to(AuditState.REGRESSION_TESTING)
                all_jobs = self.store.get_jobs()
                regression = regression_test(proposed, all_jobs)["per_rule"]

                self._to(AuditState.AWAITING_RULE_APPROVAL)
                approved_rules = self._approve_rules(proposed, regression)
                self.report.approved_rules = len(approved_rules)
        else:
            console.print("[green]No confirmed misses — no rules to propose.[/]")

        # --- write rules -------------------------------------------------
        if approved_rules:
            self._to(AuditState.WRITING_RULES)
            result = write_rules(approved_rules, self.rules_path, dry_run=self.dry_run)
            self.report.rules_written = result["rules_written"]
            verb = "Would write" if self.dry_run else "Wrote"
            console.print(f"{verb} [bold]{result['rules_written']}[/] rule(s) to {result['path']}.")

        # --- tag DB ------------------------------------------------------
        self._to(AuditState.TAGGING_DB)
        self._tag_results(confirmed_misses, kept_ok, confirmed_fps)

        self._to(AuditState.DONE)
        self._print_summary()
        return self.report

    # ----------------------------------------------------------- sub-routines
    def _review_suspects(self, suspects: list[dict]) -> tuple[list[dict], list[dict]]:
        """Per-job [d]isqualify / [k]eep / [s]kip. Returns (confirmed_misses, kept_ok)."""
        confirmed: list[dict] = []
        kept: list[dict] = []
        if not suspects:
            return confirmed, kept

        table_rows = [
            {
                "title": s.get("title"),
                "company": s.get("company"),
                "flag": f"exp={s['exp_flag']} clr={s['clr_flag']}",
                "reason": s.get("exp_reason") or s.get("clr_reason") or "",
            }
            for s in suspects
        ]
        console.rule("[bold]Suspect jobs (filter may have missed these)")
        for i, s in enumerate(suspects, start=1):
            answer = ask_human(
                f"Job {i}/{len(suspects)}: {s.get('title')} @ {s.get('company')} — "
                f"[d]isqualify / [k]eep / [s]kip?",
                context={"job": table_rows[i - 1]},
                options=["d", "k", "s"],
            )
            if answer == "d":
                confirmed.append(s)
            elif answer == "k":
                kept.append(s)
            # 's' -> skip (no DB change)
        return confirmed, kept

    def _review_false_positives(self, fps: list[dict]) -> list[dict]:
        """Confirm which disqualified jobs were rejected by mistake (restore to passed)."""
        confirmed: list[dict] = []
        if not fps:
            return confirmed
        console.rule("[bold]Possible false positives (disqualified by mistake?)")
        for i, fp in enumerate(fps, start=1):
            answer = ask_human(
                f"FP {i}/{len(fps)}: {fp.get('title')} @ {fp.get('company')} — "
                f"[r]estore to passed / [k]eep disqualified / [s]kip?",
                context={"reason": fp.get("reason", "")},
                options=["r", "k", "s"],
            )
            if answer == "r":
                confirmed.append(fp)
        return confirmed

    def _approve_rules(self, proposed: list[dict], regression: list[dict]) -> list[dict]:
        """Show each rule + its regression impact; collect explicit approvals."""
        by_id = {r["rule_id"]: r for r in regression}
        approved: list[dict] = []
        console.rule("[bold]Proposed rules — review impact before approving")
        for idx, rule in enumerate(proposed):
            reg = by_id.get(f"proposed_{idx + 1}", {})
            context = {
                "category": _infer_category(rule),
                "type": rule.get("type"),
                "pattern": rule.get("pattern"),
                "field": rule.get("field"),
                "description": rule.get("description"),
                "newly_disqualified": reg.get("newly_disqualified"),
                "corroborates_disqualified": reg.get("flipped_correct"),
                "sample_matches": reg.get("sample_matches", []),
            }
            answer = ask_human(
                f"Approve rule {idx + 1}/{len(proposed)}? "
                f"(would newly disqualify {reg.get('newly_disqualified')} passed job(s))",
                context=context,
                options=["approve", "reject"],
            )
            if answer == "approve":
                approved.append(rule)
        return approved

    def _tag_results(
        self,
        confirmed_misses: list[dict],
        kept_ok: list[dict],
        confirmed_fps: list[dict],
    ) -> None:
        total = 0
        if confirmed_misses:
            ids = [m["id"] for m in confirmed_misses]
            res = tag_jobs(self.store, ids, FLAGGED_STATUS,
                           "confirmed miss during audit", self.run_id, self.dry_run)
            total += res.get("updated", 0)
        if kept_ok:
            ids = [k["id"] for k in kept_ok]
            res = tag_jobs(self.store, ids, REVIEWED_OK_STATUS,
                           "reviewed, kept as passed", self.run_id, self.dry_run)
            total += res.get("updated", 0)
        if confirmed_fps:
            ids = [f["id"] for f in confirmed_fps]
            # Restore mistaken disqualifications back to passed, flagged as reviewed.
            res = tag_jobs(self.store, ids, PASSED_STATUS,
                           "restored: incorrectly disqualified", self.run_id, self.dry_run)
            total += res.get("updated", 0)
        self.report.jobs_tagged = total
        verb = "Would tag" if self.dry_run else "Tagged"
        console.print(f"{verb} [bold]{total}[/] job(s) in the DB.")

    def _print_summary(self) -> None:
        r = self.report
        console.rule("[bold green]Audit complete")
        console.print(
            f"run_id={r.run_id} dry_run={r.dry_run}\n"
            f"  passed audited:   {r.passed_count}\n"
            f"  disqualified:     {r.disqualified_count}\n"
            f"  suspects:         {r.suspects}\n"
            f"  confirmed misses: {r.confirmed_misses}\n"
            f"  kept OK:          {r.kept_ok}\n"
            f"  false-pos fixed:  {r.false_positives_confirmed}\n"
            f"  rules proposed:   {r.proposed_rules}\n"
            f"  rules approved:   {r.approved_rules}\n"
            f"  rules written:    {r.rules_written}\n"
            f"  jobs tagged:      {r.jobs_tagged}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_clients():
    """Audit model (qwen via Ollama) + rule model (Claude Haiku via anthropic)."""
    from agent.anthropic_client import AnthropicClient
    from agent.ollama_client import OllamaClient

    audit_client = OllamaClient(model=AUDIT_MODEL)
    rule_client = AnthropicClient()
    return audit_client, rule_client


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Filter QA audit agent.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the full audit but write nothing to the DB or YAML.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Audit only N passed jobs (development/testing).")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose logging of LLM calls.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.dry_run:
        console.print("[yellow]DRY RUN — no DB or YAML writes will be made.[/]")
    if yes_all_enabled():
        console.print("[yellow]AGENT_YES_ALL=1 — prompts auto-answer with their first option.[/]")

    store = JobStore()
    try:
        audit_client, rule_client = _build_clients()
        qa = FilterQA(
            store, audit_client, rule_client,
            dry_run=args.dry_run, limit=args.limit,
        )
        qa.run()
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
