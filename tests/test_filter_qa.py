"""Tests for filter_qa.py — no network, no real LLM.

The audit (qwen) and rule (Claude) clients are replaced by fakes that return
scripted JSON in the same shape the real clients yield.
"""
from __future__ import annotations

import os

import yaml

import filter_qa
from filter_qa import (
    AuditState,
    FilterQA,
    audit_batch,
    audit_false_positives,
    propose_rules,
    regression_test,
    write_rules,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeStore:
    def __init__(self, by_status):
        self._by_status = by_status
        self.tagged: list[tuple] = []

    def get_jobs_by_filter_status(self, status, limit=None):
        jobs = list(self._by_status.get(status, []))
        return jobs[:limit] if limit else jobs

    def get_jobs(self):
        out = []
        for status, jobs in self._by_status.items():
            for j in jobs:
                out.append({**j, "filter_status": status})
        return out

    def tag_audit(self, ids, status, reason, run_id):
        self.tagged.append((list(ids), status, reason, run_id))
        return len(ids)

    def close(self):
        pass


class FakeOllama:
    """Audit model stand-in: branches on the FP prompt vs the suspect prompt."""

    def generate_json(self, prompt, default=None, system=None):
        if "INCORRECTLY disqualified" in prompt:
            return {"results": [
                {"id": 2, "incorrectly_disqualified": True, "reason": "nice-to-have only"}
            ]}
        return {"results": [
            {"id": 1, "exp_flag": "yes", "clr_flag": "no",
             "exp_reason": "5 years required", "clr_reason": ""}
        ]}


class FakeClaude:
    def generate_json(self, prompt, default=None, system=None):
        return {"rules": [
            {"category": "experience", "pattern": r"level\s*[4-9]", "type": "regex",
             "field": "title", "action": "disqualify",
             "description": "Level 4+ seniority", "example_match": "Level 4"},
            {"category": "experience", "pattern": "[invalid(", "type": "regex",
             "field": "title", "action": "disqualify", "description": "bad regex"},
        ]}


PASSED = [{"id": 1, "title": "Engineer Level 4", "company": "Raytheon",
           "required_qualifications": ["5 years of experience"], "description_full": "..."}]
DISQUALIFIED = [{"id": 2, "title": "Analyst", "company": "MITRE",
                 "required_qualifications": ["clearance preferred"], "description_full": "..."}]


# --------------------------------------------------------------------------- #
# Tool-level tests
# --------------------------------------------------------------------------- #
def test_audit_batch_flags_suspects():
    out = audit_batch(PASSED, FakeOllama())
    assert out["flagged"] == 1
    assert out["suspects"][0]["id"] == 1
    assert out["suspects"][0]["exp_flag"] == "yes"


def test_audit_false_positives():
    out = audit_false_positives(DISQUALIFIED, FakeOllama())
    assert out["count"] == 1
    assert out["false_positives"][0]["id"] == 2


def test_propose_rules_drops_bad_regex():
    misses = [{"_job": PASSED[0], "exp_reason": "5 years"}]
    out = propose_rules(misses, FakeClaude())
    assert len(out["proposed_rules"]) == 1  # the "[invalid(" pattern is dropped
    assert out["proposed_rules"][0]["pattern"] == r"level\s*[4-9]"


def test_regression_test_counts_newly_disqualified():
    all_jobs = (
        [{**PASSED[0], "filter_status": filter_qa.PASSED_STATUS}]
        + [{**DISQUALIFIED[0], "filter_status": filter_qa.DISQUALIFIED_STATUS}]
    )
    rule = {"pattern": r"level\s*[4-9]", "type": "regex", "field": "title",
            "action": "disqualify"}
    out = regression_test([rule], all_jobs)
    per = out["per_rule"][0]
    assert per["newly_disqualified"] == 1
    assert per["sample_matches"] and "Level 4" in per["sample_matches"][0]


def test_regression_test_marks_llm_rule_untested():
    rule = {"pattern": r"clearance", "type": "llm_disambiguate", "field": "full_text"}
    out = regression_test([rule], [])
    assert out["per_rule"][0]["newly_disqualified"] is None


def test_write_rules_auto_increments_and_splits_category(tmp_path):
    path = str(tmp_path / "filter_rules.yaml")
    exp_rule = {"category": "experience", "pattern": r"level\s*[4-9]", "type": "regex",
                "field": "title", "action": "disqualify", "description": "lvl"}
    clr_rule = {"category": "clearance", "pattern": r"ts/sci required", "type": "regex",
                "field": "full_text", "action": "disqualify", "description": "tssci"}

    first = write_rules([exp_rule], path)
    assert first["rules_written"] == 1
    second = write_rules([exp_rule, clr_rule], path)  # appends
    assert second["rules_written"] == 2

    data = yaml.safe_load(open(path, encoding="utf-8"))
    exp_ids = [r["id"] for r in data["experience_rules"]]
    clr_ids = [r["id"] for r in data["clearance_rules"]]
    assert exp_ids == ["exp_001", "exp_002"]
    assert clr_ids == ["clr_001"]
    assert data["experience_rules"][0]["approved_by"] == "human"


def test_write_rules_dry_run_writes_nothing(tmp_path):
    path = str(tmp_path / "filter_rules.yaml")
    out = write_rules([{"category": "experience", "pattern": "x", "type": "regex"}],
                      path, dry_run=True)
    assert out["rules_written"] == 1
    assert out["dry_run"] is True
    assert not os.path.exists(path)


# --------------------------------------------------------------------------- #
# End-to-end orchestration (AGENT_YES_ALL auto-answers every gate)
# --------------------------------------------------------------------------- #
def test_full_audit_dry_run(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_YES_ALL", "1")
    store = FakeStore({
        filter_qa.PASSED_STATUS: PASSED,
        filter_qa.DISQUALIFIED_STATUS: DISQUALIFIED,
    })
    rules_path = str(tmp_path / "filter_rules.yaml")
    qa = FilterQA(store, FakeOllama(), FakeClaude(),
                  dry_run=True, rules_path=rules_path)
    report = qa.run()

    assert qa.state is AuditState.DONE
    assert report.suspects == 1
    assert report.confirmed_misses == 1          # yes-all picks "d" (disqualify)
    assert report.false_positives_confirmed == 1  # yes-all picks "r" (restore)
    assert report.proposed_rules == 1
    assert report.approved_rules == 1
    assert report.rules_written == 1
    # Dry run: nothing persisted.
    assert report.jobs_tagged == 0
    assert store.tagged == []
    assert not os.path.exists(rules_path)


def test_full_audit_persists_when_not_dry_run(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_YES_ALL", "1")
    store = FakeStore({
        filter_qa.PASSED_STATUS: PASSED,
        filter_qa.DISQUALIFIED_STATUS: DISQUALIFIED,
    })
    rules_path = str(tmp_path / "filter_rules.yaml")
    qa = FilterQA(store, FakeOllama(), FakeClaude(),
                  dry_run=False, rules_path=rules_path)
    report = qa.run()

    assert report.rules_written == 1
    assert os.path.exists(rules_path)
    # confirmed miss -> flagged, kept none, FP restored -> three tag groups (2 with rows)
    statuses = {status for _, status, _, _ in store.tagged}
    assert filter_qa.FLAGGED_STATUS in statuses
    assert filter_qa.PASSED_STATUS in statuses  # restored false positive
    assert report.jobs_tagged >= 2
