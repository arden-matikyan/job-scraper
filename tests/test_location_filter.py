"""Non-US location filter: deny-list, fail-open.

Covers the pure `non_us_location` helper and its integration into the runner's
pre-LLM `_ingest` stage. No live model or network.
"""
from __future__ import annotations

import os

import pytest

from pipeline.runner import Runner
from scrapers.base import RawJob, non_us_location
from storage.job_store import JobStore


# --------------------------------------------------------------------------- #
# Pure helper
# --------------------------------------------------------------------------- #
KEEP_CASES = [
    "Indianapolis, IN, USA",     # the substring trap: 'india' must NOT fire
    "Indianapolis, Indiana",     # state name, no country
    "Austin, TX",                # bare state code
    "Seattle, Washington, USA",
    "US, WA, Seattle",           # compact, leading US- prefix-ish / state code
    "Remote - US",
    "United States",
    "San Jose, CA",              # California, not Canada
    "NYC",                       # ambiguous → fail-open keep
    "Multiple Locations",        # ambiguous → keep
    "",                          # empty → keep
]

DROP_CASES = [
    "Bengaluru, Karnataka, IND",
    "London, UK",
    "London, England, GBR",
    "Toronto, Ontario, CAN",
    "Vancouver, British Columbia, CAN",
    "Dublin, IRL",
    "Tokyo, JPN",
    "Sydney, New South Wales, AUS",
    "SG-01-SINGAPORE-005 B1 ~ 5 Tuas Dr 2",   # leading ISO-2 prefix
    "Gurgaon, India",
    "Mexico City, Mexico City, MEX",
]


@pytest.mark.parametrize("loc", KEEP_CASES)
def test_keep(loc):
    assert non_us_location(loc) is False


@pytest.mark.parametrize("loc", DROP_CASES)
def test_drop(loc):
    assert non_us_location(loc) is True


def test_indianapolis_is_not_india():
    # both the US-signal short-circuit and word boundaries must protect it
    assert non_us_location("Indianapolis, IN") is False
    assert non_us_location("Indianapolis") is False  # ambiguous → keep, not 'india'


def test_multi_location_keeps_on_any_us_match():
    # a job listed in both Dublin and New York is a US-eligible job → keep
    assert non_us_location("Dublin, IE", ["Dublin, IE", "New York, NY"]) is False
    # all-foreign multi-location → drop
    assert non_us_location("Dublin, IE", ["Dublin, IE", "London, UK"]) is True


def test_locations_all_only():
    # primary location empty, foreign signal only in locations_all
    assert non_us_location(None, ["Bengaluru, Karnataka, IND"]) is True
    assert non_us_location(None, ["Austin, TX"]) is False


def test_deny_extra_extends_list():
    # a no-country UK county is kept by default (fail-open) ...
    assert non_us_location("Warminster, Wiltshire") is False
    # ... but droppable via location_deny_extra
    assert non_us_location("Warminster, Wiltshire", deny_extra=["Wiltshire"]) is True


# --------------------------------------------------------------------------- #
# Runner integration (pre-LLM)
# --------------------------------------------------------------------------- #
class CountingOllama:
    embed_model = None  # no embeddings needed here

    def __init__(self):
        self.generate_calls = 0

    def generate_json(self, prompt, default=None, system=None):
        self.generate_calls += 1
        return {"title": "ok"}

    def embed(self, text):
        return None


def _runner(tmp_path, us_only):
    store = JobStore(os.path.join(str(tmp_path), "loc.db"))
    ollama = CountingOllama()
    runner = Runner(
        ollama=ollama, store=store,
        scraper_configs={"us_only": us_only},  # minimal config, no keyword filter
        embed_jobs=False,
    )
    return runner, ollama, store


def test_runner_drops_foreign_before_llm(tmp_path):
    runner, ollama, store = _runner(tmp_path, us_only=True)
    us = RawJob(source_url="https://x/1", raw_text="role", scraper_key="x",
                title="Software Engineer", company="Acme", location="Austin, TX")
    foreign = RawJob(source_url="https://x/2", raw_text="role", scraper_key="x",
                     title="Software Engineer", company="Acme",
                     location="Bengaluru, Karnataka, IND")

    assert runner._ingest(us, "Acme") == "new"
    assert runner._ingest(foreign, "Acme") == "filtered"
    # the LLM ran exactly once — only for the US job (foreign dropped pre-extraction)
    assert ollama.generate_calls == 1
    assert store.count_jobs() == 1


def test_runner_us_only_off_keeps_foreign(tmp_path):
    runner, ollama, _ = _runner(tmp_path, us_only=False)
    foreign = RawJob(source_url="https://x/3", raw_text="role", scraper_key="x",
                     title="Software Engineer", company="Acme",
                     location="Bengaluru, Karnataka, IND")
    assert runner._ingest(foreign, "Acme") == "new"  # filter disabled
    assert ollama.generate_calls == 1
