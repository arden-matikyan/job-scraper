"""Keyword pre-filter: a new job without a keyword skips extraction entirely.

Uses a counting fake Ollama so we can prove the LLM is NOT called for filtered
jobs. No live model or network.
"""
from __future__ import annotations

import os

from pipeline.runner import Runner
from scrapers.base import RawJob
from storage.job_store import JobStore


class CountingOllama:
    def __init__(self):
        self.generate_calls = 0
        self.embed_calls = 0

    def generate_json(self, prompt, default=None, system=None):
        self.generate_calls += 1
        return {"title": "ok"}

    def embed(self, text):
        self.embed_calls += 1
        return [0.1, 0.2]


def _runner(tmp_path, keywords):
    store = JobStore(os.path.join(str(tmp_path), "kw.db"))
    ollama = CountingOllama()
    runner = Runner(
        ollama=ollama, store=store, scraper_configs={},  # {} => don't read real config
        keyword_filter=keywords, embed_jobs=True,
    )
    return runner, ollama, store


def test_keyword_filter_skips_nonmatching(tmp_path):
    runner, ollama, store = _runner(tmp_path, ["software", "developer", "engineer"])
    match = RawJob(source_url="https://x/1", raw_text="Backend role building services.",
                   scraper_key="x", title="Software Engineer", company="Acme")
    nomatch = RawJob(source_url="https://x/2", raw_text="Manage logistics and vendors.",
                     scraper_key="x", title="Project Manager", company="Acme")

    assert runner._ingest(match, "Acme") == "new"
    assert runner._ingest(nomatch, "Acme") == "filtered"
    # the LLM + embedding ran exactly once — only for the matching job
    assert ollama.generate_calls == 1
    assert ollama.embed_calls == 1
    assert store.count_jobs() == 1


def test_keyword_match_in_description_only(tmp_path):
    runner, ollama, _ = _runner(tmp_path, ["software"])
    job = RawJob(source_url="https://x/3", raw_text="You will write software every day.",
                 scraper_key="x", title="Member of Technical Staff", company="Acme")
    assert runner._ingest(job, "Acme") == "new"   # keyword in description still matches
    assert ollama.generate_calls == 1


def test_empty_filter_processes_everything(tmp_path):
    runner, ollama, _ = _runner(tmp_path, [])
    job = RawJob(source_url="https://x/4", raw_text="Manage logistics.",
                 scraper_key="x", title="Project Manager", company="Acme")
    assert runner._ingest(job, "Acme") == "new"   # no filter => not skipped
    assert ollama.generate_calls == 1


def test_scrape_and_ingest_counts_filtered(tmp_path):
    runner, ollama, _ = _runner(tmp_path, ["engineer"])
    runner.extract_workers = 1  # sequential for a deterministic tally

    class FakeScraper:
        def scrape(self, url, company_name=None, config=None):
            yield RawJob(source_url="u1", raw_text="x", scraper_key="f",
                         title="Software Engineer", company=company_name)
            yield RawJob(source_url="u2", raw_text="x", scraper_key="f",
                         title="Recruiter", company=company_name)

    found, new, filtered = runner._scrape_and_ingest(FakeScraper(), "http://x", "Acme")
    assert (found, new, filtered) == (2, 1, 1)
