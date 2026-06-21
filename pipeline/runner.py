"""Pipeline runner: resolve a scraper per URL, scrape, extract, embed, store.

For each tracked company: recon resolves a scraper (or skips with a logged reason),
the scraper yields RawJobs, the extractor + embedder turn each into a record, and
the store dedups + saves. A rich summary table is printed at the end. Every job is
wrapped in try/except so one failure can't abort the run.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.table import Table

from agent.job_extractor import JobExtractor
from agent.llm import LLMClient, get_llm_client
from agent.recon_agent import (
    PlatformKB,
    ReconAgent,
    ReconResult,
    ReconStatus,
    config_path,
    load_yaml,
)
from scrapers.base import DEFAULT_USER_AGENT, HttpClient
from scrapers.registry import registry as default_registry
from storage.job_store import JobStore, compute_hash

logger = logging.getLogger(__name__)

_EMBED_CHARS = 2000


def load_tracked_urls() -> dict:
    return load_yaml(config_path("tracked_urls.yaml"))


def load_scraper_configs() -> dict:
    return load_yaml(config_path("scraper_configs.yaml"))


@dataclass
class RunResult:
    name: str
    url: str
    scraper_key: Optional[str]
    found: int
    new: int
    filtered: int
    seconds: float
    status: str


class Runner:
    def __init__(
        self,
        ollama: Optional[LLMClient] = None,
        store: Optional[JobStore] = None,
        kb: Optional[PlatformKB] = None,
        http: Optional[HttpClient] = None,
        scraper_configs: Optional[dict] = None,
        console: Optional[Console] = None,
        embed_jobs: Optional[bool] = None,
        extract_workers: Optional[int] = None,
        keyword_filter: Optional[list[str]] = None,
    ):
        self.scraper_configs = scraper_configs if scraper_configs is not None else load_scraper_configs()
        defaults = (self.scraper_configs or {}).get("defaults", {}) or {}
        self.http = http or HttpClient(
            timeout=float(defaults.get("request_timeout", 30)),
            max_retries=int(defaults.get("max_retries", 3)),
            user_agent=defaults.get("user_agent", DEFAULT_USER_AGENT),
        )
        self.ollama = ollama or get_llm_client()
        self.store = store or JobStore()
        self.kb = kb or PlatformKB()
        self.recon = ReconAgent(http=self.http, ollama=self.ollama, kb=self.kb)
        self.extractor = JobExtractor(self.ollama)
        self.registry = default_registry
        self.console = console or Console()
        # speed knobs (overridable per-run): skip unused embeddings, and run
        # extraction across a bounded thread pool so multiple jobs hit the model
        # at once (set OLLAMA_NUM_PARALLEL on the server to actually parallelize).
        self.embed_jobs = bool(defaults.get("embed_jobs", True)) if embed_jobs is None else bool(embed_jobs)
        # Providers without an embedding model (e.g. Claude) can't embed — skip it.
        if getattr(self.ollama, "embed_model", None) is None:
            self.embed_jobs = False
        self.extract_workers = (
            int(defaults.get("extract_workers", 4)) if extract_workers is None else int(extract_workers)
        )
        # Keyword pre-filter: a NEW job whose title/description contains none of
        # these (case-insensitive substring) is skipped before any LLM work. Empty
        # list disables it. Default comes from scraper_configs top-level keyword_filter.
        _kw = keyword_filter if keyword_filter is not None else ((self.scraper_configs or {}).get("keyword_filter") or [])
        self.keyword_filter = [str(k).lower() for k in _kw if str(k).strip()]
        # Title hard-exclude: jobs whose title contains any of these whole words
        # (case-insensitive) are dropped before any LLM work.
        _te = (self.scraper_configs or {}).get("title_exclude") or []
        self._title_exclude_pattern = (
            re.compile(
                r"\b(?:" + "|".join(re.escape(str(t)) for t in _te if str(t).strip()) + r")\b",
                re.IGNORECASE,
            )
            if _te
            else None
        )

    # ------------------------------------------------------------------- run
    def run(self, entries: list[dict]) -> list[RunResult]:
        results: list[RunResult] = []
        for entry in entries or []:
            if entry.get("skip"):
                name = entry.get("name", "?")
                self.console.print(f"[dim]-- {name} skipped (skip: true)[/]")
                continue
            try:
                results.append(self._run_one(entry))
            except Exception as exc:  # a company failure never aborts the run
                name = entry.get("name", "?")
                logger.error("Unexpected failure for %s: %s", name, exc)
                results.append(RunResult(name, entry.get("url", ""), None, 0, 0, 0, 0.0, "ERROR"))
        self._print_summary(results)
        return results

    def _config_for(self, scraper_key: str) -> dict:
        defaults = (self.scraper_configs or {}).get("defaults", {}) or {}
        specific = (self.scraper_configs or {}).get(scraper_key, {}) or {}
        return {**defaults, **specific}

    def _run_one(self, entry: dict) -> RunResult:
        name = entry.get("name") or ""
        url = entry.get("url") or ""
        t0 = time.time()
        self.console.print(f"[bold cyan]>>[/] {name} — {url}")

        # Allow tracked_urls.yaml to pin a scraper and skip recon entirely.
        forced_key = entry.get("scraper_key") or None
        if forced_key:
            recon = ReconResult(url, forced_key, forced_key, 1.0,
                                ReconStatus.MAPPED, f"forced via tracked_urls ({forced_key})")
            self.console.print(f"   [dim]scraper forced: {forced_key}[/]")
        else:
            recon = self.recon.investigate(url, company_name=name)
        if recon.status != ReconStatus.MAPPED or not recon.scraper_key:
            self.store.log_recon(url, recon.platform, recon.scraper_key, 0,
                                 f"{recon.status}: {recon.notes}")
            self.console.print(f"   [yellow]SKIP[/] {recon.status}: {recon.notes}")
            return RunResult(name, url, recon.scraper_key, 0, 0, 0, time.time() - t0, recon.status)

        cfg = self._config_for(recon.scraper_key)
        _ENTRY_META = {"name", "url", "scraper_key", "notes", "skip"}
        entry_extras = {k: v for k, v in entry.items() if k not in _ENTRY_META}
        cfg = {**cfg, **entry_extras}
        scraper = self.registry.get(recon.scraper_key, http=self.http, config=cfg)
        if scraper is None:
            self.store.log_recon(url, recon.platform, recon.scraper_key, 0, "instantiation failed")
            return RunResult(name, url, recon.scraper_key, 0, 0, 0, time.time() - t0, "ERROR")

        # Give the scraper the set of already-saved URLs so detail-fetch scrapers
        # can skip re-downloading job pages we already have (huge win on reruns).
        try:
            scraper.seen_urls = self.store.all_source_urls()
        except Exception as exc:
            logger.warning("could not load seen_urls: %s", exc)

        self.console.print(f"   scraper: [green]{recon.scraper_key}[/] — scraping…")
        found, new, filtered = self._scrape_and_ingest(scraper, url, name)

        if found:
            self.recon.record_success(url, recon.platform)  # Stage 6 KB update
        self.store.log_recon(url, recon.platform, recon.scraper_key, found, recon.notes)
        elapsed = time.time() - t0
        extra = f", {filtered} filtered" if filtered else ""
        self.console.print(f"   [green]done[/] {found} found, {new} new{extra} in {elapsed:.1f}s")
        return RunResult(name, url, recon.scraper_key, found, new, filtered, elapsed, recon.status)

    def _ingest(self, raw, company_name: str) -> str:
        # Returns "new" | "duplicate" | "filtered".
        # Cheap dedup FIRST. The hash uses only scraper-known fields (company,
        # title, source_url) — all available before any LLM call — so seen jobs
        # skip the expensive extraction + embedding entirely. This is what makes
        # daily/repeat runs fast: only genuinely new postings pay the LLM cost.
        company = raw.company or company_name
        hash_value = compute_hash(company, raw.title, raw.source_url)
        if self.store.is_seen(hash_value):
            return "duplicate"

        # Title hard-exclude: drop seniority/management titles before any LLM work.
        if self._title_exclude_pattern and self._title_excluded(raw):
            return "filtered"

        # Keyword pre-filter: skip the LLM entirely for a NEW job whose listing
        # (title + description) mentions none of the configured keywords.
        if self.keyword_filter and not self._matches_keywords(raw):
            return "filtered"

        hints = raw.authoritative_fields()
        if not hints.get("company"):
            hints["company"] = company_name
        record = self.extractor.extract(raw.raw_text, raw.source_url, hints=hints)
        record["scraper_key"] = raw.scraper_key
        record["platform"] = raw.platform or raw.scraper_key
        record["hash"] = hash_value  # reuse the same hash for the insert (stay consistent)
        if not record.get("company"):
            record["company"] = company_name
        emb_text = ((record.get("title") or "") + "\n\n" + (record.get("description_full") or "")).strip()
        record["embedding"] = (
            self.ollama.embed(emb_text[:_EMBED_CHARS]) if (self.embed_jobs and emb_text) else None
        )
        return "new" if self.store.save_job(record) else "duplicate"

    def _title_excluded(self, raw) -> bool:
        title = getattr(raw, "title", "") or ""
        return bool(self._title_exclude_pattern and self._title_exclude_pattern.search(title))

    def _matches_keywords(self, raw) -> bool:
        title = f" {(getattr(raw, 'title', '') or '').lower()} "
        return any(kw in title for kw in self.keyword_filter)

    def _safe_ingest(self, raw, company_name: str) -> str:
        try:
            return self._ingest(raw, company_name)
        except Exception as exc:
            logger.warning("ingest failed for %s: %s", getattr(raw, "source_url", "?"), exc)
            return "error"

    def _scrape_and_ingest(self, scraper, url: str, name: str) -> tuple[int, int, int]:
        """Drive a scraper's RawJobs through ingest, sequentially or concurrently.

        With extract_workers > 1, ingest tasks (the expensive LLM extraction) run in
        a bounded thread pool. The scraper generator is still consumed in this thread
        (its per-job HTTP fetches stay ordered); only extraction/embedding/save fan
        out. The store is lock-guarded and the Ollama httpx client is thread-safe, so
        this is safe; dedup stays correct via INSERT OR IGNORE on the unique hash.
        """
        workers = max(1, int(self.extract_workers))
        found = new = filtered = 0

        def tally(status: str) -> None:
            nonlocal new, filtered
            if status == "new":
                new += 1
            elif status == "filtered":
                filtered += 1

        if workers == 1:
            try:
                for raw in scraper.scrape(url, company_name=name):
                    found += 1
                    if getattr(raw, "already_seen", False):
                        continue  # detail fetch skipped; already in DB
                    tally(self._safe_ingest(raw, name))
                    if found % 25 == 0:
                        self.console.print(f"   …{found} found, {new} new, {filtered} filtered")
            except Exception as exc:
                logger.error("scrape failed for %s: %s", url, exc)
            return found, new, filtered

        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        cap = workers * 2  # bound in-flight futures so huge boards don't balloon memory
        inflight: set = set()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest") as pool:
            try:
                for raw in scraper.scrape(url, company_name=name):
                    found += 1
                    if getattr(raw, "already_seen", False):
                        continue  # detail fetch skipped; already in DB
                    inflight.add(pool.submit(self._safe_ingest, raw, name))
                    if len(inflight) >= cap:
                        done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                        for f in done:
                            tally(f.result())
                    if found % 25 == 0:
                        self.console.print(f"   …{found} found, {new}+ new, {filtered}+ filtered")
            except Exception as exc:
                logger.error("scrape failed for %s: %s", url, exc)
            for f in inflight:  # drain remaining
                tally(f.result())
        return found, new, filtered

    def _print_summary(self, results: list[RunResult]) -> None:
        table = Table(title="Scrape Summary")
        table.add_column("Company", style="cyan")
        table.add_column("Scraper")
        table.add_column("Found", justify="right")
        table.add_column("New", justify="right", style="green")
        table.add_column("Filtered", justify="right", style="dim")
        table.add_column("Time", justify="right")
        table.add_column("Status")
        total_found = total_new = total_filtered = 0
        for r in results:
            total_found += r.found
            total_new += r.new
            total_filtered += r.filtered
            status_style = "green" if r.status == ReconStatus.MAPPED else "yellow"
            table.add_row(
                r.name, r.scraper_key or "-", str(r.found), str(r.new), str(r.filtered),
                f"{r.seconds:.1f}s", f"[{status_style}]{r.status}[/]",
            )
        table.add_section()
        table.add_row("[bold]TOTAL[/]", "", f"[bold]{total_found}[/]", f"[bold]{total_new}[/]",
                      f"[bold]{total_filtered}[/]", "", "")
        self.console.print(table)
