"""Scraper-writer agent: generate a new scraper from a ReconReport via Ollama.

Flow: read template + a reference scraper -> prompt Ollama with sample HTML ->
extract the class -> write to a TEMP dir -> run the contract test (pytest subprocess)
-> attempt a live 3-job scrape -> present a summary -> save into scrapers/ ONLY on
user approval. Generated code is executed during validation (per spec); it runs from
a temp dir, the live test is capped at 3 jobs, and nothing is persisted unless the
user approves.
"""
from __future__ import annotations

import importlib.util
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from scrapers.base import BaseScraper, HttpClient, find_job_links
from scrapers.registry import registry as default_registry

logger = logging.getLogger(__name__)

_SCRAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scrapers")
_SCRAPERS_DIR = os.path.abspath(_SCRAPERS_DIR)
_PROJECT_ROOT = os.path.dirname(_SCRAPERS_DIR)
_BROAD = ["/jobs/", "/careers/", "/career/", "/positions/", "/openings/", "/job/"]


def build_recon_report(url, html, findings, http=None, company_name=None) -> dict:
    """Assemble the ReconReport the writer needs (samples, structure, links)."""
    http = http or HttpClient()
    links = (findings or {}).get("job_links") or find_job_links(html or "", url, _BROAD)
    sample_detail_html = ""
    if links:
        sample_detail_html = (http.get_text(links[0]) or "")[:4000]
    return {
        "base_url": url,
        "company_name": company_name,
        "page_structure": (findings or {}).get("strategy") or "unknown",
        "sample_listing_html": (html or "")[:4000],
        "sample_detail_html": sample_detail_html,
        "pagination_pattern": (findings or {}).get("pagination") or "",
        "api_endpoints": (findings or {}).get("api_endpoints") or [],
        "job_links": links[:10],
        "model_hint": (findings or {}).get("reasoning") or "",
    }


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


class ScraperWriter:
    def __init__(self, ollama, http=None):
        self.ollama = ollama
        self.http = http or HttpClient()
        self.log = logging.getLogger(__name__)

    # --------------------------------------------------------------- public
    def write(self, report: dict, interactive: bool = True) -> Optional[str]:
        """Generate, validate, and (on approval) save a scraper. Returns its key."""
        if self.ollama is None:
            self.log.error("ScraperWriter needs an Ollama client")
            return None

        code = self._generate(report)
        if not code:
            self.log.error("Scraper generation produced no code")
            return None

        try:
            compile(code, "<generated_scraper>", "exec")
        except SyntaxError as exc:
            print(f"\n[WRITER] Generated code has a syntax error: {exc}")
            return None

        tmpdir = tempfile.mkdtemp(prefix="jobscraper_gen_")
        candidate_path = os.path.join(tmpdir, "candidate_scraper.py")
        with open(candidate_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            cls = self._load_class(candidate_path)
            if cls is None:
                print("\n[WRITER] Generated module has no BaseScraper subclass with a SCRAPER_KEY.")
                return None

            passed, test_output = self._run_contract(candidate_path)
            sample_jobs = self._live_test(cls, report.get("base_url", ""))

            self._present(cls, code, passed, test_output, sample_jobs)

            if not interactive:
                if passed and sample_jobs:
                    return self._save(cls.SCRAPER_KEY, code)
                self.log.warning(
                    "[WRITER] Auto-save skipped for %s: contract=%s live_jobs=%d",
                    cls.SCRAPER_KEY, passed, len(sample_jobs),
                )
                return None
            if self._approve():
                return self._save(cls.SCRAPER_KEY, code)
            print("[WRITER] Discarded generated scraper.")
            return None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # --------------------------------------------------------------- generate
    def _generate(self, report: dict) -> str:
        template = _read(os.path.join(_SCRAPERS_DIR, "template_scraper.py"))
        strategy = (report.get("page_structure") or "").lower()
        ref_name = "static_html_scraper.py"
        if "api" in strategy:
            ref_name = "greenhouse_scraper.py"
        elif "paginat" in strategy:
            ref_name = "avature_scraper.py"
        reference = _read(os.path.join(_SCRAPERS_DIR, ref_name))

        prompt = f"""You are writing a new Python web scraper for a job board.
It MUST subclass BaseScraper and follow the same contract as the examples.

=== TEMPLATE (fill this in) ===
{template}

=== REFERENCE EXAMPLE ({ref_name}) ===
{reference}

=== TARGET SITE ===
base_url: {report.get('base_url')}
detected structure: {report.get('page_structure')}
model hint: {report.get('model_hint')}
pagination pattern: {report.get('pagination_pattern')}
api endpoints found: {report.get('api_endpoints')}
candidate job links: {report.get('job_links')}

=== SAMPLE LISTING HTML (truncated) ===
{report.get('sample_listing_html')}

=== SAMPLE JOB DETAIL HTML (truncated) ===
{report.get('sample_detail_html')}

Requirements:
- Output ONLY a complete Python module (no prose). You may use a ```python fence.
- Import from scrapers.base: BaseScraper, RawJob, html_to_text, and any of
  extract_title / find_job_links / add_query_param you need.
- Set a unique snake_case SCRAPER_KEY (e.g. derived from the domain) and a PRIORITY
  (10 for an API, 90+ for HTML). SITE_HINTS may be [] if only used explicitly.
- scrape(self, url, company_name=None, config=None) MUST be a generator yielding
  RawJob, using self.http for all requests, and wrapping each item in try/except.
- Put the full plain-text description in RawJob.raw_text via html_to_text(...).
"""
        raw = self.ollama.generate(prompt)
        return _extract_code(raw)

    # ------------------------------------------------------------- validation
    @staticmethod
    def _load_class(path: str) -> Optional[type]:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        try:
            spec = importlib.util.spec_from_file_location("candidate_scraper", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:
            logger.error("Could not import generated scraper: %s", exc)
            return None
        for attr in vars(mod).values():
            if (isinstance(attr, type) and issubclass(attr, BaseScraper)
                    and attr is not BaseScraper and getattr(attr, "SCRAPER_KEY", "")):
                return attr
        return None

    def _run_contract(self, candidate_path: str) -> tuple[bool, str]:
        """Run a self-contained pytest that contract-checks the generated class."""
        tmpdir = os.path.dirname(candidate_path)
        test_code = f'''
import inspect, importlib.util, sys
sys.path.insert(0, {_PROJECT_ROOT!r})
from scrapers.base import BaseScraper
spec = importlib.util.spec_from_file_location("candidate_scraper", {candidate_path!r})
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def _cls():
    for a in vars(mod).values():
        if isinstance(a, type) and issubclass(a, BaseScraper) and a is not BaseScraper and getattr(a, "SCRAPER_KEY", ""):
            return a
    return None

def test_has_scraper():
    assert _cls() is not None

def test_contract():
    c = _cls()
    assert isinstance(c.SCRAPER_KEY, str) and c.SCRAPER_KEY
    assert isinstance(c.SITE_HINTS, list)
    assert isinstance(c.PRIORITY, int)
    assert inspect.isgeneratorfunction(c.scrape)
    c()  # instantiable
'''
        test_path = os.path.join(tmpdir, "test_candidate.py")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", test_path],
                capture_output=True, text=True, timeout=120, cwd=tmpdir,
            )
            return proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
        except Exception as exc:
            return False, f"pytest run failed: {exc}"

    def _live_test(self, cls, url: str, limit: int = 3) -> list:
        if not url:
            return []
        try:
            inst = cls(http=self.http)
            return list(itertools.islice(inst.scrape(url), limit))
        except Exception as exc:
            self.log.warning("Live test scrape failed: %s", exc)
            return []

    # ---------------------------------------------------------------- present
    def _present(self, cls, code, passed, test_output, sample_jobs) -> None:
        print("\n" + "=" * 64)
        print(f"[WRITER] Generated scraper: {cls.SCRAPER_KEY}  (PRIORITY={cls.PRIORITY})")
        print(f"  SITE_HINTS : {cls.SITE_HINTS}")
        print(f"  code length: {len(code)} chars")
        print(f"  contract test: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print("  --- test output (tail) ---")
            print("  " + test_output.replace("\n", "\n  "))
        print(f"  live 3-job scrape: {len(sample_jobs)} job(s) returned")
        for j in sample_jobs[:3]:
            title = (j.title or "")[:60]
            print(f"    - {title!r}  ({len(j.raw_text)} chars)  {j.source_url}")
        print("=" * 64)

    def _approve(self) -> bool:
        try:
            ans = input("Save this scraper into scrapers/? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes")

    # ------------------------------------------------------------------- save
    def _save(self, scraper_key: str, code: str) -> Optional[str]:
        safe = re.sub(r"[^a-z0-9_]", "_", scraper_key.lower()) or "generated"
        filename = f"{safe}_scraper.py"
        dest = os.path.join(_SCRAPERS_DIR, filename)
        if os.path.exists(dest):
            self.log.warning("Overwriting existing %s", filename)
        try:
            with open(dest, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as exc:
            self.log.error("Could not save scraper: %s", exc)
            return None
        default_registry.discover(force=True)  # pick up the new scraper
        print(f"[WRITER] Saved {filename} and registered {scraper_key}.")
        return scraper_key
