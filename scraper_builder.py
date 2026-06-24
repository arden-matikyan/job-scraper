"""scraper_builder.py — agentic builder for a new career-page scraper.

Given a target career URL, it: runs recon, asks Claude Haiku to write a BaseScraper
subclass (using real scrapers from this repo as few-shot examples), runs the draft
live, captures the HTTP traffic into a replay fixture, re-runs against the fixture to
prove the result is stable, then — only after the human confirms the job count and a
sample — registers the scraper to its final filename and ``config/tracked_urls.yaml``.

    python scraper_builder.py <url> [--name COMPANY] [--debug]

The open-ended part (write code -> run -> read error -> rewrite) is a genuine agentic
loop, so it runs through ``agent/tool_runner.py`` with Claude Haiku. The surrounding
flow is an explicit ``BuildState`` machine with human gates. Drafts are always written
to ``scrapers/_draft_*.py`` and are only renamed to the final path after confirmation.

Constraint: this script imports only from ``agent/`` and ``scrapers/`` — never from
``pipeline/``, ``storage/`` or ``filter/``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

from rich.console import Console

from agent.human_input import ask_human, yes_all_enabled
from agent.tool_runner import MaxIterationsExceeded, ToolDef, ToolRunner
from scrapers.base import BaseScraper, HttpClient
from scrapers.registry import registry

logger = logging.getLogger(__name__)
console = Console()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRAPERS_DIR = os.path.join(PROJECT_ROOT, "scrapers")
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
TRACKED_URLS_PATH = os.path.join(PROJECT_ROOT, "config", "tracked_urls.yaml")

EXAMPLES_CHAR_CAP = 12_000
RUN_TIMEOUT_SECONDS = 60
MAX_JOBS_COLLECTED = 200
MAX_DEBUG_ATTEMPTS = 3
STABLE_DELTA = 2
_XHR_PATTERNS = ("/api/", "jobs.json", "/search")


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
class BuildState(Enum):
    SUBMITTED = "submitted"
    RECONNING = "reconning"
    AWAITING_RECON_CONFIRM = "awaiting_recon_confirm"
    WRITING = "writing"
    TESTING_LIVE = "testing_live"
    AWAITING_JOB_COUNT = "awaiting_job_count"
    SAVING_FIXTURE = "saving_fixture"
    TESTING_FIXTURE = "testing_fixture"
    AWAITING_SAMPLE_CONFIRM = "awaiting_sample_confirm"
    REGISTERING = "registering"
    DONE = "done"
    DEBUGGING = "debugging"
    ESCALATED = "escalated"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return slug or "scraper"


def name_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    host = re.sub(r"^www\.", "", host)
    first = host.split(".")[0] if host else "scraper"
    return slugify(first)


# --------------------------------------------------------------------------- #
# Scraper writer system prompt
# --------------------------------------------------------------------------- #
SCRAPER_WRITER_SYSTEM = """\
You are an expert Python developer building web scrapers for a job aggregation system.

Your task: given a target URL and recon results, write a complete Python scraper class
that subclasses BaseScraper and implements the scrape() method returning Iterator[RawJob].

Rules:
1. Always subclass BaseScraper from scrapers/base.py
2. Always define SCRAPER_KEY as a class attribute (lowercase_underscore)
3. Always define SITE_HINTS as a list of URL substrings
4. The scrape() method must be a generator (use yield, not return list)
5. Use self.http (the shared HttpClient) for all HTTP requests — never instantiate httpx directly
6. Handle pagination — never return only the first page
7. Construct RawJob with at minimum: title, company, url (source_url), source_url
8. Never use external libraries beyond what's already imported in the examples

Workflow you must follow with the tools provided:
- Call fetch_scraper_examples first to see real scrapers from this codebase.
- Call write_scraper with the full module source (it writes scrapers/_draft_<name>.py).
- Call run_scraper to execute the draft live and inspect the count/sample/errors.
- If it fails or returns zero jobs, debug and call write_scraper again, then run_scraper.
- When the draft returns a reasonable number of jobs, stop (end your turn) with a short
  summary. Do NOT register anything — a human handles confirmation and registration.

When debugging: you receive the original code, the error or mismatch description, and
the raw HTML/JSON the scraper attempted to parse. Identify the root cause precisely
before rewriting.
"""


# --------------------------------------------------------------------------- #
# Shared build context (tools mutate this)
# --------------------------------------------------------------------------- #
@dataclass
class BuildContext:
    url: str
    name: str
    recon: dict = field(default_factory=dict)
    draft_path: Optional[str] = None
    last_code: Optional[str] = None
    last_run: Optional[dict] = None
    raw_responses: list[dict] = field(default_factory=list)
    attempts: int = 0
    good: bool = False


# --------------------------------------------------------------------------- #
# Tool: recon
# --------------------------------------------------------------------------- #
def recon(url: str, http: Optional[HttpClient] = None, ollama=None) -> dict:
    """Fetch the page, run the existing ReconAgent, and sniff for XHR/API patterns."""
    from agent.recon_agent import ReconAgent

    http = http or HttpClient()
    page = http.get_text(url)
    signals: list[str] = []
    low = (page or "").lower()
    for pat in _XHR_PATTERNS:
        if pat in low:
            signals.append(f"xhr pattern {pat!r} present in page source")

    agent = ReconAgent(http=http, ollama=ollama)
    try:
        result = agent.investigate(url)
        platform = result.platform
        confidence = result.confidence
        suggested = result.scraper_key
        if result.notes:
            signals.append(result.notes)
    except Exception as exc:  # recon must never crash the build
        logger.warning("recon investigate failed: %s", exc)
        platform, confidence, suggested = None, 0.0, None
        signals.append(f"recon error: {exc}")

    return {
        "platform": platform,
        "confidence": confidence,
        "signals": signals,
        "suggested_scraper_key": suggested,
    }


# --------------------------------------------------------------------------- #
# Tool: fetch_scraper_examples
# --------------------------------------------------------------------------- #
def _read_scraper(filename: str) -> Optional[dict]:
    path = os.path.join(SCRAPERS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return {"filename": filename, "content": f.read()}
    except Exception as exc:
        logger.warning("Could not read example %s: %s", filename, exc)
        return None


def fetch_scraper_examples(platform: Optional[str] = None) -> dict:
    """Load a few real scrapers as few-shot examples, capped at 12k chars total."""
    wanted: list[str] = []

    key = (platform or "").lower()
    # If recon named a platform we recognise, lead with that scraper file.
    if key:
        registry.discover()
        for scraper_key in registry.all_keys():
            if key in scraper_key.lower() or scraper_key.lower() in key:
                cls = registry.get_class(scraper_key)
                if cls is not None:
                    mod = getattr(cls, "__module__", "")
                    fname = mod.split(".")[-1] + ".py"
                    wanted.append(fname)
                    break

    # Always include a representative API scraper and a Playwright scraper, plus the
    # static fallback, so the model sees the range of patterns.
    for default in ("greenhouse_scraper.py", "javascript_scraper.py", "static_html_scraper.py"):
        if default not in wanted:
            wanted.append(default)

    examples: list[dict] = []
    total = 0
    for fname in wanted:
        ex = _read_scraper(fname)
        if ex is None:
            continue
        content = ex["content"]
        if total + len(content) > EXAMPLES_CHAR_CAP:
            remaining = max(0, EXAMPLES_CHAR_CAP - total)
            content = content[:remaining] + "\n# ... truncated\n"
        ex["content"] = content
        examples.append(ex)
        total += len(content)
        if total >= EXAMPLES_CHAR_CAP:
            break
    return {"examples": examples}


# --------------------------------------------------------------------------- #
# Tool: write_scraper
# --------------------------------------------------------------------------- #
def _draft_path(name: str) -> str:
    return os.path.join(SCRAPERS_DIR, f"_draft_{slugify(name)}.py")


def write_scraper(code: str, name: str) -> dict:
    """Validate then write the draft to scrapers/_draft_<name>.py."""
    errors: list[str] = []
    if "class " not in code:
        errors.append("no class definition found")
    if "SCRAPER_KEY" not in code:
        errors.append("SCRAPER_KEY class attribute missing")
    if not re.search(r"def\s+scrape\s*\(", code):
        errors.append("scrape() method missing")
    try:
        compile(code, "<draft>", "exec")
    except SyntaxError as exc:
        errors.append(f"syntax error: {exc}")

    path = _draft_path(name)
    if errors:
        return {"path": path, "valid": False, "validation_errors": errors}

    os.makedirs(SCRAPERS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return {"path": path, "valid": True, "validation_errors": []}


# --------------------------------------------------------------------------- #
# Tool: run_scraper
# --------------------------------------------------------------------------- #
def _load_scraper_class(filepath: str):
    """Import a draft module fresh (unique name dodges import caching) and find its class."""
    modname = f"_draftmod_{slugify(os.path.basename(filepath))}_{int(time.time()*1000)}"
    spec = importlib.util.spec_from_file_location(modname, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseScraper)
            and attr is not BaseScraper
        ):
            return attr
    raise ImportError("no BaseScraper subclass found in draft")


def _job_to_dict(job: Any) -> dict:
    """Compact RawJob -> dict for samples (full raw_text would bloat the LLM context)."""
    raw = getattr(job, "raw_text", "") or ""
    return {
        "title": getattr(job, "title", None),
        "company": getattr(job, "company", None),
        "location": getattr(job, "location", None),
        "source_url": getattr(job, "source_url", None),
        "raw_text_snippet": raw[:200],
    }


def run_scraper(filepath: str, url: str) -> dict:
    """Import + run the draft live, capturing HTTP responses, with a 60s hard timeout."""
    import scrapers.base as sbase

    recorded: list[dict] = []
    orig_request = sbase.HttpClient.request

    def _recording_request(self, method, req_url, **kwargs):
        resp = orig_request(self, method, req_url, **kwargs)
        try:
            recorded.append({
                "url": str(req_url),
                "status": getattr(resp, "status_code", 0) if resp is not None else 0,
                "headers": dict(resp.headers) if resp is not None else {},
                "body": resp.text if resp is not None else "",
            })
        except Exception:  # capture is best-effort, never break the run
            pass
        return resp

    state: dict[str, Any] = {"jobs": [], "error": None}

    def _worker():
        try:
            inst = _load_scraper_class(filepath)()
            for job in inst.scrape(url):
                state["jobs"].append(job)
                if len(state["jobs"]) >= MAX_JOBS_COLLECTED:
                    break
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"

    sbase.HttpClient.request = _recording_request
    started = time.time()
    try:
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join(RUN_TIMEOUT_SECONDS)
        timed_out = worker.is_alive()
    finally:
        sbase.HttpClient.request = orig_request

    duration = round(time.time() - started, 2)
    jobs = state["jobs"]
    errors: list[str] = []
    if state["error"]:
        errors.append(state["error"])
    if timed_out:
        errors.append(f"timed out after {RUN_TIMEOUT_SECONDS}s (collected {len(jobs)} so far)")

    return {
        "count": len(jobs),
        "sample": [_job_to_dict(j) for j in jobs[:3]],
        "errors": errors,
        "duration_seconds": duration,
        "raw_responses": recorded,
    }


# --------------------------------------------------------------------------- #
# Tool: save_fixture
# --------------------------------------------------------------------------- #
def save_fixture(name: str, raw_responses: list[dict], url: str = "") -> dict:
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    path = os.path.join(FIXTURES_DIR, f"{slugify(name)}.json")
    payload = {
        "url": url,
        "captured_at": _now(),
        "responses": raw_responses or [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"path": path, "response_count": len(raw_responses or [])}


# --------------------------------------------------------------------------- #
# Tool: run_fixture_test
# --------------------------------------------------------------------------- #
_UNSAFE_REPLAY_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}


def run_fixture_test(filepath: str, fixture_path: str, live_count: Optional[int] = None) -> dict:
    """Replay the fixture's HTTP responses through a patched httpx and compare counts.

    ``stable`` is True when the replayed count is within ±2 of the live count.
    """
    import httpx

    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)
    responses = fixture.get("responses", []) or []
    by_url: dict[str, dict] = {}
    for r in responses:
        by_url.setdefault(str(r.get("url")), r)

    def _make_response(method: str, req_url: str) -> "httpx.Response":
        entry = by_url.get(str(req_url))
        request = httpx.Request(method, req_url)
        if entry is None:
            return httpx.Response(404, request=request)
        headers = {
            k: v for k, v in (entry.get("headers") or {}).items()
            if k.lower() not in _UNSAFE_REPLAY_HEADERS
        }
        body = (entry.get("body") or "").encode("utf-8")
        return httpx.Response(int(entry.get("status") or 200), headers=headers,
                              content=body, request=request)

    orig_request = httpx.Client.request
    orig_get = httpx.Client.get

    def _fake_request(self, method, req_url, **kwargs):
        return _make_response(method, req_url)

    def _fake_get(self, req_url, **kwargs):
        return _make_response("GET", req_url)

    httpx.Client.request = _fake_request
    httpx.Client.get = _fake_get
    try:
        inst = _load_scraper_class(filepath)()
        count = 0
        for _ in inst.scrape(fixture.get("url", "")):
            count += 1
            if count >= MAX_JOBS_COLLECTED:
                break
    finally:
        httpx.Client.request = orig_request
        httpx.Client.get = orig_get

    delta = count - (live_count if live_count is not None else count)
    return {"count": count, "delta": delta, "stable": abs(delta) <= STABLE_DELTA}


# --------------------------------------------------------------------------- #
# Tool: register_scraper
# --------------------------------------------------------------------------- #
def _extract_scraper_key(code: str) -> Optional[str]:
    m = re.search(r"""SCRAPER_KEY\s*[:=]\s*["']([^"']+)["']""", code)
    return m.group(1) if m else None


def register_scraper(draft_path: str, name: str, url: str) -> dict:
    """Rename the draft to its final filename and append an entry to tracked_urls.yaml."""
    if not os.path.exists(draft_path):
        return {"final_path": None, "yaml_updated": False,
                "error": f"draft not found: {draft_path}"}

    with open(draft_path, encoding="utf-8") as f:
        code = f.read()
    scraper_key = _extract_scraper_key(code) or slugify(name)

    final_path = os.path.join(SCRAPERS_DIR, f"{slugify(name)}.py")
    os.replace(draft_path, final_path)

    yaml_updated = _append_tracked_url(name, url, scraper_key)
    return {"final_path": final_path, "yaml_updated": yaml_updated,
            "scraper_key": scraper_key}


def _append_tracked_url(name: str, url: str, scraper_key: str) -> bool:
    """Append a company entry to config/tracked_urls.yaml (textual, preserves the file)."""
    entry = (
        f"- name: {name}\n"
        f"  url: {url}\n"
        f"  scraper_key: {scraper_key}\n"
    )
    try:
        existing = ""
        if os.path.exists(TRACKED_URLS_PATH):
            with open(TRACKED_URLS_PATH, encoding="utf-8") as f:
                existing = f.read()
        body = existing.rstrip() + "\n" + entry if existing.strip() else entry
        with open(TRACKED_URLS_PATH, "w", encoding="utf-8") as f:
            f.write(body)
        return True
    except Exception as exc:
        logger.error("Could not append to tracked_urls.yaml: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Builder orchestrator
# --------------------------------------------------------------------------- #
class ScraperBuilder:
    def __init__(self, url: str, name: Optional[str] = None, debug: bool = False,
                 llm_client=None, recon_ollama=None):
        self.ctx = BuildContext(url=url, name=name or name_from_url(url))
        self.debug = debug
        self.llm = llm_client
        self.recon_ollama = recon_ollama
        self.state = BuildState.SUBMITTED
        self.tool_calls: list[dict] = []

    def _to(self, state: BuildState) -> None:
        self.state = state
        console.print(f"[dim][{_now()}] STATE -> {state.value}[/]")
        logger.info("STATE -> %s", state.value)

    # ------------------------------------------------------- agentic sub-loop
    def _agentic_tooldefs(self) -> list[ToolDef]:
        ctx = self.ctx

        def _fetch(platform: Optional[str] = None) -> dict:
            return fetch_scraper_examples(platform or (ctx.recon.get("platform")))

        def _write(code: str, name: Optional[str] = None) -> dict:
            self._to(BuildState.WRITING)
            ctx.last_code = code
            result = write_scraper(code, name or ctx.name)
            if result["valid"]:
                ctx.draft_path = result["path"]
            return result

        def _run(filepath: Optional[str] = None, url: Optional[str] = None) -> dict:
            self._to(BuildState.TESTING_LIVE)
            ctx.attempts += 1
            path = filepath or ctx.draft_path
            if not path:
                return {"count": 0, "errors": ["no draft written yet — call write_scraper first"]}
            result = run_scraper(path, url or ctx.url)
            ctx.last_run = result
            ctx.raw_responses = result.get("raw_responses", [])
            if result["count"] > 0 and not result["errors"]:
                ctx.good = True
            else:
                self._to(BuildState.DEBUGGING)
            # Trim the bulky raw_responses out of what the model sees.
            visible = {k: v for k, v in result.items() if k != "raw_responses"}
            if ctx.attempts >= MAX_DEBUG_ATTEMPTS and not ctx.good:
                visible["notice"] = (
                    f"Reached {MAX_DEBUG_ATTEMPTS} attempts. If still failing, stop and "
                    "summarize the blocker so a human can take over."
                )
            return visible

        def _ask(question: str, context: Optional[dict] = None) -> dict:
            return {"response": ask_human(question, context)}

        return [
            ToolDef("fetch_scraper_examples",
                    "Load real scrapers from this repo as few-shot examples.",
                    {"type": "object", "properties": {"platform": {"type": "string"}}},
                    _fetch),
            ToolDef("write_scraper",
                    "Write the scraper module to scrapers/_draft_<name>.py (validated).",
                    {"type": "object", "required": ["code"], "properties": {
                        "code": {"type": "string"}, "name": {"type": "string"}}},
                    _write),
            ToolDef("run_scraper",
                    "Run the current draft live and return count/sample/errors.",
                    {"type": "object", "properties": {
                        "filepath": {"type": "string"}, "url": {"type": "string"}}},
                    _run),
            ToolDef("ask_human", "Ask the human operator for guidance.",
                    {"type": "object", "required": ["question"], "properties": {
                        "question": {"type": "string"}, "context": {"type": "object"}}},
                    _ask),
        ]

    def _run_agentic_write(self) -> None:
        """Drive the write/run/debug loop with Claude Haiku via ToolRunner."""
        runner = ToolRunner(
            self.llm,
            self._agentic_tooldefs(),
            system_prompt=SCRAPER_WRITER_SYSTEM,
            max_iterations=2 + MAX_DEBUG_ATTEMPTS * 3,
            verbose=self.debug,
            console=console,
        )
        user_message = (
            f"Build a scraper named '{self.ctx.name}' for this career page.\n"
            f"URL: {self.ctx.url}\n"
            f"Recon results: {json.dumps(self.ctx.recon, default=str)}\n\n"
            "Fetch examples, write the draft, run it, and debug until it yields jobs. "
            "Then stop with a one-line summary."
        )
        try:
            result = runner.run(user_message)
            self.tool_calls = result.tool_calls
        except MaxIterationsExceeded as exc:
            self.tool_calls = exc.tool_calls
            logger.warning("Agentic write loop hit max iterations: %s", exc)

    # ----------------------------------------------------------------- run
    def run(self) -> BuildState:
        self._to(BuildState.SUBMITTED)

        # --- recon -------------------------------------------------------
        self._to(BuildState.RECONNING)
        self.ctx.recon = recon(self.ctx.url, ollama=self.recon_ollama)
        console.print(f"Recon: {json.dumps(self.ctx.recon, default=str, indent=2)}")

        self._to(BuildState.AWAITING_RECON_CONFIRM)
        answer = ask_human(
            "Proceed to write a scraper based on this recon?",
            context=self.ctx.recon,
            options=["proceed", "abort"],
        )
        if answer == "abort":
            console.print("[yellow]Aborted by human at recon confirm.[/]")
            self._to(BuildState.ESCALATED)
            return self.state

        # --- agentic write + live test ----------------------------------
        if self.llm is None:
            from agent.anthropic_client import AnthropicClient
            self.llm = AnthropicClient()
        self._run_agentic_write()

        if not self.ctx.good:
            return self._escalate("draft never produced jobs within the debug budget")

        # --- confirm job count ------------------------------------------
        self._to(BuildState.AWAITING_JOB_COUNT)
        count = self.ctx.last_run["count"]
        answer = ask_human(
            f"The draft scraped {count} jobs. Does that look right?",
            context={"count": count, "sample": self.ctx.last_run.get("sample", [])},
            options=["yes", "no"],
        )
        if answer == "no":
            return self._escalate("human rejected the live job count")

        # --- fixture capture + replay -----------------------------------
        self._to(BuildState.SAVING_FIXTURE)
        fixture = save_fixture(self.ctx.name, self.ctx.raw_responses, self.ctx.url)
        console.print(f"Saved fixture with {fixture['response_count']} response(s).")

        self._to(BuildState.TESTING_FIXTURE)
        replay = run_fixture_test(self.ctx.draft_path, fixture["path"], live_count=count)
        console.print(f"Fixture replay: count={replay['count']} delta={replay['delta']} "
                      f"stable={replay['stable']}")
        if not replay["stable"]:
            console.print("[yellow]Warning: fixture replay count is not stable (±2).[/]")

        # --- confirm sample + register ----------------------------------
        self._to(BuildState.AWAITING_SAMPLE_CONFIRM)
        answer = ask_human(
            "Register this scraper to its final filename and tracked_urls.yaml?",
            context={"sample": self.ctx.last_run.get("sample", []),
                     "fixture_stable": replay["stable"]},
            options=["register", "abort"],
        )
        if answer == "abort":
            console.print("[yellow]Left draft in place; not registered.[/]")
            self._to(BuildState.DONE)
            return self.state

        self._to(BuildState.REGISTERING)
        reg = register_scraper(self.ctx.draft_path, self.ctx.name, self.ctx.url)
        console.print(f"Registered: {json.dumps(reg, default=str)}")

        self._to(BuildState.DONE)
        return self.state

    # ----------------------------------------------------------- escalation
    def _escalate(self, reason: str) -> BuildState:
        self._to(BuildState.ESCALATED)
        os.makedirs(LOGS_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"scraper_builder_{slugify(self.ctx.name)}_{ts}.txt")
        last_run = self.ctx.last_run or {}
        report = [
            f"Scraper builder escalation report",
            f"timestamp: {_now()}",
            f"reason: {reason}",
            f"url: {self.ctx.url}",
            f"name: {self.ctx.name}",
            f"attempts: {self.ctx.attempts}",
            f"recon: {json.dumps(self.ctx.recon, default=str, indent=2)}",
            f"last run count: {last_run.get('count')}",
            f"last run errors: {last_run.get('errors')}",
            "",
            "=== last generated code ===",
            self.ctx.last_code or "(none)",
            "",
            "=== tool call audit trail ===",
            json.dumps(self.tool_calls, default=str, indent=2),
        ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(report))
            console.print(f"[red]Escalated.[/] Debug report written to {path}")
        except Exception as exc:
            console.print(f"[red]Escalated[/] but could not write report: {exc}")
        return self.state


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic scraper builder.")
    parser.add_argument("url", help="The career page URL.")
    parser.add_argument("--name", default=None, help="Company slug override.")
    parser.add_argument("--debug", action="store_true", help="Verbose tool/LLM logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if yes_all_enabled():
        console.print("[yellow]AGENT_YES_ALL=1 — prompts auto-answer with their first option.[/]")

    builder = ScraperBuilder(args.url, name=args.name, debug=args.debug)
    final = builder.run()
    return 0 if final in (BuildState.DONE,) else 1


if __name__ == "__main__":
    raise SystemExit(main())
