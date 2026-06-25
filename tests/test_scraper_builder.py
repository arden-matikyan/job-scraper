"""Tests for scraper_builder.py — no network, no real LLM.

Deterministic tools are tested directly. The agentic write/debug loop is driven by
a scripted fake LLM (same response shape as the anthropic SDK), and recon is stubbed.
File-writing tools are redirected at temp dirs via monkeypatch.
"""
from __future__ import annotations

import json
import os

import scraper_builder as sb
from scraper_builder import (
    BuildState,
    ScraperBuilder,
    register_scraper,
    run_fixture_test,
    run_scraper,
    save_fixture,
    write_scraper,
)


# --------------------------------------------------------------------------- #
# Scripted fake LLM (mimics anthropic tool_use / text blocks)
# --------------------------------------------------------------------------- #
class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, id, name, input):  # noqa: A002
        self.id = id
        self.name = name
        self.input = input


class _Resp:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)

    def complete(self, messages, tools=None, system=None):
        return self.scripted.pop(0)


DRAFT_OK = '''\
from scrapers.base import BaseScraper, RawJob


class DemoBuilderScraper(BaseScraper):
    SCRAPER_KEY = "demo_e2e"
    SITE_HINTS = ["demo"]

    def scrape(self, url, company_name=None, config=None):
        for i in range(3):
            yield RawJob(source_url=f"{url}/{i}", raw_text="t",
                         scraper_key=self.SCRAPER_KEY, title=f"Job {i}")
'''

DRAFT_EMPTY = '''\
from scrapers.base import BaseScraper, RawJob


class EmptyScraper(BaseScraper):
    SCRAPER_KEY = "empty_e2e"
    SITE_HINTS = ["x"]

    def scrape(self, url, company_name=None, config=None):
        if False:
            yield None
'''


def _redirect_dirs(monkeypatch, tmp_path):
    scrapers_dir = tmp_path / "scrapers"
    scrapers_dir.mkdir()
    monkeypatch.setattr(sb, "SCRAPERS_DIR", str(scrapers_dir))
    monkeypatch.setattr(sb, "FIXTURES_DIR", str(tmp_path / "fixtures"))
    monkeypatch.setattr(sb, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(sb, "TRACKED_URLS_PATH", str(tmp_path / "tracked_urls.yaml"))
    return scrapers_dir


# --------------------------------------------------------------------------- #
# write_scraper
# --------------------------------------------------------------------------- #
def test_write_scraper_valid(monkeypatch, tmp_path):
    _redirect_dirs(monkeypatch, tmp_path)
    out = write_scraper(DRAFT_OK, "demo")
    assert out["valid"] is True
    assert os.path.exists(out["path"])


def test_write_scraper_rejects_missing_pieces(monkeypatch, tmp_path):
    _redirect_dirs(monkeypatch, tmp_path)
    out = write_scraper("x = 1\n", "bad")
    assert out["valid"] is False
    assert any("class" in e for e in out["validation_errors"])
    assert any("SCRAPER_KEY" in e for e in out["validation_errors"])


def test_write_scraper_rejects_syntax_error(monkeypatch, tmp_path):
    _redirect_dirs(monkeypatch, tmp_path)
    code = "class X:\n  SCRAPER_KEY='x'\n  def scrape(self): yield (\n"
    out = write_scraper(code, "broken")
    assert out["valid"] is False
    assert any("syntax error" in e for e in out["validation_errors"])


# --------------------------------------------------------------------------- #
# run_scraper / save_fixture / run_fixture_test
# --------------------------------------------------------------------------- #
def test_run_scraper_no_network(monkeypatch, tmp_path):
    _redirect_dirs(monkeypatch, tmp_path)
    path = write_scraper(DRAFT_OK, "demo")["path"]
    out = run_scraper(path, "https://demo.test/jobs")
    assert out["count"] == 3
    assert len(out["sample"]) == 3
    assert out["errors"] == []
    assert out["sample"][0]["title"] == "Job 0"


def test_save_fixture(monkeypatch, tmp_path):
    _redirect_dirs(monkeypatch, tmp_path)
    responses = [{"url": "https://x/api", "status": 200, "headers": {}, "body": "{}"}]
    out = save_fixture("demo", responses, "https://x/list")
    assert out["response_count"] == 1
    saved = json.load(open(out["path"], encoding="utf-8"))
    assert saved["url"] == "https://x/list"
    assert saved["responses"][0]["status"] == 200


FX_DRAFT = '''\
from scrapers.base import BaseScraper, RawJob


class FxScraper(BaseScraper):
    SCRAPER_KEY = "fx_demo"
    SITE_HINTS = ["fx"]
    API = "https://fx.example.com/api/jobs"

    def scrape(self, url, company_name=None, config=None):
        data = self.http.get_json(self.API)
        for j in (data or {}).get("jobs", []):
            yield RawJob(source_url=j["url"], raw_text="t",
                         scraper_key=self.SCRAPER_KEY, title=j["title"])
'''


def test_run_fixture_test_replays_http(monkeypatch, tmp_path):
    _redirect_dirs(monkeypatch, tmp_path)
    path = write_scraper(FX_DRAFT, "fx")["path"]
    body = json.dumps({"jobs": [{"url": "a", "title": "A"}, {"url": "b", "title": "B"}]})
    responses = [{
        "url": "https://fx.example.com/api/jobs", "status": 200,
        "headers": {"content-type": "application/json"}, "body": body,
    }]
    fixture = save_fixture("fx", responses, "https://fx.example.com/list")
    out = run_fixture_test(path, fixture["path"], live_count=2)
    assert out["count"] == 2
    assert out["delta"] == 0
    assert out["stable"] is True


# --------------------------------------------------------------------------- #
# register_scraper
# --------------------------------------------------------------------------- #
def test_register_scraper_renames_and_appends_yaml(monkeypatch, tmp_path):
    scrapers_dir = _redirect_dirs(monkeypatch, tmp_path)
    draft = write_scraper(DRAFT_OK, "demo")["path"]
    out = register_scraper(draft, "demo", "https://demo.test/jobs")

    assert out["scraper_key"] == "demo_e2e"
    assert os.path.exists(os.path.join(str(scrapers_dir), "demo.py"))
    assert not os.path.exists(draft)  # draft consumed by the rename
    assert out["yaml_updated"] is True
    yaml_text = open(sb.TRACKED_URLS_PATH, encoding="utf-8").read()
    assert "name: demo" in yaml_text
    assert "scraper_key: demo_e2e" in yaml_text


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #
def _stub_recon(monkeypatch):
    monkeypatch.setattr(sb, "recon", lambda url, **kw: {
        "platform": None, "confidence": 0.0, "signals": [], "suggested_scraper_key": None,
    })


def test_end_to_end_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_YES_ALL", "1")
    _redirect_dirs(monkeypatch, tmp_path)
    _stub_recon(monkeypatch)

    llm = FakeLLM([
        _Resp([_ToolUse("t1", "fetch_scraper_examples", {})], "tool_use"),
        _Resp([_ToolUse("t2", "write_scraper", {"code": DRAFT_OK})], "tool_use"),
        _Resp([_ToolUse("t3", "run_scraper", {})], "tool_use"),
        _Resp([_Text("Scraper produced 3 jobs.")], "end_turn"),
    ])
    builder = ScraperBuilder("https://demo.test/jobs", name="demo", llm_client=llm)
    final = builder.run()

    assert final is BuildState.DONE
    assert builder.ctx.good is True
    assert builder.ctx.last_run["count"] == 3
    assert os.path.exists(os.path.join(sb.SCRAPERS_DIR, "demo.py"))
    assert "name: demo" in open(sb.TRACKED_URLS_PATH, encoding="utf-8").read()


def test_end_to_end_escalates_on_zero_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_YES_ALL", "1")
    _redirect_dirs(monkeypatch, tmp_path)
    _stub_recon(monkeypatch)

    llm = FakeLLM([
        _Resp([_ToolUse("t1", "write_scraper", {"code": DRAFT_EMPTY})], "tool_use"),
        _Resp([_ToolUse("t2", "run_scraper", {})], "tool_use"),
        _Resp([_Text("Could not find any jobs.")], "end_turn"),
    ])
    builder = ScraperBuilder("https://demo.test/jobs", name="empty", llm_client=llm)
    final = builder.run()

    assert final is BuildState.ESCALATED
    logs = os.listdir(sb.LOGS_DIR)
    assert any(f.startswith("scraper_builder_empty_") for f in logs)
