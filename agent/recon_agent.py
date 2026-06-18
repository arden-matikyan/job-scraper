"""Recon agent: decide which scraper handles a URL.

Staged decision sequence (cheap -> expensive):
  1.  KB known_companies (no network)
  1b. registry SITE_HINTS match on the URL (no network)
  2.  fetch once, evaluate KB detection_signals
  3.  deep investigation (job-link scan + Ollama reasoning)

When stages 1-3 don't resolve a scraper, the agent returns NEEDS_ATTENTION with
its findings (platform guess, strategy, job links) so a scraper can be built by
hand. record_success() persists a confirmed domain back into the KB.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import yaml

from scrapers.base import HttpClient, find_job_links
from scrapers.registry import registry as default_registry

logger = logging.getLogger(__name__)

_BROAD_LINK_PATTERNS = [
    "/jobs/", "/careers/", "/career/", "/positions/", "/openings/",
    "/job/", "/vacancy/", "/opportunities/",
]
_QUOTE_RE = re.compile(r'"([^"]+)"')


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path(name: str) -> str:
    return os.path.join(project_root(), "config", name)


def load_yaml(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Could not load YAML %s: %s", path, exc)
        return {}


# --------------------------------------------------------------------------- #
# Status + result
# --------------------------------------------------------------------------- #
class ReconStatus:
    MAPPED = "MAPPED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    SKIPPED = "SKIPPED"


@dataclass
class ReconResult:
    url: str
    scraper_key: Optional[str]
    platform: Optional[str]
    confidence: float
    status: str
    notes: str = ""


@dataclass
class Detection:
    platform: str
    scraper_key: Optional[str]
    confidence: float
    matched: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Platform knowledge base
# --------------------------------------------------------------------------- #
class PlatformKB:
    def __init__(self, path: Optional[str] = None):
        self.path = path or config_path("platform_kb.yaml")
        self._data = load_yaml(self.path)

    def platforms(self) -> dict:
        return (self._data or {}).get("platforms", {}) or {}

    def scraper_key_for(self, platform_name: str) -> Optional[str]:
        return (self.platforms().get(platform_name) or {}).get("scraper_key")

    def known_company_scraper(self, domain: str) -> tuple[Optional[str], Optional[str]]:
        """(scraper_key, platform_name) if ``domain`` is a known company, else (None, None)."""
        d = (domain or "").lower()
        if not d:
            return None, None
        for name, p in self.platforms().items():
            for kc in (p.get("known_companies") or []):
                k = (kc or "").lower()
                if k and (d == k or d.endswith(k) or k.endswith(d)):
                    return p.get("scraper_key"), name
        return None, None

    @staticmethod
    def _parse_signal(signal: str) -> Optional[tuple[str, str]]:
        m = _QUOTE_RE.search(signal)
        if not m:
            return None
        substr = m.group(1)
        low = signal.lower()
        if "domain" in low:
            where = "domain"
        elif "url" in low:
            where = "url"
        else:  # href / page source / html meta
            where = "source"
        return substr, where

    def detect(self, url: str, page_source: Optional[str]) -> list[Detection]:
        domain = urlparse(url).netloc.lower()
        full = (url or "").lower()
        src = (page_source or "").lower()
        results: list[Detection] = []
        for name, p in self.platforms().items():
            matched: list[str] = []
            strong = False
            for sig in (p.get("detection_signals") or []):
                parsed = self._parse_signal(sig)
                if not parsed:
                    continue
                substr, where = parsed
                if "{" in substr:  # placeholder pattern; not a literal match
                    continue
                s = substr.lower()
                if where == "domain":
                    hit = s in domain
                elif where == "url":
                    hit = s in full
                else:
                    hit = s in src
                if hit:
                    matched.append(sig)
                    if where == "domain":
                        strong = True
            if matched:
                conf = 0.9 if strong else 0.75
                if len(matched) >= 2:
                    conf = min(0.97, conf + 0.05)
                results.append(Detection(name, p.get("scraper_key"), conf, matched))
        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    def add_known_company(self, platform_name: str, domain: str) -> bool:
        if not platform_name or not domain:
            return False
        try:
            platforms = self._data.setdefault("platforms", {})
            p = platforms.get(platform_name)
            if p is None:
                return False
            kc = p.get("known_companies")
            if not isinstance(kc, list):
                kc = []
                p["known_companies"] = kc
            if domain not in kc:
                kc.append(domain)
                self._save()
                logger.info("KB: added %s -> %s", domain, platform_name)
            return True
        except Exception as exc:
            logger.error("KB add_known_company failed: %s", exc)
            return False

    def _save(self) -> None:
        # Note: rewriting drops YAML comments (accepted limitation).
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- #
# Recon agent
# --------------------------------------------------------------------------- #
class ReconAgent:
    def __init__(self, http=None, ollama=None, kb: Optional[PlatformKB] = None, registry=None):
        self.http = http or HttpClient()
        self.ollama = ollama
        self.kb = kb or PlatformKB()
        self.registry = registry or default_registry
        self.log = logging.getLogger(__name__)

    # ------------------------------------------------------------- main entry
    def investigate(
        self, url: str, company_name: Optional[str] = None
    ) -> ReconResult:
        domain = urlparse(url).netloc

        # Stage 1 — KB known company (no network)
        key, platform = self.kb.known_company_scraper(domain)
        if key:
            self.log.info("[RECON] %s -> %s (known company)", url, key)
            return ReconResult(url, key, platform, 1.0, ReconStatus.MAPPED, "known company (KB)")

        # Stage 1b — registry SITE_HINTS on the URL (no network)
        cls = self.registry.match(url)
        if cls is not None:
            self.log.info("[RECON] %s -> %s (URL hint)", url, cls.SCRAPER_KEY)
            return ReconResult(url, cls.SCRAPER_KEY, cls.SCRAPER_KEY, 0.85,
                               ReconStatus.MAPPED, "matched SITE_HINTS on URL")

        # Stage 2 — fetch once, evaluate detection signals
        html = self.http.get_text(url)
        if not html:
            return ReconResult(url, None, None, 0.0, ReconStatus.NEEDS_ATTENTION,
                               "could not fetch page (network/WAF?)")
        detections = self.kb.detect(url, html)
        if detections:
            best = detections[0]
            runner_up = detections[1].confidence if len(detections) > 1 else 0.0
            if best.confidence >= 0.7 and (best.confidence - runner_up) >= 0.1:
                self.kb.add_known_company(best.platform, domain)  # Stage 6 (early)
                self.log.info("[RECON] %s -> %s (signals)", url, best.scraper_key)
                return ReconResult(url, best.scraper_key, best.platform, best.confidence,
                                   ReconStatus.MAPPED, f"detected via signals: {best.matched}")

        # Stage 3 — deep investigation; no scraper resolved -> hand-build needed
        findings = self._deep_investigate(url, html, detections)
        note = findings.get("summary", "ambiguous platform")
        self.log.info("[RECON] %s — no known scraper (build by hand): %s", url, note)
        return ReconResult(url, None, None, 0.0, ReconStatus.NEEDS_ATTENTION, note)

    # -------------------------------------------------------- stage 3 helpers
    def _deep_investigate(self, url: str, html: str, detections: list[Detection]) -> dict:
        links = find_job_links(html, url, _BROAD_LINK_PATTERNS) if html else []
        findings: dict = {"job_link_count": len(links), "job_links": links[:10]}
        if self.ollama:
            snippet = (html or "")[:3000]
            prompt = (
                "You are analyzing a company careers page to pick a scraping strategy.\n"
                f"URL: {url}\n"
                "Known ATS platforms: greenhouse, lever, smartrecruiters, workday, icims, avature.\n"
                f"Found {len(links)} candidate job links on the page.\n"
                "HTML snippet:\n"
                f"{snippet}\n\n"
                'Return JSON: {"platform": one of the known platforms or "unknown", '
                '"confidence": 0.0-1.0, '
                '"strategy": "public_api|static_html|javascript_rendered|custom", '
                '"reasoning": "one short sentence"}'
            )
            res = self.ollama.generate_json(prompt, default={})
            if isinstance(res, dict):
                for k in ("platform", "confidence", "strategy", "reasoning"):
                    if res.get(k) is not None:
                        findings[k] = res.get(k)
        findings["summary"] = findings.get("reasoning") or (
            f"{len(links)} job links found; no ATS auto-detected"
        )
        return findings

    # ----------------------------------------------------------- KB persistence
    def record_success(self, url: str, platform: Optional[str]) -> None:
        if platform:
            self.kb.add_known_company(platform, urlparse(url).netloc)
