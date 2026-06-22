"""Freddie Mac careers scraper (careers.freddiemac.com — Phenom People SPA).

Phenom People renders job listings entirely client-side. The SPA fetches job
data on page load; there is no public JSON API accessible without a full
browser context. We use headless Playwright to render each search-results page,
extract job links from the DOM, then navigate to each detail page for the full
description text.

Listing  : /us/en/search-results?keywords={kw}&from={offset}&s=1
           (10 jobs/page; pagination links are rendered into the DOM)
Detail   : /us/en/job/{JR_ID}/{slug}
"""
from __future__ import annotations

import os
import re
from typing import Iterator, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_JOB_HREF_RE = re.compile(r"/us/en/job/(JR\d+)/", re.I)
_JOBS_PER_PAGE = 10


def _apply_stealth(page) -> None:
    try:
        from playwright_stealth import stealth_sync
        stealth_sync(page)
        return
    except Exception:
        pass
    try:
        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page)
    except Exception:
        pass


class FreddieMacScraper(BaseScraper):
    SCRAPER_KEY = "freddiemac"
    SITE_HINTS = ["careers.freddiemac.com"]
    PRIORITY = 20

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            self.log.error("Playwright not available: %s", exc)
            return

        profile_dir = os.path.expanduser(cfg.get("profile_dir", "~/.job-scraper/browser-profile"))
        try:
            os.makedirs(profile_dir, exist_ok=True)
        except Exception:
            pass

        headless = bool(cfg.get("headless", True))
        nav_timeout = int(cfg.get("nav_timeout_ms", 45000))
        spa_wait_ms = int(cfg.get("spa_wait_ms", 10000))   # ms to wait after domcontentloaded
        detail_wait_ms = int(cfg.get("detail_wait_ms", 6000))
        max_jobs = int(cfg.get("max_jobs", 200))
        company = company_name or "Freddie Mac"

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        qs = parse_qs(p.query)
        keywords = (qs.get("keywords") or qs.get("keyword") or ["software"])[0]

        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(
                    profile_dir,
                    headless=headless,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                try:
                    page = ctx.new_page()
                    _apply_stealth(page)
                    yield from self._walk_all(
                        page, base, keywords, company,
                        max_jobs, nav_timeout, spa_wait_ms, detail_wait_ms,
                    )
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as exc:
            self.log.error("FreddieMac Playwright session failed: %s", exc)

    # ----------------------------------------------------------------- listing

    def _walk_all(
        self, page, base: str, keywords: str, company: str,
        max_jobs: int, nav_timeout: int, spa_wait_ms: int, detail_wait_ms: int,
    ) -> Iterator[RawJob]:
        seen_urls: set[str] = set()
        offset = 0

        while len(seen_urls) < max_jobs:
            listing_url = self._page_url(base, keywords, offset)
            cards = self._listing_page(page, listing_url, base, nav_timeout, spa_wait_ms)
            if not cards:
                self.log.info("FreddieMac: no jobs at offset=%d, stopping", offset)
                break

            self.log.info("FreddieMac: offset=%d found %d jobs", offset, len(cards))

            for durl, title, job_id in cards:
                if durl in seen_urls:
                    continue
                seen_urls.add(durl)

                if durl in self.seen_urls:
                    yield RawJob(
                        source_url=durl,
                        raw_text="",
                        scraper_key=self.SCRAPER_KEY,
                        job_id=job_id,
                        title=title,
                        company=company,
                        already_seen=True,
                        platform="phenompeople",
                    )
                    continue

                raw_text = self._detail(page, durl, nav_timeout, detail_wait_ms)
                if not raw_text:
                    continue

                yield RawJob(
                    source_url=durl,
                    raw_text=raw_text,
                    scraper_key=self.SCRAPER_KEY,
                    job_id=job_id,
                    title=title,
                    company=company,
                    platform="phenompeople",
                )

            if len(cards) < _JOBS_PER_PAGE:
                break
            offset += _JOBS_PER_PAGE

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _page_url(base: str, keywords: str, offset: int) -> str:
        # Mirror the site's own pagination exactly: page 1 is the BASE url (no `from`);
        # pages 2+ add from=10, from=20, … The SPA's no-`from` landing IS the real
        # first page — explicitly sending from=0 returns a different view and skips
        # those listings. (Listing cards lazy-render, so _listing_page scrolls.)
        params: dict = {"keywords": keywords, "s": "1"}
        if offset:
            params["from"] = offset
        return f"{base}/us/en/search-results?{urlencode(params)}"

    def _listing_page(
        self, page, url: str, base: str, nav_timeout: int, spa_wait_ms: int
    ) -> list[tuple[str, Optional[str], Optional[str]]]:
        """Navigate to a listing page and return (abs_url, title, job_id) for each job."""
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
            page.wait_for_timeout(spa_wait_ms)
            # Scroll through the page so the SPA renders all job cards — Phenom People
            # uses virtual/lazy rendering and only mounts cards that enter the viewport.
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1500)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
        except Exception as exc:
            self.log.warning("FreddieMac listing nav failed for %s: %s", url, exc)

        out: list[tuple[str, Optional[str], Optional[str]]] = []
        try:
            anchors = page.query_selector_all("a[href*='/us/en/job/']")
            seen: set[str] = set()
            for a in anchors:
                href = a.get_attribute("href") or ""
                if not href or href in seen:
                    continue
                seen.add(href)
                durl = href if href.startswith("http") else f"{base}{href}"
                title = (a.inner_text() or "").strip() or None
                m = _JOB_HREF_RE.search(href)
                job_id = m.group(1) if m else None
                out.append((durl, title, job_id))
        except Exception as exc:
            self.log.warning("FreddieMac listing parse failed: %s", exc)
        return out

    def _detail(self, page, url: str, nav_timeout: int, wait_ms: int) -> str:
        """Navigate to a job detail page and return plain text."""
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
            page.wait_for_timeout(wait_ms)
            return html_to_text(page.content())
        except Exception as exc:
            self.log.warning("FreddieMac detail failed for %s: %s", url, exc)
            return ""
