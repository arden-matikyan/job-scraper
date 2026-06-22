"""Freddie Mac careers scraper (careers.freddiemac.com — Phenom People SPA).

Phenom People renders job listings entirely client-side. The SPA fetches job
data on page load; there is no public JSON API accessible without a full
browser context. We use headless Playwright to render the search-results page,
extract job links from the DOM, then navigate to each detail page for the full
description text.

IMPORTANT: the SPA's default sort ("Most relevant") returns a RANDOMIZED, rotating
subset of the result pool on every load — the same URL yields different jobs each
time, and the ?from= offset param is unreliable. So we switch the on-page "Sort by"
dropdown to "Most recent" (deterministic order) and page via the UI "View next page"
control rather than the URL. We collect every listing card first, then fetch details.

Listing  : /us/en/search-results?keywords={kw}&s=1  (sort set to "Most recent" in-page;
           10 jobs/page; paged via the DOM "View next page" button)
Detail   : /us/en/job/{JR_ID}/{slug}
"""
from __future__ import annotations

import os
import re
from typing import Iterator, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_JOB_HREF_RE = re.compile(r"/us/en/job/(JR\d+)/", re.I)


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
        max_pages = int(cfg.get("max_pages", 30))          # safety cap on UI pagination clicks
        sort_label = cfg.get("sort_label", "Most recent")  # deterministic sort (default rotates)
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
                        sort_label, max_pages,
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
        sort_label: str, max_pages: int,
    ) -> Iterator[RawJob]:
        # Phase 1: collect every listing card (deterministic "Most recent" order, UI
        # pagination). Phase 2: fetch each detail page. They're separated because
        # fetching a detail page navigates `page` away from the listing, which would
        # otherwise break the in-page pagination state.
        cards = self._collect_cards(
            page, base, keywords, max_jobs, nav_timeout, spa_wait_ms, sort_label, max_pages
        )
        self.log.info("FreddieMac: collected %d unique jobs", len(cards))

        for durl, title, job_id in cards:
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

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _page_url(base: str, keywords: str) -> str:
        # Page 1 only — pagination is driven via the on-page "View next page" control,
        # not the URL (the SPA's ?from= offset is unreliable; see module docstring).
        params = {"keywords": keywords, "s": "1"}
        return f"{base}/us/en/search-results?{urlencode(params)}"

    def _collect_cards(
        self, page, base: str, keywords: str, max_jobs: int,
        nav_timeout: int, spa_wait_ms: int, sort_label: str, max_pages: int,
    ) -> list[tuple[str, Optional[str], Optional[str]]]:
        """Walk every listing page via the UI and return (abs_url, title, job_id) tuples."""
        out: list[tuple[str, Optional[str], Optional[str]]] = []
        seen: set[str] = set()
        try:
            page.goto(self._page_url(base, keywords), wait_until="domcontentloaded", timeout=nav_timeout)
            page.wait_for_timeout(spa_wait_ms)
            self._set_sort(page, sort_label, spa_wait_ms)
        except Exception as exc:
            self.log.warning("FreddieMac listing nav/sort failed: %s", exc)
            return out

        page_num = 0
        while len(seen) < max_jobs and page_num < max_pages:
            page_num += 1
            cards = self._read_cards(page, base)
            new = 0
            for durl, title, job_id in cards:
                if durl in seen:
                    continue
                seen.add(durl)
                out.append((durl, title, job_id))
                new += 1
            self.log.info(
                "FreddieMac: page %d — %d cards, %d new (%d unique total)",
                page_num, len(cards), new, len(seen),
            )
            # The "View next page" control never disables — past the last page the SPA
            # just re-renders the same final page. So "no new jobs this page" is our
            # real end-of-results signal.
            if not cards or new == 0:
                break
            if not self._next_page(page, spa_wait_ms):
                break
        return out

    def _set_sort(self, page, sort_label: str, spa_wait_ms: int) -> None:
        """Switch the on-page 'Sort by' dropdown to a deterministic order (e.g. Most recent)."""
        try:
            page.select_option("#sortselect", label=sort_label)
            page.wait_for_timeout(spa_wait_ms)  # SPA re-renders the sorted list
        except Exception as exc:
            self.log.warning("FreddieMac: could not set sort to %r: %s", sort_label, exc)

    def _read_cards(self, page, base: str) -> list[tuple[str, Optional[str], Optional[str]]]:
        """Scroll the current listing page (lazy render) and parse its job cards."""
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1500)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
        except Exception as exc:
            self.log.warning("FreddieMac scroll failed: %s", exc)

        out: list[tuple[str, Optional[str], Optional[str]]] = []
        try:
            seen: set[str] = set()
            for a in page.query_selector_all("a[href*='/us/en/job/']"):
                href = a.get_attribute("href") or ""
                if not href or href in seen:
                    continue
                seen.add(href)
                durl = href if href.startswith("http") else f"{base}{href}"
                title = (a.inner_text() or "").strip() or None
                m = _JOB_HREF_RE.search(href)
                out.append((durl, title, m.group(1) if m else None))
        except Exception as exc:
            self.log.warning("FreddieMac listing parse failed: %s", exc)
        return out

    def _next_page(self, page, spa_wait_ms: int) -> bool:
        """Click the UI 'View next page' control. Returns False when there's no next page."""
        try:
            nxt = page.query_selector("a[aria-label='View next page']")
            if nxt is None:
                return False
            if nxt.get_attribute("aria-disabled") == "true" or "disabled" in (nxt.get_attribute("class") or ""):
                return False
            nxt.evaluate("e => { e.scrollIntoView({block: 'center'}); e.click(); }")
            page.wait_for_timeout(spa_wait_ms)
            return True
        except Exception as exc:
            self.log.warning("FreddieMac: next-page click failed: %s", exc)
            return False

    def _detail(self, page, url: str, nav_timeout: int, wait_ms: int) -> str:
        """Navigate to a job detail page and return plain text."""
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
            page.wait_for_timeout(wait_ms)
            return html_to_text(page.content())
        except Exception as exc:
            self.log.warning("FreddieMac detail failed for %s: %s", url, exc)
            return ""
