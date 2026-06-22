"""SAIC careers scraper (jobs.saic.com — Talemetry, Cloudflare-protected).

jobs.saic.com sits behind Cloudflare's managed challenge (every path returns
``cf-mitigated: challenge`` to a plain HTTP client), so this scraper drives a
persistent **headful** Chromium via Playwright (+ stealth) to clear it once, then
pulls the remaining pages with lightweight same-origin in-page ``fetch()`` calls —
the same approach as leidos_scraper.py.

Listing : {url}&page=N   (server-rendered; 25 rows/page; each job is a
          ``div.row`` with .large-6 (title + link), .large-4 (location),
          .large-2 ("Date Posted: Jun 10, 2026"))
Detail  : https://jobs.saic.com/jobs/{numericId}-{slug}
          The detail JSON-LD is malformed, so the description is taken from the
          rendered page text; the req id is read from the "Job ID: ..." label.

Headful is required (Cloudflare). Date filtering: set ``oldest_date:
"YYYY-MM-DD"`` (per-company in tracked_urls.yaml or scraper_configs) to skip
postings older than that — applied from the listing date, before any detail fetch.
"""
from __future__ import annotations

import os
import re
from datetime import date as _date
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

from scrapers.base import (
    BaseScraper,
    RawJob,
    add_query_param,
    html_to_text,
    parse_posted_date,
)

_JOB_HREF_RE = re.compile(r"/jobs/(\d+)-", re.I)
_JOBID_RE = re.compile(r"Job\s*ID\s*:\s*([A-Za-z0-9\-]+)", re.I)
_CHALLENGE_MARKERS = ("just a moment", "attention required", "challenge-platform", "cf-error-details")


def _apply_stealth(page) -> None:
    """Apply playwright-stealth across its differing API versions; ignore failures."""
    try:
        from playwright_stealth import stealth_sync  # older API

        stealth_sync(page)
        return
    except Exception:
        pass
    try:
        from playwright_stealth import Stealth  # newer API

        Stealth().apply_stealth_sync(page)
    except Exception:
        pass


class SaicScraper(BaseScraper):
    SCRAPER_KEY = "saic"
    SITE_HINTS = ["jobs.saic.com"]
    PRIORITY = 20
    REQUIRES_INTERACTION = True  # launches a (headful) browser to clear Cloudflare

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
        headless = bool(cfg.get("headless", False))  # Cloudflare normally needs headful
        nav_timeout = int(cfg.get("nav_timeout_ms", 60000))
        max_pages = int(cfg.get("max_pages", 50))
        company = company_name or "SAIC"
        oldest = self._parse_oldest(cfg.get("oldest_date"))
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"

        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(
                    profile_dir, headless=headless,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                try:
                    page = ctx.new_page()
                    _apply_stealth(page)
                    if not self._open(page, url, nav_timeout):
                        self.log.error("SAIC: could not clear Cloudflare challenge for %s", url)
                        return
                    yield from self._walk_listing(
                        page, url, base, company, max_pages, nav_timeout, oldest
                    )
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as exc:
            self.log.error("SAIC Playwright session failed: %s", exc)

    # ----------------------------------------------------------------- listing
    def _walk_listing(
        self, page, url: str, base: str, company: str, max_pages: int,
        timeout: int, oldest: Optional[_date],
    ) -> Iterator[RawJob]:
        seen: set[str] = set()
        for pno in range(1, max_pages + 1):
            page_url = add_query_param(url, "page", pno)
            html = self._fetch(page, page_url, timeout)
            rows = [r for r in self._parse_rows(html, base) if r["url"] not in seen]
            if not rows:
                break  # no new jobs => past the last results page
            for card in rows:
                seen.add(card["url"])
                if oldest and card["posted_date"]:
                    try:
                        if _date.fromisoformat(card["posted_date"]) < oldest:
                            continue  # too old — skip detail fetch entirely
                    except ValueError:
                        pass
                try:
                    if card["url"] in self.seen_urls:
                        yield RawJob(
                            source_url=card["url"], raw_text="", scraper_key=self.SCRAPER_KEY,
                            job_id=self._id_from_url(card["url"]), title=card["title"],
                            company=company, location=card["location"],
                            posted_date=card["posted_date"], already_seen=True, platform="talemetry",
                        )
                        continue
                    job = self._detail(page, card, company, timeout)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("SAIC detail failed for %s: %s", card["url"], exc)

    def _parse_rows(self, html: str, base: str) -> list[dict]:
        """Return one dict per job row: url, title, location, posted_date."""
        out: list[dict] = []
        if not html:
            return out
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            seen: set[str] = set()
            for row in soup.select("div.row"):
                a = row.select_one('.large-6 a[href*="/jobs/"]')
                if not a:
                    continue
                href = a.get("href", "")
                if not _JOB_HREF_RE.search(href):
                    continue  # skip facet / non-detail links
                durl = urljoin(base, href)
                if durl in seen:
                    continue
                seen.add(durl)
                loc_el = row.select_one(".large-4")
                date_el = row.select_one(".large-2")
                location = self._clean(loc_el.get_text(" ", strip=True)) if loc_el else None
                date_raw = date_el.get_text(" ", strip=True) if date_el else None
                out.append({
                    "url": durl,
                    "title": a.get_text(strip=True) or None,
                    "location": location,
                    "posted_date": parse_posted_date(date_raw),
                })
        except Exception as exc:
            self.log.warning("SAIC listing parse failed: %s", exc)
        return out

    @staticmethod
    def _clean(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        t = re.sub(r"^\s*Location\s*:?\s*", "", text, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip().strip(",").strip()
        return t or None

    # ------------------------------------------------------------------ detail
    def _detail(self, page, card: dict, company: str, timeout: int) -> Optional[RawJob]:
        html = self._fetch(page, card["url"], timeout)
        if not html:
            return None
        raw_text = html_to_text(html)
        if not raw_text:
            return None
        m = _JOBID_RE.search(html)
        req_id = (m.group(1) if m else None) or self._id_from_url(card["url"])
        return RawJob(
            source_url=card["url"],
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=req_id,
            title=card["title"],
            company=company,
            location=card["location"],
            posted_date=card["posted_date"],
            platform="talemetry",
        )

    @staticmethod
    def _id_from_url(url: str) -> Optional[str]:
        m = _JOB_HREF_RE.search(url)
        return m.group(1) if m else None

    def _parse_oldest(self, raw) -> Optional[_date]:
        if not raw:
            return None
        try:
            return _date.fromisoformat(str(raw))
        except ValueError:
            self.log.warning("SAIC: invalid oldest_date %r — ignoring", raw)
            return None

    # -------------------------------------------------------------- navigation
    def _fetch(self, page, url: str, timeout: int) -> str:
        """Lightweight same-origin in-page fetch; falls back to a full nav if the
        session hit a Cloudflare interstitial (e.g. the clearance cookie expired)."""
        try:
            res = page.evaluate(
                """async (u) => {
                    try {
                        const r = await fetch(u, {headers: {'X-Requested-With': 'XMLHttpRequest'}});
                        return {status: r.status, text: await r.text()};
                    } catch (e) { return {status: 0, text: ''}; }
                }""",
                url,
            )
        except Exception as exc:
            self.log.debug("SAIC in-page fetch error for %s: %s", url, exc)
            res = {"status": 0, "text": ""}
        text = (res or {}).get("text") or ""
        if (res or {}).get("status") == 200 and not self._is_challenge(text):
            return text
        if self._open(page, url, timeout):
            try:
                return page.content()
            except Exception:
                return ""
        return ""

    def _open(self, page, url: str, timeout: int) -> bool:
        """Navigate and wait for Cloudflare's challenge to clear (title stops being
        the interstitial). Returns True once the real page is showing."""
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except Exception as exc:
            self.log.warning("SAIC goto %s failed: %s", url, exc)
        for _ in range(8):
            page.wait_for_timeout(2000)
            try:
                title = (page.title() or "").lower()
            except Exception:
                title = ""
            if title and not any(m in title for m in ("just a moment", "attention required")):
                return True
        return False

    @staticmethod
    def _is_challenge(html: str) -> bool:
        low = (html or "")[:4000].lower()
        return any(m in low for m in _CHALLENGE_MARKERS)
