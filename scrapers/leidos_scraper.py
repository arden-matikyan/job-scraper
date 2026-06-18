"""Leidos careers scraper (careers.leidos.com — Talemetry, Cloudflare-protected).

careers.leidos.com sits behind Cloudflare's *managed challenge*, which a plain
HTTP client (and even headless Chromium) cannot clear — it just gets the
"Just a moment..." interstitial. A real, headful browser passes the challenge
automatically, so this scraper drives a persistent headful Chromium via
Playwright (+ stealth), exactly like ``javascript_rendered`` but specialised for
this site.

Once the challenge is solved on the first navigation, the cf_clearance cookie is
live for the session, so the remaining pages are pulled with lightweight
*same-origin in-page* ``fetch()`` calls (no per-page rendering) — both fast and
fingerprint-consistent.

Listing  : {url}&page=N   (server-rendered, 25 rows/page; <div class="row"> per job)
Detail   : https://careers.leidos.com/jobs/{numericId}-{slug}
           every detail page embeds a JSON-LD JobPosting with a clean title,
           datePosted, full HTML description, hiringOrganization and jobLocation.

Headful is required: set ``headless: true`` in scraper_configs only if the host
can clear Cloudflare without a visible window (it usually cannot).
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

from scrapers.base import (
    BaseScraper,
    RawJob,
    add_query_param,
    extract_title,
    html_to_text,
)

_JOB_HREF_RE = re.compile(r"/jobs/(\d+)", re.I)
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


def _text_after(text: str, label: str, stops: tuple[str, ...]) -> Optional[str]:
    """Substring after ``label`` up to the first of ``stops`` (or end)."""
    i = text.find(label)
    if i < 0:
        return None
    rest = text[i + len(label):]
    end = len(rest)
    for s in stops:
        j = rest.find(s)
        if 0 <= j < end:
            end = j
    return rest[:end].strip() or None


class LeidosScraper(BaseScraper):
    SCRAPER_KEY = "leidos"
    SITE_HINTS = ["careers.leidos.com"]
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
        company = company_name or "Leidos"
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
                    # Solve the Cloudflare challenge once on the listing page.
                    if not self._open(page, url, nav_timeout):
                        self.log.error("Leidos: could not clear Cloudflare challenge for %s", url)
                        return
                    yield from self._walk_listing(page, url, base, company, max_pages, nav_timeout)
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as exc:
            self.log.error("Leidos Playwright session failed: %s", exc)

    # ----------------------------------------------------------------- listing
    def _walk_listing(
        self, page, url: str, base: str, company: str, max_pages: int, timeout: int
    ) -> Iterator[RawJob]:
        seen_pages: set[str] = set()
        for pno in range(1, max_pages + 1):
            page_url = add_query_param(url, "page", pno)
            html = self._fetch(page, page_url, timeout)
            rows = [r for r in self._parse_rows(html, base) if r[0] not in seen_pages]
            if not rows:
                break  # no new jobs on this page => past the last results page
            for durl, title, location, req in rows:
                seen_pages.add(durl)
                try:
                    if durl in self.seen_urls:
                        yield RawJob(
                            source_url=durl, raw_text="", scraper_key=self.SCRAPER_KEY,
                            job_id=req or self._id_from_url(durl), title=title,
                            company=company, location=location, already_seen=True,
                            platform="leidos",
                        )
                        continue
                    job = self._detail(page, durl, title, location, req, company, timeout)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("Leidos detail failed for %s: %s", durl, exc)

    def _parse_rows(self, html: str, base: str) -> list[tuple[str, Optional[str], Optional[str], Optional[str]]]:
        """Return (detail_url, title, location, req_number) for each job row."""
        out: list[tuple[str, Optional[str], Optional[str], Optional[str]]] = []
        if not html:
            return out
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            seen: set[str] = set()
            for a in soup.select('a[href*="/jobs/"]'):
                href = a.get("href", "")
                if not _JOB_HREF_RE.search(href):
                    continue
                durl = urljoin(base, href)
                if durl in seen:
                    continue
                seen.add(durl)
                title = a.get_text(strip=True) or None
                location = req = None
                row = a
                for _ in range(6):
                    row = row.parent
                    if row is None:
                        break
                    if row.name == "div" and "row" in (row.get("class") or []):
                        break
                if row is not None:
                    rowtext = row.get_text(" ", strip=True)
                    location = _text_after(rowtext, "Location:", ("Clearance:", "Req Number:"))
                    req = _text_after(rowtext, "Req Number:", ())
                out.append((durl, title, location, req))
        except Exception as exc:
            self.log.warning("Leidos listing parse failed: %s", exc)
        return out

    # ------------------------------------------------------------------ detail
    def _detail(
        self, page, durl: str, listing_title: Optional[str], listing_loc: Optional[str],
        req: Optional[str], company: str, timeout: int,
    ) -> Optional[RawJob]:
        html = self._fetch(page, durl, timeout)
        if not html:
            return None
        ld = self._jobposting_ld(html)
        desc_html = (ld or {}).get("description")
        raw_text = html_to_text(desc_html) if desc_html else html_to_text(html)
        if not raw_text:
            return None
        title = (ld or {}).get("title") or listing_title or extract_title(html)
        return RawJob(
            source_url=durl,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=self._req_from_ld(ld) or req or self._id_from_url(durl),
            title=title,
            company=company,
            location=listing_loc or self._location_from_ld(ld),
            posted_date=(ld or {}).get("datePosted"),
            platform="leidos",
        )

    # ----------------------------------------------------------- JSON-LD helpers
    @staticmethod
    def _jobposting_ld(html: str) -> Optional[dict]:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for s in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(s.string or "{}")
                except Exception:
                    continue
                candidates = data.get("@graph", [data]) if isinstance(data, dict) else (
                    data if isinstance(data, list) else []
                )
                for c in candidates:
                    if isinstance(c, dict) and c.get("@type") == "JobPosting":
                        return c
        except Exception:
            pass
        return None

    @staticmethod
    def _location_from_ld(ld: Optional[dict]) -> Optional[str]:
        if not ld:
            return None
        loc = ld.get("jobLocation")
        if isinstance(loc, list):
            loc = loc[0] if loc else None
        addr = (loc or {}).get("address") if isinstance(loc, dict) else None
        if not isinstance(addr, dict):
            return None
        parts = [addr.get("addressLocality"), addr.get("addressRegion")]
        return ", ".join(p for p in parts if p) or None

    @staticmethod
    def _req_from_ld(ld: Optional[dict]) -> Optional[str]:
        if not ld:
            return None
        ident = ld.get("identifier")
        if isinstance(ident, dict):
            return ident.get("value") or None
        if isinstance(ident, str):
            return ident or None
        return None

    @staticmethod
    def _id_from_url(url: str) -> Optional[str]:
        m = _JOB_HREF_RE.search(url)
        return m.group(1) if m else None

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
            self.log.debug("Leidos in-page fetch error for %s: %s", url, exc)
            res = {"status": 0, "text": ""}
        text = (res or {}).get("text") or ""
        if (res or {}).get("status") == 200 and not self._is_challenge(text):
            return text
        # Re-solve via a real navigation, then return the rendered HTML.
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
            self.log.warning("Leidos goto %s failed: %s", url, exc)
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
