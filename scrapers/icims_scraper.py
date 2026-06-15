"""iCIMS scrapers: static ID-walking and a Playwright variant for WAF-blocked sites.

iCIMS job detail pages live at {base}/jobs/{id}/job. There is no clean public
listing API, so the static scraper finds the highest job id referenced on the
landing/search page, scans a window of ids around it (high -> low), and treats a
redirect away from /jobs/{id}/ as a miss. This is WAF-sensitive by nature.

IcimsJsScraper uses Playwright for all requests (bypasses WAF) and paginates the
search listing to collect every visible job ID before navigating to detail pages.
"""
from __future__ import annotations

import os
import re
from typing import Iterator, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from scrapers.base import BaseScraper, RawJob, extract_title, html_to_text


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


class IcimsScraper(BaseScraper):
    SCRAPER_KEY = "icims"
    SITE_HINTS = [".icims.com", "careers-", "icims.com/jobs"]
    PRIORITY = 20

    _ID_RE = re.compile(r"/jobs/(\d+)/")

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        window = int(cfg.get("scan_window", 300))
        max_miss = int(cfg.get("max_consecutive_misses", 150))

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        start_id = self._find_max_id(url, base)
        if start_id is None:
            self.log.error("iCIMS: could not find any job id near %s", url)
            return

        lo = max(1, start_id - window)
        hi = start_id + window
        misses = 0
        self.log.info("iCIMS: scanning ids %d..%d (start=%d)", hi, lo, start_id)
        for jid in range(hi, lo - 1, -1):  # walk downward from the top of the window
            job_url = f"{base}/jobs/{jid}/job"
            if job_url in self.seen_urls:
                # already saved => known hit; skip the probe fetch and reset misses
                misses = 0
                yield RawJob(
                    source_url=job_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                    job_id=str(jid), company=company_name, already_seen=True, platform="icims",
                )
                continue
            job = self._try_job(base, jid, company_name)
            if job is None:
                misses += 1
                if misses >= max_miss:
                    self.log.info("iCIMS: stopping after %d consecutive misses", misses)
                    break
                continue
            misses = 0
            yield job

    def _find_max_id(self, url: str, base: str) -> Optional[int]:
        candidates = (
            url,
            f"{base}/jobs/search?ss=1&searchKeyword=&in_iframe=1",
            f"{base}/jobs/search",
            f"{base}/jobs",
        )
        for candidate in candidates:
            html = self.http.get_text(candidate)
            ids = [int(m) for m in self._ID_RE.findall(html or "")]
            if ids:
                self.log.info("iCIMS: found max id %d from %s", max(ids), candidate)
                return max(ids)
        return None

    def _try_job(self, base: str, jid: int, company_name: Optional[str]) -> Optional[RawJob]:
        job_url = f"{base}/jobs/{jid}/job"
        resp = self.http.get(job_url)
        if resp is None or resp.status_code >= 400:
            return None
        final = str(resp.url)
        if "/jobs/intro" in final or "/search" in final or f"/jobs/{jid}/" not in final:
            return None
        body = resp.text or ""
        text = html_to_text(body)
        if not text or (len(text) < 200 and "iCIMS" not in body):
            return None
        return RawJob(
            source_url=job_url,
            raw_text=text,
            scraper_key=self.SCRAPER_KEY,
            job_id=str(jid),
            title=extract_title(body),
            company=company_name,
            platform="icims",
        )


class IcimsJsScraper(BaseScraper):
    """iCIMS scraper using Playwright for WAF-blocked sites.

    Paginates the search listing page via Playwright to collect all job IDs, then
    navigates to each detail page. Set scraper_key: icims_javascript in
    tracked_urls.yaml to use it (never auto-matched).
    """

    SCRAPER_KEY = "icims_javascript"
    SITE_HINTS = []  # explicit only; never auto-matched
    PRIORITY = 25

    _ID_RE = re.compile(r"/jobs/(\d+)/")

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
        os.makedirs(profile_dir, exist_ok=True)
        wait_until = cfg.get("wait_until", "networkidle")
        timeout_ms = int(cfg.get("nav_timeout_ms", 45000))
        max_listing_pages = int(cfg.get("max_listing_pages", 50))

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"

        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(profile_dir, headless=True)
                try:
                    page = ctx.new_page()
                    _apply_stealth(page)

                    job_ids = self._collect_ids(page, url, wait_until, timeout_ms, max_listing_pages)
                    if not job_ids:
                        self.log.error("iCIMS JS: no job IDs found at %s", url)
                        return

                    self.log.info("iCIMS JS: found %d unique job IDs", len(job_ids))

                    for jid in sorted(job_ids, reverse=True):
                        job_url = f"{base}/jobs/{jid}/job"
                        if job_url in self.seen_urls:
                            yield RawJob(
                                source_url=job_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                                job_id=str(jid), company=company_name, already_seen=True,
                                platform="icims",
                            )
                            continue
                        job = self._fetch_detail(page, base, jid, company_name, wait_until, timeout_ms)
                        if job:
                            yield job
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as exc:
            self.log.error("iCIMS JS: Playwright session failed: %s", exc)

    def _collect_ids(self, page, url: str, wait_until: str, timeout_ms: int, max_pages: int) -> set[int]:
        """Render the search listing across all pages and return all discovered job IDs."""
        ids: set[int] = set()
        parsed = urlparse(url)
        params = {k: v[0] if len(v) == 1 else v
                  for k, v in parse_qs(parsed.query, keep_blank_values=True).items()
                  if k not in ("searchPage", "startrow")}

        for page_num in range(1, max_pages + 1):
            params["searchPage"] = str(page_num)
            listing_url = urlunparse(parsed._replace(query=urlencode(params, doseq=True)))
            try:
                page.goto(listing_url, wait_until=wait_until, timeout=timeout_ms)
            except Exception as exc:
                self.log.warning("iCIMS JS: listing page %d failed: %s", page_num, exc)
                break

            new_ids = {int(m) for m in self._ID_RE.findall(page.content())} - ids
            if not new_ids:
                break
            ids.update(new_ids)
            self.log.debug("iCIMS JS: listing page %d → %d new IDs (%d total)", page_num, len(new_ids), len(ids))

        return ids

    def _fetch_detail(
        self, page, base: str, jid: int, company_name: Optional[str], wait_until: str, timeout_ms: int
    ) -> Optional[RawJob]:
        job_url = f"{base}/jobs/{jid}/job"
        try:
            page.goto(job_url, wait_until=wait_until, timeout=timeout_ms)
            final = page.url
            if "/jobs/intro" in final or "/search" in final or f"/jobs/{jid}/" not in final:
                return None
            content = page.content()
            text = html_to_text(content)
            if not text or (len(text) < 200 and "iCIMS" not in content):
                return None
            return RawJob(
                source_url=job_url,
                raw_text=text,
                scraper_key=self.SCRAPER_KEY,
                job_id=str(jid),
                title=extract_title(content),
                company=company_name,
                platform="icims",
            )
        except Exception as exc:
            self.log.warning("iCIMS JS: detail %s failed: %s", job_url, exc)
            return None
