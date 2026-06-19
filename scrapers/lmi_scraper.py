"""LMI careers scraper (careers-lmi.icims.com — iCIMS behind AWS WAF).

LMI runs a modern iCIMS instance: the careers shell embeds its listing and
detail content in an ``#icims_content_iframe`` iframe, and the whole site sits
behind an AWS WAF captcha challenge (``x-amzn-waf-action: captcha``). Plain HTTP
gets the WAF page, so the generic ``icims`` ID-walking scraper can't reach it,
and the generic ``icims_javascript`` scraper reads the top frame / paginates via
``searchPage`` — neither matches this layout. Hence a dedicated Playwright
scraper that reads the content iframe and paginates via the iCIMS ``pr`` param.

Listing : /jobs/search?pr={page}&searchKeyword={kw}&in_iframe=1
          (20 jobs/page; job links live in the content iframe as
           /jobs/{id}/{slug}/job. An out-of-range page renders zero links.)
Detail  : /jobs/{id}/{slug}/job?in_iframe=1
          (full description text rendered inside the content iframe)

A persistent browser profile carries the WAF token across runs; headless clears
the challenge once the profile is warm. Set ``headless: false`` in the config if
the WAF starts demanding an interactive captcha.
"""
from __future__ import annotations

import os
import re
import time
from typing import Iterator, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from scrapers.base import BaseScraper, RawJob, extract_title, html_to_text

_ID_RE = re.compile(r"/jobs/(\d+)/")


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


class LmiScraper(BaseScraper):
    SCRAPER_KEY = "lmi"
    SITE_HINTS = ["careers-lmi.icims.com"]
    # win auto-routing over the generic ``icims`` scraper (also PRIORITY 20),
    # which would otherwise match on ``.icims.com``.
    PRIORITY = 15

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            self.log.error("Playwright not available: %s", exc)
            return

        profile_dir = os.path.expanduser(
            cfg.get("profile_dir", "~/.job-scraper/browser-profile")
        )
        try:
            os.makedirs(profile_dir, exist_ok=True)
        except Exception:
            pass
        headless = bool(cfg.get("headless", True))
        wait_until = cfg.get("wait_until", "networkidle")
        timeout_ms = int(cfg.get("nav_timeout_ms", 45000))
        spa_wait_ms = int(cfg.get("spa_wait_ms", 2500))
        max_pages = int(cfg.get("max_pages", 50))

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"

        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(profile_dir, headless=headless)
                try:
                    page = ctx.new_page()
                    _apply_stealth(page)

                    job_ids = self._collect_ids(
                        page, url, wait_until, timeout_ms, spa_wait_ms, max_pages
                    )
                    if not job_ids:
                        self.log.error("LMI: no job IDs found at %s", url)
                        return
                    self.log.info("LMI: found %d unique job IDs", len(job_ids))

                    for jid in sorted(job_ids, reverse=True):
                        job_url = f"{base}/jobs/{jid}/job"
                        if job_url in self.seen_urls:
                            yield RawJob(
                                source_url=job_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                                job_id=str(jid), company=company_name, already_seen=True,
                                platform="icims",
                            )
                            continue
                        job = self._fetch_detail(
                            page, base, jid, company_name, wait_until, timeout_ms, spa_wait_ms
                        )
                        if job:
                            yield job
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as exc:
            self.log.error("LMI: Playwright session failed: %s", exc)

    @staticmethod
    def _content_frame(page, must_contain: str):
        """The iCIMS content iframe holding the listing/detail, or the page itself."""
        for f in page.frames:
            if must_contain in f.url and "in_iframe=1" in f.url:
                return f
        for f in page.frames:
            if must_contain in f.url:
                return f
        return page

    def _collect_ids(
        self, page, url: str, wait_until: str, timeout_ms: int, spa_wait_ms: int, max_pages: int
    ) -> set[int]:
        """Paginate the iframe listing via the ``pr`` param; collect all job IDs."""
        ids: set[int] = set()
        parsed = urlparse(url)
        params = {
            k: v[0] if len(v) == 1 else v
            for k, v in parse_qs(parsed.query, keep_blank_values=True).items()
            if k != "pr"
        }
        params["in_iframe"] = "1"

        for page_num in range(0, max_pages):
            params["pr"] = str(page_num)
            listing_url = urlunparse(parsed._replace(query=urlencode(params, doseq=True)))
            try:
                page.goto(listing_url, wait_until=wait_until, timeout=timeout_ms)
            except Exception as exc:
                self.log.warning("LMI: listing page pr=%d failed: %s", page_num, exc)
                break
            if spa_wait_ms:
                time.sleep(spa_wait_ms / 1000.0)

            frame = self._content_frame(page, "/jobs/search")
            try:
                content = frame.content()
            except Exception:
                content = page.content()
            new_ids = {int(m) for m in _ID_RE.findall(content)} - ids
            if not new_ids:
                break
            ids.update(new_ids)
            self.log.debug(
                "LMI: listing pr=%d -> %d new IDs (%d total)", page_num, len(new_ids), len(ids)
            )

        return ids

    def _fetch_detail(
        self, page, base: str, jid: int, company_name: Optional[str],
        wait_until: str, timeout_ms: int, spa_wait_ms: int,
    ) -> Optional[RawJob]:
        nav_url = f"{base}/jobs/{jid}/job?in_iframe=1"
        job_url = f"{base}/jobs/{jid}/job"
        try:
            page.goto(nav_url, wait_until=wait_until, timeout=timeout_ms)
            if spa_wait_ms:
                time.sleep(spa_wait_ms / 1000.0)
            if "/jobs/intro" in page.url or "/search" in page.url:
                return None
            frame = self._content_frame(page, f"/jobs/{jid}/")
            try:
                content = frame.content()
            except Exception:
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
            self.log.warning("LMI: detail %s failed: %s", job_url, exc)
            return None
