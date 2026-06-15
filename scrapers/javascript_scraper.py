"""JavaScript-rendered fallback scraper (Playwright sync API + stealth).

Used for sites that need a real browser. Renders the listing page, extracts job
links by href patterns, then renders each detail page for text. Playwright and
playwright-stealth are imported lazily so the module always imports even when the
browser isn't installed (the scraper just no-ops with a logged error).

Note: aggressively anti-bot or auth-walled sites (LinkedIn, Indeed) frequently
fail even with stealth; this is best-effort.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

from scrapers.base import BaseScraper, RawJob, find_job_links, html_to_text

_DEFAULT_PATTERNS = ["/jobs/", "/careers/", "/positions/", "/openings/", "/job/", "/viewjob"]


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


class JavascriptScraper(BaseScraper):
    SCRAPER_KEY = "javascript_rendered"
    SITE_HINTS = ["linkedin.com/jobs", "indeed.com", "workday.com"]
    PRIORITY = 90

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
        wait_until = cfg.get("wait_until", "networkidle")
        timeout_ms = int(cfg.get("nav_timeout_ms", 45000))
        max_pages = int(cfg.get("max_pages", 20))
        patterns = cfg.get("link_patterns", _DEFAULT_PATTERNS)

        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(profile_dir, headless=True)
                try:
                    page = ctx.new_page()
                    _apply_stealth(page)

                    links: list[str] = []
                    try:
                        page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                        links = find_job_links(page.content(), url, patterns)
                    except Exception as exc:
                        self.log.warning("JS render of %s failed: %s", url, exc)

                    for durl in links[:max_pages]:
                        try:
                            page.goto(durl, wait_until=wait_until, timeout=timeout_ms)
                            text = html_to_text(page.content())
                            if not text:
                                continue
                            yield RawJob(
                                source_url=durl,
                                raw_text=text,
                                scraper_key=self.SCRAPER_KEY,
                                title=page.title() or None,
                                company=company_name,
                                platform="javascript_rendered",
                            )
                        except Exception as exc:
                            self.log.warning("JS render detail %s failed: %s", durl, exc)
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as exc:
            self.log.error("Playwright session failed: %s", exc)
