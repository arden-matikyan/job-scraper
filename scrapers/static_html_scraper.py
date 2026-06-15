"""Static HTML fallback scraper.

No SITE_HINTS => never auto-matched; it is selected explicitly by recon when no
ATS fits. Finds job links on a listing page by href substring patterns and fetches
each detail page for plain text. Capped at ``max_pages`` detail fetches.
"""
from __future__ import annotations

from typing import Iterator, Optional

from scrapers.base import (
    BaseScraper,
    RawJob,
    extract_title,
    find_job_links,
    html_to_text,
)

_DEFAULT_PATTERNS = ["/jobs/", "/careers/", "/positions/", "/openings/"]


class StaticHtmlScraper(BaseScraper):
    SCRAPER_KEY = "static_html"
    SITE_HINTS: list[str] = []  # empty => never auto-matched
    PRIORITY = 100

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        patterns = cfg.get("link_patterns", _DEFAULT_PATTERNS)
        max_pages = int(cfg.get("max_pages", 20))

        html = self.http.get_text(url)
        if not html:
            self.log.error("static_html: no HTML returned from %s", url)
            return
        links = find_job_links(html, url, patterns)
        if not links:
            self.log.warning("static_html: found no job links on %s", url)
            return
        for durl in links[:max_pages]:
            try:
                if durl in self.seen_urls:
                    yield RawJob(
                        source_url=durl, raw_text="", scraper_key=self.SCRAPER_KEY,
                        job_id=self._id_from_url(durl), company=company_name,
                        already_seen=True, platform="static_html",
                    )
                    continue
                detail_html = self.http.get_text(durl)
                text = html_to_text(detail_html)
                if not text:
                    continue
                yield RawJob(
                    source_url=durl,
                    raw_text=text,
                    scraper_key=self.SCRAPER_KEY,
                    job_id=self._id_from_url(durl),
                    title=extract_title(detail_html),
                    company=company_name,
                    platform="static_html",
                )
            except Exception as exc:
                self.log.warning("static_html detail failed for %s: %s", durl, exc)

    @staticmethod
    def _id_from_url(url: str) -> Optional[str]:
        seg = [s for s in url.split("?")[0].split("/") if s]
        return seg[-1] if seg else None
