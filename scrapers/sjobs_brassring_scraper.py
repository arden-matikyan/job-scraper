from __future__ import annotations

from typing import Iterator, Optional
from scrapers.base import (
    BaseScraper,
    RawJob,
    add_query_param,  # noqa: F401
    extract_title,    # noqa: F401
    find_job_links,   # noqa: F401
    html_to_text,
)


class SjobsBrassringScraper(BaseScraper):
    SCRAPER_KEY = "sjobs_brassring"
    SITE_HINTS: list[str] = []  # empty => never auto-matched
    PRIORITY = 100              # TODO: 10 for APIs, 90+ for HTML/JS fallbacks

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        
        # Fetch the detail page
        detail_url = url
        
        try:
            html = self.http.get_text(detail_url)
            if not html:
                self.log.error("sjobs_brassring: no HTML returned from %s", url)
                return
        except Exception as exc:
            self.log.warning("sjobs_brassring detail failed for %s: %s", url, exc)
            return

        # Extract job ID and title
        job_id = extract_title(html)
        if not job_id:
            self.log.warning("sjobs_brassring: no job found in %s", html)
            return
        
        # Yield a RawJob per listing
        yield RawJob(
            source_url=url,
            raw_text=html_to_text(html),
            scraper_key=self.SCRAPER_KEY,
            job_id=str(job_id),
            title=job_id,  # use the extracted job ID as title
            company=company_name,
            platform="sjobs_brassring",
        )