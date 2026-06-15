from __future__ import annotations

from typing import Iterator, Optional
from scrapers.base import (
    BaseScraper,
    RawJob,
    extract_title,
    html_to_text,
)

class DynamicHtmlScraper(BaseScraper):
    SCRAPER_KEY = "dynamic_html"
    SITE_HINTS: list[str] = ["https://careers.freddiemac.com"]
    PRIORITY = 100

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        html = self.http.get_text(url)
        if not html:
            self.log.error("dynamic_html: no HTML returned from %s", url)
            return

        text = html_to_text(html)

        # Find the job title
        title = extract_title(text)

        yield RawJob(
            source_url=url,
            raw_text=text,
            scraper_key=self.SCRAPER_KEY,
            job_id="",
            title=title,
            company=company_name,
            platform=self.SCRAPER_KEY,
        )