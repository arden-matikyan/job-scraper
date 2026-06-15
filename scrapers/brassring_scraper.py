from __future__ import annotations

from typing import Iterator, Optional

from scrapers.base import (
    BaseScraper,
    RawJob,
    add_query_param,
    extract_title,
    find_job_links,
    html_to_text,
)


class BrassringScraper(BaseScraper):
    SCRAPER_KEY = "brassring"
    SITE_HINTS: list[str] = []  # empty => never auto-matched
    PRIORITY = 100

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}

        # 1) Fetch the job details (HTML)
        html = self.http.get_text(url)

        if not html:
            self.log.error("brassring: no HTML returned from %s", url)
            return

        try:
            yield RawJob(
                source_url=url,
                raw_text=html_to_text(html),
                scraper_key=self.SCRAPER_KEY,
                job_id="",
                title=extract_title(html),
                company=company_name,
                platform=self.SCRAPER_KEY,
            )
        except Exception as exc:
            self.log.warning("brassring: parse failed for %s: %s", url, exc)

    def _get_job_id(self, url: str) -> Optional[str]:
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        return parsed_url.path.split("/")[-1]

    def _add_query_params(self, base_url: str, params: dict) -> str:
        from urllib.parse import urlencode, urlparse

        query_params = urlencode(params, doseq=True)
        base_url += "?" + query_params
        return base_url


if __name__ == "__main__":
    print("This is the brassring scraper module.")