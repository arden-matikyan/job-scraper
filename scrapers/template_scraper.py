"""TEMPLATE scraper — the base the scraper-writer agent edits to create a new scraper.

This module is intentionally NOT registered (the registry skips ``template_scraper``)
and ``SCRAPER_KEY`` is left blank. The scraper-writer reads this file as a starting
point, fills in the class below from a ReconReport, and saves the result as a new
``*_scraper.py`` module under ``scrapers/`` only after user approval.

Contract reminders (see scrapers/base.py):
  * Set a unique SCRAPER_KEY (snake_case) and a PRIORITY (API=10, fallback=90+).
  * SITE_HINTS are substrings matched against a URL / page source for auto-routing.
    Leave empty if the scraper should only ever be selected explicitly.
  * scrape() MUST be a generator yielding RawJob, and MUST wrap every network /
    parse call in try/except so one bad record can't abort the run.
  * Use self.http (a shared HttpClient) for all requests — never raw httpx.
  * Put the complete plain-text description in RawJob.raw_text (use html_to_text);
    fill any fields you can read cleanly (job_id, title, location, ...) so they win
    over the LLM during extraction.
"""
from __future__ import annotations

from typing import Iterator, Optional

from scrapers.base import (
    BaseScraper,
    RawJob,
    add_query_param,  # noqa: F401  (handy for pagination params)
    extract_title,    # noqa: F401  (h1/title/og:title helper)
    find_job_links,   # noqa: F401  (same-host links by href pattern)
    html_to_text,
)


class TemplateScraper(BaseScraper):
    SCRAPER_KEY = ""            # TODO: unique snake_case key, e.g. "acme_careers"
    SITE_HINTS: list[str] = []  # TODO: e.g. ["acme.com/careers", "careers.acme"]
    PRIORITY = 100              # TODO: 10 for APIs, 90+ for HTML/JS fallbacks

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}

        # 1) Fetch the listing (JSON API preferred; HTML otherwise).
        #    data = self.http.get_json(api_url)            # for APIs
        #    html = self.http.get_text(url)                # for HTML
        #
        # 2) Iterate listings; for each, fetch detail if needed and build raw_text.
        #
        # 3) Yield a RawJob per listing. Example skeleton:
        #
        #    for item in items:
        #        try:
        #            detail_html = self.http.get_text(item["url"])
        #            yield RawJob(
        #                source_url=item["url"],
        #                raw_text=html_to_text(detail_html),
        #                scraper_key=self.SCRAPER_KEY,
        #                job_id=str(item.get("id")),
        #                title=extract_title(detail_html),
        #                company=company_name,
        #                platform=self.SCRAPER_KEY,
        #            )
        #        except Exception as exc:
        #            self.log.warning("parse failed for %s: %s", item, exc)
        #
        # Until implemented, yield nothing.
        return
        yield  # pragma: no cover  (marks this as a generator)
