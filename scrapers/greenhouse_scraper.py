"""Greenhouse public board API scraper.

Endpoint: https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
Returns every job with its (HTML-escaped) content, so a single request yields
full descriptions — no per-job detail fetch needed.
"""
from __future__ import annotations

import html
from typing import Iterator, Optional
from urllib.parse import parse_qs, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text


class GreenhouseScraper(BaseScraper):
    SCRAPER_KEY = "greenhouse_api"
    SITE_HINTS = ["greenhouse.io", "boards.greenhouse", "grnh.se"]
    PRIORITY = 10

    API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        slug = self._slug(url)
        if not slug:
            self.log.error("Could not extract Greenhouse slug from %s", url)
            return
        data = self.http.get_json(self.API.format(slug=slug))
        if not isinstance(data, dict) or "jobs" not in data:
            self.log.error("Greenhouse API returned no jobs for slug %r", slug)
            return
        for job in data.get("jobs", []):
            try:
                yield self._parse(job, company_name)
            except Exception as exc:  # never let one bad record abort the rest
                self.log.warning("Failed to parse Greenhouse job: %s", exc)

    def _parse(self, job: dict, company_name: Optional[str]) -> RawJob:
        content_html = html.unescape(job.get("content") or "")
        text = html_to_text(content_html)
        loc = (job.get("location") or {}).get("name")
        offices = [o.get("name") for o in (job.get("offices") or []) if o.get("name")]
        locations_all = offices or ([loc] if loc else None)
        job_id = job.get("id")
        return RawJob(
            source_url=job.get("absolute_url") or "",
            raw_text=text,
            scraper_key=self.SCRAPER_KEY,
            job_id=str(job_id) if job_id is not None else None,
            title=job.get("title"),
            company=company_name,
            location=loc,
            locations_all=locations_all if locations_all and len(locations_all) > 1 else None,
            posted_date=job.get("updated_at"),
            platform="greenhouse",
        )

    @staticmethod
    def _slug(url: str) -> Optional[str]:
        p = urlparse(url)
        qs = parse_qs(p.query)
        if qs.get("for"):  # embedded board: ...?for=companyslug
            return qs["for"][0]
        segs = [s for s in p.path.split("/") if s]
        if not segs:
            return None
        # company slug is the first path segment on boards.greenhouse.io;
        # skip the 'embed' wrapper segment if present.
        if segs[0] == "embed":
            return segs[-1] if len(segs) > 1 else None
        return segs[0]
