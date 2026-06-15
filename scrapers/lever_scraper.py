"""Lever public postings API scraper.

Endpoint: https://api.lever.co/v0/postings/{slug}?mode=json
Returns a JSON list of postings, each with plain-text + HTML list sections that
we stitch into a single raw_text. No per-job detail fetch needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, Optional
from urllib.parse import urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_EMPLOYMENT = {
    "full-time": "full_time",
    "part-time": "part_time",
    "contract": "contract",
    "internship": "internship",
    "intern": "internship",
    "temporary": "contract",
}
_REMOTE = {"remote": "remote", "on-site": "onsite", "onsite": "onsite", "hybrid": "hybrid"}


class LeverScraper(BaseScraper):
    SCRAPER_KEY = "lever_api"
    SITE_HINTS = ["lever.co", "jobs.lever"]
    PRIORITY = 10

    API = "https://api.lever.co/v0/postings/{slug}?mode=json"

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        slug = self._slug(url)
        if not slug:
            self.log.error("Could not extract Lever slug from %s", url)
            return
        data = self.http.get_json(self.API.format(slug=slug))
        if not isinstance(data, list):
            self.log.error("Lever API returned no postings for slug %r", slug)
            return
        for posting in data:
            try:
                yield self._parse(posting, company_name)
            except Exception as exc:
                self.log.warning("Failed to parse Lever posting: %s", exc)

    def _parse(self, posting: dict, company_name: Optional[str]) -> RawJob:
        parts: list[str] = []
        if posting.get("descriptionPlain"):
            parts.append(posting["descriptionPlain"])
        for lst in posting.get("lists", []) or []:
            heading = (lst.get("text") or "").strip()
            body = html_to_text(lst.get("content") or "")
            if heading or body:
                parts.append(f"{heading}\n{body}".strip())
        if posting.get("additionalPlain"):
            parts.append(posting["additionalPlain"])
        raw_text = "\n\n".join(p for p in parts if p) or (posting.get("text") or "")

        cats = posting.get("categories") or {}
        location = cats.get("location")
        all_locs = cats.get("allLocations") or ([location] if location else None)
        commitment = (cats.get("commitment") or "").strip().lower()
        workplace = (posting.get("workplaceType") or "").strip().lower()

        return RawJob(
            source_url=posting.get("hostedUrl") or posting.get("applyUrl") or "",
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=str(posting.get("id")) if posting.get("id") else None,
            title=posting.get("text"),
            company=company_name,
            location=location,
            locations_all=all_locs if all_locs and len(all_locs) > 1 else None,
            employment_type=_EMPLOYMENT.get(commitment),
            remote_type=_REMOTE.get(workplace),
            posted_date=self._posted(posting.get("createdAt")),
            platform="lever",
        )

    @staticmethod
    def _posted(created_at) -> Optional[str]:
        if isinstance(created_at, (int, float)) and created_at > 0:
            try:
                return datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).date().isoformat()
            except Exception:
                return None
        return None

    @staticmethod
    def _slug(url: str) -> Optional[str]:
        p = urlparse(url)
        segs = [s for s in p.path.split("/") if s]
        return segs[0] if segs else None
