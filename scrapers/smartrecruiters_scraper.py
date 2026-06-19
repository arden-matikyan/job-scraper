"""SmartRecruiters public API scraper.

List:   https://api.smartrecruiters.com/v1/companies/{slug}/postings?offset&limit
Detail: https://api.smartrecruiters.com/v1/companies/{slug}/postings/{id}
The list is paginated (offset/limit=100); full description text lives in the
detail response's jobAd.sections, so we fetch each posting's detail.
"""
from __future__ import annotations

from typing import Iterator, Optional
from urllib.parse import urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_EMPLOYMENT = {
    "full-time": "full_time",
    "part-time": "part_time",
    "contract": "contract",
    "internship": "internship",
    "temporary": "contract",
}


class SmartRecruitersScraper(BaseScraper):
    SCRAPER_KEY = "smartrecruiters"
    SITE_HINTS = ["smartrecruiters.com", "jobs.smartrecruiters"]
    PRIORITY = 10

    LIST_API = "https://api.smartrecruiters.com/v1/companies/{slug}/postings"
    DETAIL_API = "https://api.smartrecruiters.com/v1/companies/{slug}/postings/{pid}"
    PUBLIC_URL = "https://jobs.smartrecruiters.com/{slug}/{pid}"

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        slug = self._slug(url)
        if not slug:
            self.log.error("Could not extract SmartRecruiters slug from %s", url)
            return
        limit = int(cfg.get("page_limit", 100))
        offset = 0
        while True:
            data = self.http.get_json(
                self.LIST_API.format(slug=slug),
                params={"offset": offset, "limit": limit},
            )
            if not isinstance(data, dict):
                break
            content = data.get("content") or []
            if not content:
                break
            for posting in content:
                try:
                    pid = posting.get("id")
                    source_url = self.PUBLIC_URL.format(slug=slug, pid=pid) if pid else None
                    if source_url and source_url in self.seen_urls:
                        yield RawJob(
                            source_url=source_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                            job_id=str(pid), title=posting.get("name"),
                            company=company_name or (posting.get("company") or {}).get("name"),
                            already_seen=True, platform="smartrecruiters",
                        )
                        continue
                    job = self._parse(posting, slug, company_name)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("Failed to parse SmartRecruiters posting: %s", exc)
            total = int(data.get("totalFound") or 0)
            offset += limit
            if offset >= total:
                break

    def _parse(self, posting: dict, slug: str, company_name: Optional[str]) -> Optional[RawJob]:
        pid = posting.get("id")
        if not pid:
            return None
        detail = self.http.get_json(self.DETAIL_API.format(slug=slug, pid=pid)) or {}
        sections = ((detail.get("jobAd") or {}).get("sections")) or {}
        parts: list[str] = []
        for key in ("companyDescription", "jobDescription", "qualifications", "additionalInformation"):
            sec = sections.get(key) or {}
            t = html_to_text(sec.get("text") or "")
            if t:
                parts.append(t)
        raw_text = "\n\n".join(parts) or (posting.get("name") or "")

        loc = posting.get("location") or {}
        location_str = ", ".join(
            x for x in (loc.get("city"), loc.get("region"), loc.get("country")) if x
        )
        emp = ((posting.get("typeOfEmployment") or {}).get("label") or "").strip().lower()

        return RawJob(
            source_url=self.PUBLIC_URL.format(slug=slug, pid=pid),
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=str(pid),
            title=posting.get("name"),
            company=company_name or (posting.get("company") or {}).get("name"),
            location=location_str or None,
            posted_date=posting.get("releasedDate"),
            platform="smartrecruiters",
        )

    @staticmethod
    def _slug(url: str) -> Optional[str]:
        p = urlparse(url)
        segs = [s for s in p.path.split("/") if s]
        return segs[0] if segs else None
