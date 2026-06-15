"""Workday scraper using the public CXS API (verified against live tenants).

A browse URL like
    https://{tenant}.{dc}.myworkdayjobs.com/en-US/{site}
maps to:
    list   POST https://{host}/wday/cxs/{tenant}/{site}/jobs
           body {"appliedFacets":{},"limit":20,"offset":0,"searchText":""}
    detail GET  https://{host}/wday/cxs/{tenant}/{site}{externalPath}
Pagination increments offset by limit until offset >= total.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional
from urllib.parse import parse_qs, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_LOCALE_RE = re.compile(r"^[a-z]{2}-[A-Za-z]{2}$")
_TIME_TYPE = {"full time": "full_time", "part time": "part_time"}


class WorkdayScraper(BaseScraper):
    SCRAPER_KEY = "workday"
    SITE_HINTS = ["myworkdayjobs.com", "workday.com/en-US/", "wd1.myworkdayjobs", "wd5.myworkdayjobs"]
    PRIORITY = 10

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        parsed = self._parse_url(url)
        if not parsed:
            self.log.error("Could not parse Workday tenant/site from %s", url)
            return
        host, tenant, site = parsed
        base = f"https://{host}"
        cxs = f"{base}/wday/cxs/{tenant}/{site}"
        limit = int(cfg.get("page_limit", 20))
        max_pages = int(cfg.get("max_pages", 200))
        search_text = parse_qs(urlparse(url).query).get("q", [""])[0]

        offset = 0
        for _ in range(max_pages):
            body = {"appliedFacets": {}, "limit": limit, "offset": offset, "searchText": search_text}
            data = self.http.post_json(f"{cxs}/jobs", json=body)
            if not isinstance(data, dict):
                break
            postings = data.get("jobPostings") or []
            if not postings:
                break
            for jp in postings:
                try:
                    external_path = jp.get("externalPath")
                    source_url = f"{base}/{site}{external_path}" if external_path else None
                    if source_url and source_url in self.seen_urls:
                        yield RawJob(
                            source_url=source_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                            job_id=self._id_from_path(external_path), title=jp.get("title"),
                            company=company_name, location=jp.get("locationsText"),
                            already_seen=True, platform="workday",
                        )
                        continue
                    job = self._parse_job(jp, cxs, base, site, company_name)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("Workday job parse failed: %s", exc)
            total = int(data.get("total") or 0)
            offset += limit
            if offset >= total:
                break

    def _parse_job(
        self, jp: dict, cxs: str, base: str, site: str, company_name: Optional[str]
    ) -> Optional[RawJob]:
        external_path = jp.get("externalPath")
        detail = self.http.get_json(f"{cxs}{external_path}") if external_path else None
        info = (detail or {}).get("jobPostingInfo") or {}

        raw_text = html_to_text(info.get("jobDescription") or "") or (jp.get("title") or "")
        primary = info.get("location") or jp.get("locationsText")
        add_locs = info.get("additionalLocations") or []
        locations_all = None
        if add_locs:
            locations_all = ([primary] + list(add_locs)) if primary else list(add_locs)

        time_type = (info.get("timeType") or "").strip().lower()
        # Construct from externalPath so the URL is identical whether built here
        # (after the detail fetch) or pre-fetch in the loop above — needed for the
        # seen_urls skip to match. This equals Workday's canonical externalUrl.
        source_url = f"{base}/{site}{external_path}" if external_path else (info.get("externalUrl") or base)

        return RawJob(
            source_url=source_url,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=info.get("jobRequisitionId") or self._id_from_path(external_path) or info.get("id"),
            title=info.get("title") or jp.get("title"),
            company=company_name,
            location=primary,
            locations_all=locations_all,
            employment_type=_TIME_TYPE.get(time_type),
            posted_date=info.get("startDate") or jp.get("postedOn"),
            platform="workday",
        )

    @staticmethod
    def _id_from_path(external_path: Optional[str]) -> Optional[str]:
        if not external_path:
            return None
        tail = external_path.rstrip("/").rsplit("/", 1)[-1]
        # externalPath ends with ..._JR12345 or ..._01843761-1
        m = re.search(r"_([A-Za-z0-9\-]+)$", tail)
        return m.group(1) if m else tail or None

    @staticmethod
    def _parse_url(url: str) -> Optional[tuple[str, str, str]]:
        p = urlparse(url)
        host = p.netloc
        if not host:
            return None
        tenant = host.split(".")[0]
        segs = [s for s in p.path.split("/") if s]
        if segs and _LOCALE_RE.match(segs[0]):
            segs = segs[1:]
        if not segs:
            return None
        site = segs[0]
        if not tenant or not site:
            return None
        return host, tenant, site
