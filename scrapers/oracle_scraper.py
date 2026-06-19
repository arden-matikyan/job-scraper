"""Oracle Recruiting Cloud (ORC) Candidate Experience scraper.

Oracle's own careers site (careers.oracle.com) and every Fusion-HCM tenant expose
a public JSON REST API. There are no numbered pages in the UI — a "Show more
results" button just bumps the API ``offset`` — so we paginate that ourselves.

Browse URL (Oracle):
    https://careers.oracle.com/en/sites/jobsearch/jobs?keyword=software&selectedLocationsFacet=300000000149325
Backend host: careers.oracle.com redirects API calls to its Fusion pod, so we map
it to ``eeho.fa.us2.oraclecloud.com``. Other tenants are already on *.oraclecloud.com.

    list   GET {host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions
           ?onlyData=true&finder=findReqs;siteNumber=CX_1,keyword=...,limit,offset
           -> items[0].requisitionList[] (id/title/location, no full description)
           -> items[0].TotalJobsCount drives pagination
    detail GET {host}/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails
           ?expand=all&onlyData=true&finder=ById;Id={id},siteNumber=CX_1
           -> External{Description,Qualifications,Responsibilities}Str (HTML)
"""
from __future__ import annotations

import re
from typing import Iterator, Optional
from urllib.parse import parse_qs, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

# careers.oracle.com is a vanity host; its API lives on this Fusion pod.
_ORACLE_API_HOST = "eeho.fa.us2.oraclecloud.com"
_SITE_RE = re.compile(r"^CX_?\d+$", re.I)
# Browse-URL query params we forward verbatim into the API finder (facet filters).
_FACET_PARAMS = (
    "selectedLocationsFacet",
    "selectedWorkLocationsFacet",
    "selectedCategoriesFacet",
    "selectedOrganizationsFacet",
    "selectedTitlesFacet",
    "selectedPostingDatesFacet",
    "selectedWorkplaceTypesFacet",
    "selectedFlexFieldsFacets",
)


class OracleScraper(BaseScraper):
    SCRAPER_KEY = "oracle"
    SITE_HINTS = [
        "careers.oracle.com",
        "/hcmui/candidateexperience",
        "oraclecloud.com/hcmui",
        "recruitingcejobrequisitions",
    ]
    PRIORITY = 10  # public JSON API, no browser needed

    LIST_API = "/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
    DETAIL_API = "/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails"
    LIST_EXPAND = "requisitionList.secondaryLocations,flexFieldsFacet.values"

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        api_host, site, job_base, facets = self._parse_url(url, cfg)
        if not api_host:
            self.log.error("Could not parse Oracle careers URL %s", url)
            return

        list_url = f"https://{api_host}{self.LIST_API}"
        company = company_name or "Oracle"
        limit = int(cfg.get("page_limit", 25))
        max_pages = int(cfg.get("max_pages", 200))

        offset = 0
        for _ in range(max_pages):
            finder = self._build_finder(site, facets, limit, offset)
            data = self.http.get_json(
                list_url,
                params={"onlyData": "true", "expand": self.LIST_EXPAND, "finder": finder},
                headers={"Accept": "application/json"},
            )
            search = (data or {}).get("items") or []
            if not search:
                break
            block = search[0]
            reqs = block.get("requisitionList") or []
            if not reqs:
                break
            for rq in reqs:
                try:
                    rid = rq.get("Id")
                    if rid is None:
                        continue
                    source_url = f"{job_base}/job/{rid}"
                    if source_url in self.seen_urls:
                        yield RawJob(
                            source_url=source_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                            job_id=str(rid), title=rq.get("Title"), company=company,
                            location=rq.get("PrimaryLocation"),
                            already_seen=True, platform="oracle",
                        )
                        continue
                    job = self._detail(rq, api_host, site, source_url, company)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("Oracle req parse failed: %s", exc)

            total = int(block.get("TotalJobsCount") or 0)
            offset += limit
            if offset >= total:
                break

    def _detail(
        self, rq: dict, api_host: str, site: str, source_url: str, company: str
    ) -> Optional[RawJob]:
        rid = rq.get("Id")
        finder = f"ById;Id={rid},siteNumber={site}"
        data = self.http.get_json(
            f"https://{api_host}{self.DETAIL_API}",
            params={"expand": "all", "onlyData": "true", "finder": finder},
            headers={"Accept": "application/json"},
        )
        detail = ((data or {}).get("items") or [{}])[0]

        # Full posting body lives across three HTML blocks on the detail record;
        # fall back to the list-view short description if the detail fetch failed.
        parts = [
            html_to_text(detail.get(k) or "")
            for k in ("ExternalDescriptionStr", "ExternalResponsibilitiesStr", "ExternalQualificationsStr")
        ]
        raw_text = "\n\n".join(p for p in parts if p)
        if not raw_text:
            raw_text = html_to_text(rq.get("ShortDescriptionStr") or "") or (rq.get("Title") or "")

        primary = detail.get("PrimaryLocation") or rq.get("PrimaryLocation")
        secondary = [
            s.get("Name") for s in (detail.get("secondaryLocations") or rq.get("secondaryLocations") or [])
            if isinstance(s, dict) and s.get("Name")
        ]
        locations_all = ([primary] + secondary) if (primary and secondary) else (secondary or None)

        return RawJob(
            source_url=source_url,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=str(rid),
            title=detail.get("Title") or rq.get("Title"),
            company=company,
            location=primary,
            locations_all=locations_all,
            posted_date=detail.get("ExternalPostedStartDate") or rq.get("PostedDate"),
            platform="oracle",
        )

    @staticmethod
    def _build_finder(site: str, facets: dict, limit: int, offset: int) -> str:
        parts = [f"findReqs;siteNumber={site}"]
        for k, v in facets.items():
            parts.append(f"{k}={v}")
        parts += [f"limit={limit}", f"offset={offset}", "sortBy=POSTING_DATES_DESC"]
        return ",".join(parts)

    def _parse_url(
        self, url: str, cfg: dict
    ) -> tuple[Optional[str], str, str, dict]:
        """Return (api_host, siteNumber, public_job_base, facet_filters)."""
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if not host:
            return None, "", "", {}
        # careers.oracle.com proxies to a Fusion pod; tenant hosts serve the API directly.
        api_host = _ORACLE_API_HOST if "careers.oracle.com" in host else host

        # Public job-detail base: everything up to .../sites/{frontendSite}
        m = re.match(r"(.*/sites/[^/]+)", f"{p.scheme}://{p.netloc}{p.path}")
        job_base = m.group(1) if m else f"{p.scheme}://{p.netloc}{p.path}".rstrip("/").rsplit("/", 1)[0]

        # API siteNumber: a CX_n path segment if present, else config default.
        segs = [s for s in p.path.split("/") if s]
        site = next((s for s in segs if _SITE_RE.match(s)), None) or str(cfg.get("site_number", "CX_1"))

        q = {k: v[0] for k, v in parse_qs(p.query).items() if v}
        facets: dict[str, str] = {}
        if q.get("keyword"):
            facets["keyword"] = q["keyword"]
        for fp in _FACET_PARAMS:
            if q.get(fp):
                facets[fp] = q[fp]
        # Older Oracle links pass the location as ``locationId`` instead of the facet.
        if "selectedLocationsFacet" not in facets and q.get("locationId"):
            facets["selectedLocationsFacet"] = q["locationId"]

        return api_host, site, job_base, facets
