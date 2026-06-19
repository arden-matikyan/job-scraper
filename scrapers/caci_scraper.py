"""Eightfold PCSX scraper — handles any Eightfold-hosted careers site.

Eightfold is a React SPA backed by a PCSX JSON API.  A single GET to the
careers page sets the session cookies needed for subsequent API calls; no
browser or CSRF handling is required.

Search  GET /api/pcsx/search?domain={domain}&query=...&start=N...
          Returns 10 positions per page; ``data.count`` is the total.
          Pagination: increment ``start`` by 10 until positions list is empty.
Detail  GET /api/pcsx/position_details?position_id={id}&domain={domain}
          Returns the full HTML job description in ``data.jobDescription``.

Domain resolution (in order of priority):
  1. Explicit ``?domain=`` query parameter in the tracked URL.
  2. Last two labels of the hostname (e.g. 'searchcareers.caci.com' → 'caci.com').

Known tenants:
  CACI          searchcareers.caci.com      → domain caci.com
  Morgan Stanley morganstanley.eightfold.ai → domain morganstanley.com (from ?domain=)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, Optional
from urllib.parse import parse_qs, urljoin, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_REMOTE_MAP = {
    "remote": "remote",
    "onsite": "on_site",
    "on_site": "on_site",
    "hybrid": "hybrid",
}
# Query params that are UI navigation hints, not search filters — strip before
# forwarding to the API.
_SKIP_PARAMS = frozenset({"start", "pid", "domain"})


def _ts_to_iso(ts: Optional[int]) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


class EightfoldScraper(BaseScraper):
    SCRAPER_KEY = "eightfold"
    SITE_HINTS = ["searchcareers.caci.com", "eightfold.ai/careers"]
    PRIORITY = 10  # clean JSON API, no browser needed

    _SEARCH_PATH = "/api/pcsx/search"
    _DETAIL_PATH = "/api/pcsx/position_details"
    _PER_PAGE = 10  # server caps at 10; num_rec is ignored

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        max_pages = int(cfg.get("max_pages", 200))

        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        qs = parse_qs(parsed.query, keep_blank_values=True)

        # Domain: explicit param beats hostname-derived
        domain = (qs.get("domain") or [None])[0] or self._domain_from_host(parsed.netloc)

        # Forward all non-navigation search params to the API
        search_params: dict[str, str] = {"domain": domain}
        for key, vals in qs.items():
            if key not in _SKIP_PARAMS and vals:
                search_params[key] = vals[0]

        company = company_name or ""

        # One GET to the careers page establishes session cookies
        self.http.get(url)

        total: Optional[int] = None
        for page in range(max_pages):
            start = page * self._PER_PAGE
            if total is not None and start >= total:
                break
            data = self.http.get_json(
                f"{base}{self._SEARCH_PATH}",
                params={**search_params, "start": str(start)},
            )
            if not isinstance(data, dict):
                self.log.error("Eightfold search API returned non-dict at start=%d", start)
                break
            inner = data.get("data") or {}
            if total is None:
                total = int(inner.get("count") or 0)
                self.log.info(
                    "Eightfold (%s): %d total positions", domain, total
                )
            positions = inner.get("positions") or []
            if not positions:
                break
            for pos in positions:
                try:
                    yield from self._process_position(pos, base, domain, company)
                except Exception as exc:
                    self.log.warning("Eightfold position %s failed: %s", pos.get("id"), exc)

    def _process_position(
        self, pos: dict, base: str, domain: str, company: str
    ) -> Iterator[RawJob]:
        pos_id = pos.get("id")
        source_url = urljoin(base, pos.get("positionUrl") or f"/careers/job/{pos_id}")

        listing_locs: list[str] = [s for s in (pos.get("locations") or []) if s]
        posted_date = _ts_to_iso(pos.get("postedTs"))
        display_id = str(pos.get("displayJobId") or pos_id or "")

        if source_url in self.seen_urls:
            yield RawJob(
                source_url=source_url,
                raw_text="",
                scraper_key=self.SCRAPER_KEY,
                job_id=display_id,
                title=pos.get("name"),
                company=company,
                location=listing_locs[0] if listing_locs else None,

                posted_date=posted_date,
                already_seen=True,
                platform="eightfold",
            )
            return

        detail = self._fetch_detail(base, domain, pos_id)
        desc_html = (detail or {}).get("jobDescription") or ""
        raw_text = html_to_text(desc_html) if desc_html else ""
        if not raw_text:
            return

        primary_loc = (detail or {}).get("location") or (listing_locs[0] if listing_locs else None)
        if primary_loc:
            primary_loc = primary_loc.strip()
        locations_all: Optional[list[str]] = listing_locs if len(listing_locs) > 1 else None

        yield RawJob(
            source_url=source_url,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=display_id,
            title=pos.get("name"),
            company=company,
            location=primary_loc,
            locations_all=locations_all,
            posted_date=posted_date,
            platform="eightfold",
        )

    def _fetch_detail(self, base: str, domain: str, pos_id) -> Optional[dict]:
        data = self.http.get_json(
            f"{base}{self._DETAIL_PATH}",
            params={"position_id": str(pos_id), "domain": domain},
        )
        if not isinstance(data, dict):
            return None
        return data.get("data") or None

    @staticmethod
    def _domain_from_host(host: str) -> str:
        """'searchcareers.caci.com' → 'caci.com' (last two host labels)."""
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
