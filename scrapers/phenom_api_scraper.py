"""Phenom People public JSON API scraper (Publicis Groupe + S&P Global).

Phenom career sites render listings client-side, but they back the SPA with a
public JSON endpoint that needs no browser session:

    GET {base}/api/jobs?{search params}&page={N}

returning ``{"jobs": [{"data": {...}}, ...], "totalCount": N, ...}`` where each
``data`` object already carries the **full description**, location, posted date,
and req id — so there is no separate detail fetch. The search params from the
tracked URL (keywords, location, woe, regionCode, stretchUnit, stretch, sortBy)
are passed straight through, so the same filtering the careers page applies is
preserved. Pagination is 1-based ``page``; 10 jobs/page; stop at ``totalCount``.

Canonical job URL (stable, used for dedup): ``{base}/jobs/{slug}``.

Note: careers.freddiemac.com is also Phenom but gates its API behind a same-origin
browser session (see freddiemac_scraper.py). This scraper is for the Phenom sites
whose /api/jobs is reachable over plain HTTP.

Date filtering: set ``oldest_date: "YYYY-MM-DD"`` (per-company in tracked_urls.yaml
or in scraper_configs) to drop postings older than that.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text, parse_posted_date


class PhenomApiScraper(BaseScraper):
    SCRAPER_KEY = "phenom_api"
    SITE_HINTS = ["careers.spglobal.com", "careers.publicisgroupe.com"]
    PRIORITY = 10  # plain-HTTP JSON API, no browser

    _PAGE_SIZE = 10

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        max_pages = int(cfg.get("max_pages", 300))
        company = company_name or "?"

        oldest = self._parse_oldest(cfg.get("oldest_date"))

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        # Pass the careers-page search params straight through to /api/jobs; we
        # drive `page` ourselves.
        params = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() != "page"]

        seen_count = 0
        total: Optional[int] = None

        for pno in range(1, max_pages + 1):
            api_url = f"{base}/api/jobs?{urlencode(params + [('page', pno)])}"
            data = self.http.get_json(
                api_url,
                headers={"Accept": "application/json", "Referer": url,
                         "X-Requested-With": "XMLHttpRequest"},
            )
            if not isinstance(data, dict):
                self.log.warning("Phenom: non-JSON / empty response at page %d", pno)
                break
            jobs = data.get("jobs") or []
            if total is None:
                total = data.get("totalCount")
            if not jobs:
                break

            for job in jobs:
                seen_count += 1
                try:
                    rj = self._to_rawjob(job, base, company, oldest)
                    if rj is not None:
                        yield rj
                except Exception as exc:
                    self.log.warning("Phenom: job parse failed: %s", exc)

            if total is not None and seen_count >= int(total):
                break
            if len(jobs) < self._PAGE_SIZE:
                break

    # ------------------------------------------------------------------ helpers
    def _to_rawjob(
        self, job: dict, base: str, company: str, oldest: Optional[_date]
    ) -> Optional[RawJob]:
        d = (job or {}).get("data") or {}
        slug = d.get("slug") or d.get("req_id")
        if not slug:
            return None
        source_url = f"{base}/jobs/{slug}"
        posted = parse_posted_date(d.get("posted_date"))

        if oldest and posted:
            try:
                if _date.fromisoformat(posted) < oldest:
                    return None  # older than the cutoff — skip
            except ValueError:
                pass

        if source_url in self.seen_urls:
            return RawJob(
                source_url=source_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                job_id=d.get("req_id"), title=d.get("title"), company=company,
                location=d.get("short_location") or d.get("full_location"),
                posted_date=posted, already_seen=True, platform="phenompeople",
            )

        raw_text = html_to_text(d.get("description") or "")
        if not raw_text:
            return None

        return RawJob(
            source_url=source_url,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=d.get("req_id"),
            title=d.get("title"),
            company=company,
            location=d.get("short_location") or self._primary_location(d),
            locations_all=self._all_locations(d),
            posted_date=posted,
            platform="phenompeople",
        )

    @staticmethod
    def _primary_location(d: dict) -> Optional[str]:
        parts = [d.get("city"), d.get("state")]
        return ", ".join(p for p in parts if p) or d.get("country") or None

    @staticmethod
    def _all_locations(d: dict) -> Optional[list[str]]:
        out: list[str] = []
        primary = ", ".join(p for p in [d.get("city"), d.get("state")] if p)
        if primary:
            out.append(primary)
        for loc in d.get("additional_locations") or []:
            if isinstance(loc, dict):
                s = ", ".join(p for p in [loc.get("city"), loc.get("state")] if p)
                if s and s not in out:
                    out.append(s)
        return out if len(out) > 1 else None

    def _parse_oldest(self, raw) -> Optional[_date]:
        if not raw:
            return None
        try:
            return _date.fromisoformat(str(raw))
        except ValueError:
            self.log.warning("Phenom: invalid oldest_date %r — ignoring", raw)
            return None
