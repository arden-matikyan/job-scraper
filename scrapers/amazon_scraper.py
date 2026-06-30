"""Amazon.jobs public JSON API scraper.

amazon.jobs renders its search page client-side, but backs it with a public JSON
endpoint that needs no browser session: the careers search path has a ``.json``
twin that returns the listings directly.

    GET https://www.amazon.jobs/en/search.json?{search params}&offset={N}

returns ``{"jobs": [{...}], "hits": N, "error": null}`` where each job object
already carries the **full description**, basic/preferred qualifications, location,
posted date and the iCIMS req id — so there is no separate detail fetch. The
search params from the tracked URL (base_query, country[], industry_experience,
sort, radius...) are passed straight through, so the same filtering the careers
page applies is preserved. We drive ``offset`` ourselves in ``result_limit``-sized
pages and stop at ``hits`` (Amazon caps this at 10000) or the first empty page.

Canonical job URL (stable, used for dedup): ``https://www.amazon.jobs{job_path}``.

Date filtering: set ``oldest_date: "YYYY-MM-DD"`` (per-company in tracked_urls.yaml
or in scraper_configs) to drop postings older than that.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from scrapers.base import BaseScraper, RawJob, html_to_text, parse_posted_date


class AmazonScraper(BaseScraper):
    SCRAPER_KEY = "amazon"
    SITE_HINTS = ["amazon.jobs"]
    PRIORITY = 10  # plain-HTTP JSON API, no browser

    _PAGE_SIZE = 100  # amazon.jobs honours result_limit up to 100

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        max_pages = int(cfg.get("max_pages", 300))
        company = company_name or "Amazon"

        oldest = self._parse_oldest(cfg.get("oldest_date"))

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        # The .json twin of the search page returns listings directly. Pass the
        # careers-page search params straight through, but drive offset/result_limit
        # ourselves for efficient pagination.
        json_path = p.path if p.path.endswith(".json") else p.path.rstrip("/") + ".json"
        params = [
            (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in ("offset", "result_limit")
        ]

        seen_count = 0
        hits: Optional[int] = None

        for pno in range(max_pages):
            offset = pno * self._PAGE_SIZE
            page_params = params + [("offset", offset), ("result_limit", self._PAGE_SIZE)]
            api_url = urlunparse(p._replace(path=json_path, query=urlencode(page_params)))
            data = self.http.get_json(
                api_url,
                headers={"Accept": "application/json", "Referer": url,
                         "X-Requested-With": "XMLHttpRequest"},
            )
            if not isinstance(data, dict):
                self.log.warning("Amazon: non-JSON / empty response at offset %d", offset)
                break
            jobs = data.get("jobs") or []
            if hits is None:
                hits = data.get("hits")
            if not jobs:
                break

            for job in jobs:
                seen_count += 1
                try:
                    rj = self._to_rawjob(job, base, company, oldest)
                    if rj is not None:
                        yield rj
                except Exception as exc:
                    self.log.warning("Amazon: job parse failed: %s", exc)

            if hits is not None and seen_count >= int(hits):
                break
            if len(jobs) < self._PAGE_SIZE:
                break

    # ------------------------------------------------------------------ helpers
    def _to_rawjob(
        self, job: dict, base: str, company: str, oldest: Optional[_date]
    ) -> Optional[RawJob]:
        job_path = job.get("job_path")
        if not job_path:
            return None
        source_url = f"{base}{job_path}"
        posted = parse_posted_date(job.get("posted_date"))

        if oldest and posted:
            try:
                if _date.fromisoformat(posted) < oldest:
                    return None  # older than the cutoff — skip
            except ValueError:
                pass

        location = job.get("normalized_location") or job.get("location")

        if source_url in self.seen_urls:
            return RawJob(
                source_url=source_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                job_id=job.get("id_icims"), title=job.get("title"), company=company,
                location=location, posted_date=posted, already_seen=True,
                platform="amazon",
            )

        raw_text = self._build_raw_text(job)
        if not raw_text:
            return None

        return RawJob(
            source_url=source_url,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=job.get("id_icims"),
            title=job.get("title"),
            company=company,
            location=location,
            posted_date=posted,
            platform="amazon",
        )

    @staticmethod
    def _build_raw_text(job: dict) -> str:
        """Stitch description + qualifications into one plain-text block for the LLM."""
        sections = [
            ("", job.get("description")),
            ("BASIC QUALIFICATIONS", job.get("basic_qualifications")),
            ("PREFERRED QUALIFICATIONS", job.get("preferred_qualifications")),
        ]
        parts: list[str] = []
        for label, html in sections:
            text = html_to_text(html or "")
            if text:
                parts.append(f"{label}\n{text}" if label else text)
        return "\n\n".join(parts).strip()

    def _parse_oldest(self, raw) -> Optional[_date]:
        if not raw:
            return None
        try:
            return _date.fromisoformat(str(raw))
        except ValueError:
            self.log.warning("Amazon: invalid oldest_date %r — ignoring", raw)
            return None
