"""Elastic careers scraper (jobs.elastic.co — Laravel + Elastic App Search).

jobs.elastic.co is a Laravel SPA whose job board is backed by Elastic App Search
(the engine ``jobs-production``). The browser never talks to App Search directly:
it POSTs App Search query bodies to a **same-origin Laravel proxy**

    POST https://jobs.elastic.co/api/appSearch

which injects the real engine credentials server-side. The proxy is guarded by
Laravel's CSRF middleware, so each run must first GET a board page to obtain the
``XSRF-TOKEN`` cookie and echo it back in the ``X-XSRF-TOKEN`` header (the value is
URL-encoded in the cookie and must be decoded). The shared httpx client carries the
matching ``laravel_session`` cookie automatically on the subsequent POSTs.

The search response carries the **full job description** inline (``content.raw``),
so there is no separate detail fetch — one POST per page of results. The board is
ultimately Greenhouse-backed (detail pages redirect to ``?gh_jid={req_id}``).

URL → query mapping:
  * Path ``/jobs/<column>/<value>`` becomes an App Search value filter. The board
    aliases the ``department`` column to the ``subdivision`` field (matching the
    site's own JS), e.g. ``/jobs/department/engineering`` → ``subdivision: Engineering``.
  * Query-string ``filters[N][field|type][values][M]`` params (the Search-UI
    encoding, e.g. ``filters[0][field]=location``) are passed straight through.

Canonical job URL (stable dedup key): ``https://jobs.elastic.co/jobs/{url}`` where
``url`` is the relative path App Search returns (it already ends in the req id).

Optional config (per-company in tracked_urls.yaml or scraper_configs):
  keywords   : free-text search term (App Search ``query``).
  max_pages  : safety cap on pagination (default 50).
  page_size  : results per request (default 100, App Search max).
"""
from __future__ import annotations

import re
from typing import Iterator, Optional
from urllib.parse import parse_qsl, unquote, urlparse

from scrapers.base import BaseScraper, RawJob, parse_posted_date

_API_PATH = "/api/appSearch"
_FILTER_RE = re.compile(r"^filters\[(\d+)\]\[(field|type)\]$")
_VALUE_RE = re.compile(r"^filters\[(\d+)\]\[values\]\[(\d+)\]$")

# Fields requested back from App Search (keeps the response small and stable).
_RESULT_FIELDS = {
    "title": {"raw": {}},
    "content": {"raw": {}},
    "location": {"raw": {}},
    "hybrid_locations": {"raw": {}},
    "subdivision": {"raw": {}},
    "category": {"raw": {}},
    "job_type": {"raw": {}},
    "req_id": {"raw": {}},
    "url": {"raw": {}},
    "created_at": {"raw": {}},
}


class ElasticScraper(BaseScraper):
    SCRAPER_KEY = "elastic"
    SITE_HINTS = ["jobs.elastic.co"]
    PRIORITY = 10  # same-origin JSON proxy, no browser

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        company = company_name or "Elastic"
        page_size = min(int(cfg.get("page_size", 100)), 100)
        max_pages = int(cfg.get("max_pages", 50))

        p = urlparse(url)
        site = f"{p.scheme}://{p.netloc}"
        api_url = f"{site}{_API_PATH}"

        token = self._csrf_token(url)
        if not token:
            self.log.error("Elastic: could not obtain XSRF-TOKEN cookie from %s", url)
            return

        filters = self._parse_filters(p)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-XSRF-TOKEN": token,
            "X-Requested-With": "XMLHttpRequest",
        }
        query = str(cfg.get("keywords", "") or "")

        seen_count = 0
        total: Optional[int] = None
        for current in range(1, max_pages + 1):
            payload = {
                "query": query,
                "result_fields": _RESULT_FIELDS,
                "page": {"size": page_size, "current": current},
                "filters": {"all": filters},
                "sort": [{"_score": "desc"}, {"created_at": "desc"}],
            }
            data = self.http.post_json(api_url, json=payload, headers=headers)
            if not isinstance(data, dict):
                self.log.warning("Elastic: non-JSON response on page %d", current)
                break
            results = data.get("results") or []
            meta_page = (data.get("meta") or {}).get("page") or {}
            if total is None:
                total = int(meta_page.get("total_results") or 0)
            if not results:
                break
            for raw in results:
                seen_count += 1
                try:
                    rj = self._to_rawjob(raw, site, company)
                    if rj is not None:
                        yield rj
                except Exception as exc:
                    self.log.warning("Elastic: job parse failed: %s", exc)
            total_pages = int(meta_page.get("total_pages") or 0)
            if total_pages and current >= total_pages:
                break
            if len(results) < page_size:
                break

    # ------------------------------------------------------------------ helpers
    def _csrf_token(self, url: str) -> Optional[str]:
        """GET a board page to obtain the (URL-encoded) XSRF-TOKEN cookie."""
        resp = self.http.get(url)
        if resp is None:
            return None
        try:
            token = resp.cookies.get("XSRF-TOKEN")
        except Exception:
            token = None
        return unquote(token) if token else None

    @staticmethod
    def _parse_filters(p) -> list[dict]:
        """Build App Search ``filters.all`` from the board URL.

        Path ``/jobs/<column>/<value>`` and the Search-UI ``filters[N]...`` query
        params each become a ``{type: [{field: value}, ...]}`` group.
        """
        out: list[dict] = []

        parts = [seg for seg in p.path.split("/") if seg]
        if "jobs" in parts:
            i = parts.index("jobs")
            if len(parts) >= i + 3:
                column, value = parts[i + 1], parts[i + 2]
                field = "subdivision" if column == "department" else column
                val = unquote(value).replace("-", " ").title()
                out.append({"any": [{field: val}]})

        groups: dict[int, dict] = {}
        for k, v in parse_qsl(p.query):
            m = _FILTER_RE.match(k)
            if m:
                groups.setdefault(int(m.group(1)), {})[m.group(2)] = v
                continue
            m = _VALUE_RE.match(k)
            if m:
                g = groups.setdefault(int(m.group(1)), {})
                g.setdefault("values", {})[int(m.group(2))] = v
        for _, g in sorted(groups.items()):
            field = g.get("field")
            vals = g.get("values")
            if not field or not vals:
                continue
            ftype = g.get("type") or "any"
            ordered = [vals[i] for i in sorted(vals)]
            out.append({ftype: [{field: val} for val in ordered]})

        return out

    def _to_rawjob(self, raw: dict, site: str, company: str) -> Optional[RawJob]:
        rel = self._raw(raw, "url")
        req_id = self._raw(raw, "req_id")
        if not rel:
            return None
        source_url = f"{site}/jobs/{rel.lstrip('/')}"
        job_id = req_id or None
        title = self._raw(raw, "title") or None
        location = self._raw(raw, "location") or None
        locations_all = self._all_locations(raw)
        posted_date = parse_posted_date(self._raw(raw, "created_at"))

        if source_url in self.seen_urls:
            return RawJob(
                source_url=source_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                job_id=job_id, title=title, company=company, location=location,
                locations_all=locations_all, posted_date=posted_date,
                already_seen=True, platform="greenhouse",
            )

        raw_text = (self._raw(raw, "content") or "").strip()
        attrs = self._attributes_block(raw)
        if attrs:
            raw_text = (raw_text + "\n\n" + attrs).strip() if raw_text else attrs
        if not raw_text:
            return None

        return RawJob(
            source_url=source_url,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=job_id,
            title=title,
            company=company,
            location=location,
            locations_all=locations_all,
            posted_date=posted_date,
            platform="greenhouse",
        )

    @staticmethod
    def _raw(raw: dict, key: str) -> str:
        """App Search wraps every field as ``{"raw": value}``; pull it out as str."""
        field = raw.get(key)
        if isinstance(field, dict):
            val = field.get("raw")
        else:
            val = field
        return str(val).strip() if val not in (None, "") else ""

    @classmethod
    def _all_locations(cls, raw: dict) -> Optional[list[str]]:
        out: list[str] = []
        primary = cls._raw(raw, "location")
        if primary:
            out.append(primary)
        for extra in re.split(r"[;\n]", cls._raw(raw, "hybrid_locations")):
            name = extra.strip()
            if name and name not in out:
                out.append(name)
        return out if len(out) > 1 else None

    @classmethod
    def _attributes_block(cls, raw: dict) -> str:
        """Surface the App Search facet fields the description text may omit."""
        lines: list[str] = []
        for label, key in (("Department", "subdivision"), ("Team", "category"),
                           ("Work Type", "job_type")):
            val = cls._raw(raw, key)
            if val:
                lines.append(f"{label}: {val}")
        return "\n".join(lines)
