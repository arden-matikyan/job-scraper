"""Accenture Federal Services careers scraper (Salesforce Experience Cloud + Greenhouse).

smartafs.my.site.com is a Salesforce Aura (Lightning Experience Cloud) site whose
job board is backed by Greenhouse. Listings are fetched client-side through the
guest Aura Apex endpoint:

    POST {site}/s/sfsites/aura?r=N&aura.ApexAction.execute=1

calling the Apex controller ``RRGreenhouseIntergration.getJobBoard`` (the typo is
in the deployed class name) with a ``searchOptions`` block. The response carries
the **full job description HTML** (``jobContent``) inline, so there is no separate
detail fetch — one POST per page of results.

The Aura request needs two volatile tokens that change on every Salesforce deploy:
the framework uid (``fwuid``) and the app markup token (``loaded``). Both are read
fresh from the search page's HTML on each run, so the scraper survives redeploys.
``aura.token`` is the literal string ``"null"`` for an unauthenticated guest.

Canonical job URL (stable dedup key): the Greenhouse ``jobUrl``
(``https://boards.greenhouse.io/accenturefederalservices/jobs/{id}...``).

Optional config (per-company in tracked_urls.yaml or scraper_configs):
  keywords / locations : passed straight into the Greenhouse search.
  max_pages            : safety cap on pagination (default 50).
  page_size            : results per request (default 100).
"""
from __future__ import annotations

import json
import re
from typing import Iterator, Optional
from urllib.parse import urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_APEX_DESCRIPTOR = "aura://ApexActionController/ACTION$execute"
_CLASSNAME = "RRGreenhouseIntergration"
_METHOD = "getJobBoard"

_FWUID_RE = re.compile(r'"fwuid":"([^"]+)"')
_LOADED_RE = re.compile(r'APPLICATION@markup://siteforce:communityApp":"([^"]+)"')


class AFSScraper(BaseScraper):
    SCRAPER_KEY = "afs"
    SITE_HINTS = ["smartafs.my.site.com"]
    PRIORITY = 10  # guest JSON Apex endpoint, no browser

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        company = company_name or "Accenture Federal Services"
        page_size = int(cfg.get("page_size", 100))
        max_pages = int(cfg.get("max_pages", 50))

        p = urlparse(url)
        site = f"{p.scheme}://{p.netloc}"
        # Experience Cloud site prefix is everything before '/s/' (e.g. '/careers').
        prefix = p.path.split("/s/", 1)[0] if "/s/" in p.path else ""
        aura_url = f"{site}{prefix}/s/sfsites/aura?r=1&aura.ApexAction.execute=1"
        page_uri = p.path or f"{prefix}/s/search-jobs"

        ctx = self._aura_context(url)
        if ctx is None:
            self.log.error("AFS: could not read Aura context (fwuid/loaded) from %s", url)
            return

        search = {
            "keywords": str(cfg.get("keywords", "") or ""),
            "locations": str(cfg.get("locations", "") or ""),
            "clearances": "",
            "workTypes": "",
            "jobCategories": "",
            "experienceTypes": "",
            "resultsPerPage": page_size,
            "pageNumber": 1,
        }

        seen_count = 0
        total: Optional[int] = None
        for pno in range(1, max_pages + 1):
            search["pageNumber"] = pno
            rv = self._call(aura_url, ctx, search, page_uri)
            if rv is None:
                break
            if total is None:
                total = int(rv.get("totalResults") or 0)
            jobs = rv.get("responseLst") or []
            if not jobs:
                break
            for raw in jobs:
                seen_count += 1
                try:
                    rj = self._to_rawjob(raw, company)
                    if rj is not None:
                        yield rj
                except Exception as exc:
                    self.log.warning("AFS: job parse failed: %s", exc)
            if total and seen_count >= total:
                break
            if len(jobs) < page_size:
                break

    # ------------------------------------------------------------------ helpers
    def _aura_context(self, url: str) -> Optional[dict]:
        """Read the volatile fwuid + app markup token from the live search page."""
        html = self.http.get_text(url)
        if not html:
            return None
        fw = _FWUID_RE.search(html)
        lt = _LOADED_RE.search(html)
        if not fw or not lt:
            return None
        return {
            "mode": "PROD",
            "fwuid": fw.group(1),
            "app": "siteforce:communityApp",
            "loaded": {"APPLICATION@markup://siteforce:communityApp": lt.group(1)},
            "dn": [],
            "globals": {},
            "uad": True,
        }

    def _call(
        self, aura_url: str, ctx: dict, search: dict, page_uri: str
    ) -> Optional[dict]:
        message = {
            "actions": [{
                "id": "0;a",
                "descriptor": _APEX_DESCRIPTOR,
                "callingDescriptor": "UNKNOWN",
                "params": {
                    "namespace": "",
                    "classname": _CLASSNAME,
                    "method": _METHOD,
                    "params": {"searchOptions": dict(search)},
                    "cacheable": False,
                    "isContinuation": False,
                },
            }]
        }
        data = self.http.post_json(
            aura_url,
            data={
                "message": json.dumps(message),
                "aura.context": json.dumps(ctx),
                "aura.token": "null",
                "aura.pageURI": page_uri,
            },
            headers={
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded",
                "X-SFDC-Page-Scope-Id": "0",
            },
        )
        if not isinstance(data, dict):
            self.log.warning("AFS: non-JSON Aura response")
            return None
        actions = data.get("actions") or []
        if not actions or actions[0].get("state") != "SUCCESS":
            self.log.warning("AFS: Aura action not successful: %s",
                             actions[0].get("state") if actions else "no actions")
            return None
        outer = actions[0].get("returnValue") or {}
        rv = outer.get("returnValue") or {}
        if rv.get("isError"):
            self.log.warning("AFS: getJobBoard reported isError")
            return None
        return rv

    def _to_rawjob(self, raw: dict, company: str) -> Optional[RawJob]:
        source_url = raw.get("jobUrl")
        gh_id = raw.get("id")
        if not source_url or gh_id is None:
            return None
        job_id = str(gh_id)
        title = (raw.get("title") or "").strip() or None
        location = ((raw.get("location") or {}).get("name") or "").strip() or None
        locations_all = self._all_locations(raw)

        if source_url in self.seen_urls:
            return RawJob(
                source_url=source_url, raw_text="", scraper_key=self.SCRAPER_KEY,
                job_id=job_id, title=title, company=company, location=location,
                locations_all=locations_all, already_seen=True, platform="greenhouse",
            )

        raw_text = html_to_text(raw.get("jobContent") or "")
        attrs = self._attributes_block(raw.get("clearance") or [])
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
            platform="greenhouse",
        )

    @staticmethod
    def _all_locations(raw: dict) -> Optional[list[str]]:
        out: list[str] = []
        for office in raw.get("offices") or []:
            if isinstance(office, dict):
                name = (office.get("name") or "").strip()
                if name and name not in out:
                    out.append(name)
        return out if len(out) > 1 else None

    @staticmethod
    def _attributes_block(clearance: list) -> str:
        """Flatten the Greenhouse custom fields (clearance, workplace, category...)
        into a short labelled block so they're visible to the LLM extractor."""
        lines: list[str] = []
        for attr in clearance:
            if not isinstance(attr, dict):
                continue
            name = (attr.get("name") or "").strip()
            val = attr.get("value")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val if v)
            val = (str(val) if val is not None else "").strip()
            if name and val:
                lines.append(f"{name}: {val}")
        return "\n".join(lines)
