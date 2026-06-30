"""Avature scraper: static, paginated HTML.

Listing pages paginate via a ``jobOffset`` query param incrementing by
``jobRecordsPerPage`` (default 20). Each results page contains anchors to
``...JobDetail?jobId={id}`` pages, which we fetch individually for full text.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse, parse_qs

from scrapers.base import (
    BaseScraper,
    RawJob,
    add_query_param,
    extract_title,
    html_to_text,
)


class AvatureScraper(BaseScraper):
    SCRAPER_KEY = "avature"
    SITE_HINTS = ["avature.net", "JobDetail?jobId"]
    PRIORITY = 20

    _JOBID_QUERY_RE = re.compile(r"jobId=(\d+)", re.I)
    # Deloitte-style: /careers/JobDetail/Some-Title/356614
    _JOBID_PATH_RE = re.compile(r"/JobDetail/[^/?#]+/(\d+)", re.I)

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        max_pages = int(cfg.get("max_pages", 100))

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        # The server returns exactly `jobRecordsPerPage` listings per results page,
        # so the jobOffset step MUST equal the URL's value — a larger step silently
        # skips the jobs in between (e.g. Deloitte uses 10; a config step of 20 hid
        # half the catalog). The config value is only a fallback for URLs that omit it.
        qs_rpp = parse_qs(p.query).get("jobRecordsPerPage", [None])[0]
        step = int(qs_rpp or cfg.get("job_records_per_page", 20))
        seen: set[str] = set()
        for page in range(max_pages):
            page_url = add_query_param(url, "jobOffset", page * step)
            html = self.http.get_text(page_url)
            if not html:
                break
            links = [
                (jid, durl, title)
                for jid, durl, title in self._detail_links(html, base)
                if jid not in seen
            ]
            if not links:
                break
            for jid, durl, title in links:
                seen.add(jid)
                try:
                    if durl in self.seen_urls:
                        yield RawJob(
                            source_url=durl, raw_text="", scraper_key=self.SCRAPER_KEY,
                            job_id=str(jid), title=title, company=company_name,
                            already_seen=True, platform="avature",
                        )
                        continue
                    job = self._detail(durl, jid, company_name, listing_title=title)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("Avature detail failed for %s: %s", durl, exc)

    def _detail_links(self, html: str, base: str) -> list[tuple[str, str, Optional[str]]]:
        out: list[tuple[str, str, Optional[str]]] = []
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "JobDetail" not in href:
                    continue
                m = self._JOBID_QUERY_RE.search(href) or self._JOBID_PATH_RE.search(href)
                if m:
                    title = a.get_text(strip=True) or None
                    out.append((m.group(1), urljoin(base, href), title))
        except Exception as exc:
            self.log.warning("Avature link parse failed: %s", exc)
        return out

    def _detail(
        self, durl: str, jid: str, company_name: Optional[str],
        listing_title: Optional[str] = None,
    ) -> Optional[RawJob]:
        html = self.http.get_text(durl)
        if not html:
            return None
        text = html_to_text(html)
        if not text:
            return None
        title = listing_title or extract_title(html)
        return RawJob(
            source_url=durl,
            raw_text=text,
            scraper_key=self.SCRAPER_KEY,
            job_id=str(jid),
            title=title,
            company=company_name,
            platform="avature",
        )
