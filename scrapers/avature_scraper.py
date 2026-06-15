"""Avature scraper: static, paginated HTML.

Listing pages paginate via a ``jobOffset`` query param incrementing by
``jobRecordsPerPage`` (default 20). Each results page contains anchors to
``...JobDetail?jobId={id}`` pages, which we fetch individually for full text.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

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

    _JOBID_RE = re.compile(r"jobId=(\d+)", re.I)

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        step = int(cfg.get("job_records_per_page", 20))
        max_pages = int(cfg.get("max_pages", 100))

        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
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
                    job = self._detail(durl, jid, company_name)
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
                if "JobDetail" in href and "jobId=" in href:
                    m = self._JOBID_RE.search(href)
                    if m:
                        title = a.get_text(strip=True) or None
                        out.append((m.group(1), urljoin(base, href), title))
        except Exception as exc:
            self.log.warning("Avature link parse failed: %s", exc)
        return out

    def _detail(self, durl: str, jid: str, company_name: Optional[str]) -> Optional[RawJob]:
        html = self.http.get_text(durl)
        if not html:
            return None
        text = html_to_text(html)
        if not text:
            return None
        return RawJob(
            source_url=durl,
            raw_text=text,
            scraper_key=self.SCRAPER_KEY,
            job_id=str(jid),
            title=extract_title(html),
            company=company_name,
            platform="avature",
        )
