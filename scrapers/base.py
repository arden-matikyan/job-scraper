"""Core scraper contract: BaseScraper, the RawJob record, and a shared HttpClient.

Every scraper subclasses BaseScraper, declares SCRAPER_KEY / SITE_HINTS / PRIORITY,
and yields RawJob records from scrape(). All external I/O goes through HttpClient so
retries, timeouts and a realistic User-Agent stay consistent across the codebase.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def utcnow_iso() -> str:
    """UTC timestamp as an ISO-8601 'Z' string (used for scraped_at)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RawJob:
    """What every scraper yields. ``raw_text`` is the spine of LLM extraction.

    Scrapers fill source_url + raw_text at minimum, plus whatever structured fields
    they already know authoritatively (title, company, location...). Those
    authoritative fields win over the LLM during the merge step in the runner.
    """

    source_url: str
    raw_text: str
    scraper_key: str
    job_id: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    locations_all: Optional[list[str]] = None
    posted_date: Optional[str] = None
    platform: Optional[str] = None
    # set by a detail-fetch scraper when source_url was already in the DB and the
    # detail download was skipped; the runner counts it as found but does not ingest.
    already_seen: bool = False
    # free-form extras a scraper may stash; not persisted directly
    extra: dict[str, Any] = field(default_factory=dict)

    def authoritative_fields(self) -> dict[str, Any]:
        """Scraper-known fields that override LLM output during the merge step."""
        keys = (
            "job_id", "title", "company", "location", "locations_all",
            "posted_date",
        )
        return {
            k: getattr(self, k)
            for k in keys
            if getattr(self, k) not in (None, "", [])
        }


def _safe_json(resp: Optional[httpx.Response]) -> Optional[Any]:
    if resp is None:
        return None
    try:
        return resp.json()
    except Exception:
        return None


class HttpClient:
    """Thin synchronous httpx wrapper: shared UA, timeouts, retries with backoff.

    The helper methods never raise on network errors — they return None / "" so a
    single bad request can't crash a run. Status codes are *not* treated as errors;
    callers inspect ``resp.status_code`` when they care.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        user_agent: str = DEFAULT_USER_AGENT,
        headers: Optional[dict] = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        base_headers = {"User-Agent": user_agent, "Accept": "*/*"}
        if headers:
            base_headers.update(headers)
        self._client = httpx.Client(
            timeout=timeout, headers=base_headers, follow_redirects=True
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def request(self, method: str, url: str, **kwargs) -> Optional[httpx.Response]:
        """Return a Response (any status) or None if every attempt errored."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._client.request(method, url, **kwargs)
            except Exception as exc:  # timeouts, connection/protocol errors
                last_exc = exc
                logger.warning(
                    "HTTP %s %s failed (attempt %d/%d): %s",
                    method, url, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        logger.error(
            "HTTP %s %s gave up after %d attempts: %s",
            method, url, self.max_retries, last_exc,
        )
        return None

    def get(self, url: str, **kwargs) -> Optional[httpx.Response]:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> Optional[httpx.Response]:
        return self.request("POST", url, **kwargs)

    def get_text(self, url: str, **kwargs) -> str:
        resp = self.get(url, **kwargs)
        if resp is None:
            return ""
        try:
            return resp.text
        except Exception:
            return ""

    def get_json(self, url: str, **kwargs) -> Optional[Any]:
        return _safe_json(self.get(url, **kwargs))

    def post_json(self, url: str, **kwargs) -> Optional[Any]:
        return _safe_json(self.post(url, **kwargs))


_BLOCK_TAGS = (
    "p", "div", "li", "ul", "ol", "tr", "table", "section", "article",
    "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
)


def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text, preserving block structure.

    Block-level tags and <br> become newlines; inline tags (b, a, span...) do not,
    so a sentence with inline emphasis stays on one line. Any double-escaped
    entities (e.g. Workday's &#xa;) are decoded. Defensive: returns input on error.
    """
    if not html:
        return ""
    try:
        import html as _html

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for tag in soup.find_all(_BLOCK_TAGS):
            tag.append("\n")
        text = _html.unescape(soup.get_text())
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)
    except Exception:
        return html


def extract_title(html: str) -> Optional[str]:
    """Best-effort job title from a detail page: h1, then <title>, then og:title."""
    if not html:
        return None
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for sel in ("h1", "title"):
            el = soup.find(sel)
            if el and el.get_text(strip=True):
                return el.get_text(strip=True)
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
    except Exception:
        pass
    return None


def add_query_param(url: str, key: str, value: Any) -> str:
    """Return ``url`` with query param ``key`` set to ``value`` (added or replaced)."""
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q[key] = str(value)
    return urlunparse(p._replace(query=urlencode(q)))


def find_job_links(html: str, base_url: str, patterns: list[str]) -> list[str]:
    """Same-host absolute links whose href contains any of ``patterns`` (deduped)."""
    from urllib.parse import urljoin, urlparse

    out: list[str] = []
    seen: set[str] = set()
    base_host = urlparse(base_url).netloc
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(pat in href for pat in patterns):
                full = urljoin(base_url, href)
                if urlparse(full).netloc == base_host and full not in seen:
                    seen.add(full)
                    out.append(full)
    except Exception:
        pass
    return out


class BaseScraper(ABC):
    """Contract every scraper must satisfy.

    Class attributes:
      SCRAPER_KEY  unique string id; also the key used by the registry and the KB.
      SITE_HINTS   substrings matched against a URL / page source for auto-routing.
                   Empty list => never auto-matched (e.g. the static_html fallback).
      PRIORITY     lower = preferred when several scrapers match (API < fallback).
      REQUIRES_INTERACTION  hint that this scraper may need user input.
    """

    SCRAPER_KEY: str = ""
    SITE_HINTS: list[str] = []
    PRIORITY: int = 100
    REQUIRES_INTERACTION: bool = False

    def __init__(self, http: Optional[HttpClient] = None, config: Optional[dict] = None):
        self.http = http or HttpClient()
        self.config = config or {}
        self.log = logging.getLogger(
            f"scraper.{self.SCRAPER_KEY or self.__class__.__name__}"
        )
        # source_urls already saved in the DB; detail-fetch scrapers skip these to
        # avoid re-downloading job pages on reruns. The runner populates this.
        self.seen_urls: set[str] = set()

    @classmethod
    def matches(cls, url: str, page_source: Optional[str] = None) -> bool:
        """Default match: any SITE_HINT appears in the url or the page source."""
        if not cls.SITE_HINTS:
            return False
        haystack = (url or "").lower()
        if page_source:
            haystack += "\n" + page_source.lower()
        return any(hint.lower() in haystack for hint in cls.SITE_HINTS)

    @abstractmethod
    def scrape(
        self,
        url: str,
        company_name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> Iterator[RawJob]:
        """Yield a RawJob for every listing reachable from ``url``.

        Implementations MUST wrap their own network/parse calls in try/except and
        yield whatever they can; a single failure must never abort the whole run.
        """
        raise NotImplementedError
