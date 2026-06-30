"""Core scraper contract: BaseScraper, the RawJob record, and a shared HttpClient.

Every scraper subclasses BaseScraper, declares SCRAPER_KEY / SITE_HINTS / PRIORITY,
and yields RawJob records from scrape(). All external I/O goes through HttpClient so
retries, timeouts and a realistic User-Agent stay consistent across the codebase.
"""
from __future__ import annotations

import logging
import re
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
    """UTC timestamp as an ISO-8601 'Z' string (used by the recon log)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scraped_at_stamp() -> str:
    """UTC scrape timestamp formatted as DAY-MONTH-YEAR (e.g. 22-06-2026)."""
    return datetime.now(timezone.utc).strftime("%d-%m-%Y")


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


def parse_posted_date(raw: Optional[str]) -> Optional[str]:
    """Normalise a posted-date string to ISO 'YYYY-MM-DD', or None.

    Handles the common careers-site formats: ISO ('2026-05-12T00:00:00+0000'),
    US slashes ('Date Posted: 04/28/2026' → MM/DD/YYYY), and abbreviated or full
    month names ('Jun 10, 2026', 'June 10, 2026'). Any leading 'Date Posted:' /
    'Posted:' label is stripped first. Returns None on anything unrecognised.
    """
    import re as _re
    from datetime import datetime as _dt

    if not raw:
        return None
    s = _re.sub(r"^\s*(?:date\s+)?posted\s*:?\s*", "", str(raw).strip(), flags=_re.I).strip()
    if not s:
        return None
    # ISO 'YYYY-MM-DD' (optionally with time/zone) — take the date part.
    m = _re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # US 'MM/DD/YYYY'.
    m = _re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    # 'Mon DD, YYYY' or 'Month DD, YYYY'.
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return _dt.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Non-US location filter (deny-list, fail-open)
#
# A job is flagged non-US ONLY when one of its location strings positively names a
# foreign country/region AND none of them carry a US marker. A US signal anywhere
# short-circuits to "keep"; a string with no signal either way is kept (fail-open),
# so an unrecognised or sparse US format is never dropped. All matching is
# whole-token / word-boundary, so "india" never matches inside "Indianapolis" and
# California's ", CA" is distinct from Canada's "CA-" prefix / "CAN" ISO-3 code.
# --------------------------------------------------------------------------- #
_US_STATE_CODES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
).split()
_US_STATE_NAMES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia",
]
# Foreign country ISO-3 codes (whole uppercase token, e.g. "Bengaluru, …, IND").
_FOREIGN_ISO3 = (
    "IND AUS GBR IRL JPN CAN BRA MEX DEU ITA ESP NLD SGP TWN KOR ZAF THA ARE ISR "
    "JOR LUX PHL POL CHE CHN HKG NZL SWE NOR DNK FIN BEL AUT ROU CZE HUN PRT GRC "
    "TUR SAU EGY VNM IDN MYS COL CHL ARG PER"
).split()
# Foreign country / unambiguous foreign-city names (whole word, case-insensitive).
_FOREIGN_NAMES = [
    "india", "singapore", "japan", "canada", "mexico", "brazil", "germany",
    "ireland", "united kingdom", "england", "scotland", "wales", "australia",
    "israel", "poland", "philippines", "netherlands", "spain", "italy", "taiwan",
    "south korea", "thailand", "switzerland", "china", "hong kong", "new zealand",
    "romania", "luxembourg", "sweden", "norway", "denmark", "finland", "belgium",
    "austria", "portugal", "greece", "turkey", "france", "colombia", "chile",
    "argentina", "egypt", "vietnam", "indonesia", "malaysia", "saudi arabia",
    "united arab emirates", "south africa", "jordan",
    "london", "dublin", "toronto", "vancouver", "montreal", "sydney", "melbourne",
    "tokyo", "bengaluru", "bangalore", "hyderabad", "mumbai", "chennai", "gurgaon",
    "pune", "singapore city", "frankfurt", "munich", "milan", "madrid", "barcelona",
    "amsterdam", "warsaw", "krakow", "wroclaw", "tel aviv", "shanghai", "beijing",
    "seoul", "taipei", "bangkok", "manila", "sao paulo", "mexico city",
]
# Leading ISO-2 country prefix in compact Workday strings ("SG-01-…", "CA-ON-…").
_FOREIGN_ISO2_PREFIX = (
    "SG PL AU GB PH MX CA IN BR DE IE JP IL CN HK NZ SE NO DK FI BE AT RO CZ HU "
    "MY KR TW TH AE CH ES IT NL PT GR TR SA EG VN ID CO CL AR PE ZA"
).split()

_US_STATE_CODE_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(_US_STATE_CODES) + r")(?![A-Za-z])"
)
_US_PHRASE_RE = re.compile(r"(?<![a-z])u\.?s\.?(?![a-z])", re.IGNORECASE)
_US_PREFIX_RE = re.compile(r"^\s*US-")
_FOREIGN_ISO3_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(_FOREIGN_ISO3) + r")(?![A-Za-z])"
)
_FOREIGN_NAME_RE = re.compile(
    r"(?<![a-z])(?:" + "|".join(re.escape(n) for n in _FOREIGN_NAMES) + r")(?![a-z])",
    re.IGNORECASE,
)
_FOREIGN_PREFIX_RE = re.compile(
    r"^\s*(?:" + "|".join(_FOREIGN_ISO2_PREFIX) + r")-"
)


def _has_us_signal(s: str) -> bool:
    low = s.lower()
    if "united states" in low or "usa" in low or _US_PHRASE_RE.search(s):
        return True
    if _US_PREFIX_RE.match(s):
        return True
    if any(name in low for name in _US_STATE_NAMES):
        return True
    return bool(_US_STATE_CODE_RE.search(s))


def _has_foreign_signal(s: str, extra: tuple[str, ...] = ()) -> bool:
    if _FOREIGN_PREFIX_RE.match(s):
        return True
    if _FOREIGN_ISO3_RE.search(s):
        return True
    if _FOREIGN_NAME_RE.search(s):
        return True
    low = s.lower()
    return any(tok and tok.lower() in low for tok in extra)


def non_us_location(
    location: Optional[str],
    locations_all: Optional[list[str]] = None,
    deny_extra: Optional[list[str]] = None,
) -> bool:
    """True if the job is recognizably non-US (safe to drop), else False.

    Fail-open: returns False (keep) when a US signal is present anywhere, and also
    when there is no recognizable signal either way. Only a positive foreign signal
    with no US signal anywhere yields True. ``deny_extra`` adds caller-supplied
    foreign substrings (e.g. UK county names) matched case-insensitively.
    """
    candidates = [location] if location else []
    candidates += [s for s in (locations_all or []) if s]
    candidates = [s.strip() for s in candidates if s and s.strip()]
    if not candidates:
        return False  # unknown location — keep
    if any(_has_us_signal(s) for s in candidates):
        return False  # US somewhere — keep
    extra = tuple(deny_extra or ())
    return any(_has_foreign_signal(s, extra) for s in candidates)


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
