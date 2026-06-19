"""IBM careers scraper (careers.ibm.com — Radancy/TalentBrew, AWS WAF-protected).

careers.ibm.com is a server-rendered Radancy "OpenJobs" site sitting behind an
AWS WAF challenge that a plain HTTP client (and headless Chromium) cannot clear —
it just gets a 202 with an empty / challenge body. A real, headful browser passes
the challenge automatically, so this scraper drives a persistent headful Chromium
via Playwright (+ stealth), exactly like ``leidos``.

Once the WAF challenge is solved on the first navigation, the token cookie is live
for the session, so the remaining pages are pulled with lightweight same-origin
in-page ``fetch()`` calls (fast and fingerprint-consistent), falling back to a full
navigation if a page comes back as a challenge.

Listing : {url}&jobOffset=N   (N steps by jobRecordsPerPage; one <article
          class="article--card"> per job, with title, type and location inline)
Detail  : https://careers.ibm.com/en_US/careers/JobDetail?jobId={id}
          server-rendered; <main> holds the full description, and a set of
          article__content__view__field label/value pairs carry City/State,
          Contract type, salary, education, etc. <title> is "{Title} - {id} - IBM".

Headful is required: set ``headless: true`` in scraper_configs only if the host
can clear the WAF without a visible window (it usually cannot).
"""
from __future__ import annotations

import re
from typing import Iterator, Optional
from urllib.parse import parse_qs, urljoin, urlparse

from scrapers.base import (
    BaseScraper,
    RawJob,
    add_query_param,
    html_to_text,
)

_JOBID_RE = re.compile(r"[?&]jobId=(\d+)", re.I)
_CHALLENGE_MARKERS = ("awswaf", "token.awswaf", "captcha", "verify you are human",
                      "challenge-container")
_EMP_TYPES = (
    ("full time", "full_time"), ("full-time", "full_time"),
    ("part time", "part_time"), ("part-time", "part_time"),
    ("intern", "internship"), ("co-op", "internship"), ("co op", "internship"),
    ("contract", "contract"), ("temporary", "contract"),
)


def _apply_stealth(page) -> None:
    """Apply playwright-stealth across its differing API versions; ignore failures."""
    try:
        from playwright_stealth import stealth_sync  # older API

        stealth_sync(page)
        return
    except Exception:
        pass
    try:
        from playwright_stealth import Stealth  # newer API

        Stealth().apply_stealth_sync(page)
    except Exception:
        pass


def _norm_emp_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    low = value.strip().lower()
    for needle, norm in _EMP_TYPES:
        if needle in low:
            return norm
    return None


class IbmScraper(BaseScraper):
    SCRAPER_KEY = "ibm"
    SITE_HINTS = ["careers.ibm.com"]
    PRIORITY = 20
    REQUIRES_INTERACTION = True  # launches a (headful) browser to clear AWS WAF

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            self.log.error("Playwright not available: %s", exc)
            return

        import os

        profile_dir = os.path.expanduser(cfg.get("profile_dir", "~/.job-scraper/browser-profile"))
        try:
            os.makedirs(profile_dir, exist_ok=True)
        except Exception:
            pass
        headless = bool(cfg.get("headless", False))  # AWS WAF normally needs headful
        nav_timeout = int(cfg.get("nav_timeout_ms", 60000))
        max_pages = int(cfg.get("max_pages", 40))
        # records per page: honour what's already in the URL, else default to 48.
        per_page = parse_qs(urlparse(url).query).get("jobRecordsPerPage", [None])[0]
        per_page = int(cfg.get("records_per_page", per_page or 48))
        company = company_name or "IBM"
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"

        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(
                    profile_dir, headless=headless,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                try:
                    page = ctx.new_page()
                    _apply_stealth(page)
                    # Solve the WAF challenge once on the first listing page.
                    if not self._open(page, url, nav_timeout):
                        self.log.error("IBM: could not clear AWS WAF challenge for %s", url)
                        return
                    yield from self._walk_listing(
                        page, url, base, company, per_page, max_pages, nav_timeout
                    )
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as exc:
            self.log.error("IBM Playwright session failed: %s", exc)

    # ----------------------------------------------------------------- listing
    def _walk_listing(
        self, page, url: str, base: str, company: str, per_page: int,
        max_pages: int, timeout: int,
    ) -> Iterator[RawJob]:
        seen_ids: set[str] = set()
        for pno in range(max_pages):
            page_url = add_query_param(url, "jobOffset", pno * per_page)
            html = self._fetch(page, page_url, timeout)
            cards = [c for c in self._parse_cards(html, base) if c["job_id"] not in seen_ids]
            if not cards:
                break  # no new jobs on this page => past the last results page
            for card in cards:
                seen_ids.add(card["job_id"])
                durl = card["url"]
                try:
                    if durl in self.seen_urls:
                        yield RawJob(
                            source_url=durl, raw_text="", scraper_key=self.SCRAPER_KEY,
                            job_id=card["job_id"], title=card["title"], company=company,
                            location=card["location"],
                            already_seen=True, platform="ibm",
                        )
                        continue
                    job = self._detail(page, card, company, timeout)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("IBM detail failed for %s: %s", durl, exc)

    def _parse_cards(self, html: str, base: str) -> list[dict]:
        """Return one dict per job card: url, job_id, title, emp_type, location."""
        out: list[dict] = []
        if not html:
            return out
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for art in soup.select("article.article--card"):
                link = art.select_one('a[href*="JobDetail?jobId="]')
                if not link:
                    continue
                durl = urljoin(base, link.get("href", ""))
                m = _JOBID_RE.search(durl)
                if not m:
                    continue
                title_el = art.select_one("h3.article__header__text__title a") or link
                out.append({
                    "url": durl,
                    "job_id": m.group(1),
                    "title": title_el.get_text(strip=True) or None,
                    "emp_type": self._card_text(art, "span.card-item-type"),
                    "location": self._card_text(art, "span.card-item-location"),
                })
        except Exception as exc:
            self.log.warning("IBM listing parse failed: %s", exc)
        return out

    @staticmethod
    def _card_text(art, selector: str) -> Optional[str]:
        el = art.select_one(selector)
        return el.get_text(strip=True) if el and el.get_text(strip=True) else None

    # ------------------------------------------------------------------ detail
    def _detail(self, page, card: dict, company: str, timeout: int) -> Optional[RawJob]:
        durl = card["url"]
        html = self._fetch(page, durl, timeout)
        if not html:
            return None
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None

        main = soup.find("main")
        raw_text = html_to_text(str(main)) if main else html_to_text(html)
        if not raw_text:
            return None

        fields = self._fields(soup)
        title = card["title"] or self._title(soup)
        location = card["location"] or self._location_from_fields(fields)
        return RawJob(
            source_url=durl,
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=card["job_id"] or fields.get("job id"),
            title=title,
            company=company,
            location=location,
            platform="ibm",
        )

    @staticmethod
    def _fields(soup) -> dict[str, str]:
        """Map the detail page's label/value field pairs (lowercased labels)."""
        out: dict[str, str] = {}
        try:
            for field in soup.select("div.article__content__view__field"):
                label = field.select_one(".article__content__view__field__label")
                value = field.select_one(".article__content__view__field__value")
                if label and value:
                    key = label.get_text(strip=True).rstrip(":").lower()
                    val = value.get_text(" ", strip=True)
                    if key and val:
                        out[key] = val
        except Exception:
            pass
        return out

    @staticmethod
    def _location_from_fields(fields: dict[str, str]) -> Optional[str]:
        parts = [
            fields.get("city / township / village") or fields.get("city"),
            fields.get("state / province") or fields.get("state"),
            fields.get("country"),
        ]
        joined = ", ".join(p for p in parts if p)
        return joined or None

    @staticmethod
    def _salary_from_fields(fields: dict[str, str]) -> Optional[str]:
        for key, val in fields.items():
            if "salary" in key:
                return val
        return None

    @staticmethod
    def _title(soup) -> Optional[str]:
        """Detail <title> is '{Title} - {jobId} - IBM' — strip the trailing parts."""
        if not soup.title:
            return None
        raw = soup.title.get_text(strip=True)
        # drop a trailing ' - IBM' and a ' - {digits}' req id if present
        raw = re.sub(r"\s*-\s*IBM\s*$", "", raw)
        raw = re.sub(r"\s*-\s*\d+\s*$", "", raw)
        return raw or None

    # -------------------------------------------------------------- navigation
    def _fetch(self, page, url: str, timeout: int) -> str:
        """Lightweight same-origin in-page fetch; falls back to a full nav if the
        session hit a WAF challenge (e.g. the token cookie expired)."""
        try:
            res = page.evaluate(
                """async (u) => {
                    try {
                        const r = await fetch(u, {headers: {'X-Requested-With': 'XMLHttpRequest'}});
                        return {status: r.status, text: await r.text()};
                    } catch (e) { return {status: 0, text: ''}; }
                }""",
                url,
            )
        except Exception as exc:
            self.log.debug("IBM in-page fetch error for %s: %s", url, exc)
            res = {"status": 0, "text": ""}
        text = (res or {}).get("text") or ""
        if (res or {}).get("status") == 200 and not self._is_challenge(text):
            return text
        # Re-solve via a real navigation, then return the rendered HTML.
        if self._open(page, url, timeout):
            try:
                return page.content()
            except Exception:
                return ""
        return ""

    def _open(self, page, url: str, timeout: int) -> bool:
        """Navigate and wait for the WAF challenge to clear (the title stops being
        the interstitial / blank). Returns True once the real page is showing."""
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except Exception as exc:
            self.log.warning("IBM goto %s failed: %s", url, exc)
        for _ in range(10):
            page.wait_for_timeout(2000)
            try:
                title = (page.title() or "").lower()
            except Exception:
                title = ""
            if title and not any(
                m in title for m in ("just a moment", "verify", "attention", "human")
            ):
                return True
        return False

    @staticmethod
    def _is_challenge(html: str) -> bool:
        low = (html or "")[:4000].lower()
        return any(m in low for m in _CHALLENGE_MARKERS)
