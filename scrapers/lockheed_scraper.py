"""Lockheed Martin careers scraper (lockheedmartinjobs.com — Radancy/TalentBrew).

The site returns plain HTML (200 OK, no bot challenge) with 15 listings per page.
Pagination: ``?p=N`` on the base search URL (1-based; page 1 omits the param).
Detail pages embed a ``JobPosting`` JSON-LD block that carries the full description,
all locations, req ID, datePosted, and baseSalary.

Listing : {url}?p={N}   (<li> inside #search-results, each with an <a> to the detail)
Detail  : https://www.lockheedmartinjobs.com/job/{city}/{slug}/694/{numeric_id}
          JSON-LD <script type="application/ld+json"> → JobPosting @type
"""
from __future__ import annotations

import json
import re
from datetime import date as _date
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

from scrapers.base import BaseScraper, RawJob, html_to_text

_JOBID_RE = re.compile(r"/(\d+)$")
_DATE_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def _iso_date(raw: Optional[str]) -> Optional[str]:
    """Normalise 'YYYY-M-D' → 'YYYY-MM-DD'. Returns None on parse failure."""
    if not raw:
        return None
    m = _DATE_RE.match(raw.strip())
    if not m:
        return None
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def _jobld(html: str) -> Optional[dict]:
    """Extract the JobPosting JSON-LD block from a detail page."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                d = json.loads(s.string or "{}")
            except Exception:
                continue
            if isinstance(d, dict) and d.get("@type") == "JobPosting":
                return d
    except Exception:
        pass
    return None


def _location_str(place: dict) -> Optional[str]:
    addr = place.get("address") or {}
    city = addr.get("addressLocality")
    region = addr.get("addressRegion")
    country = addr.get("addressCountry") or ""
    parts = [p for p in [city, region] if p]
    if not parts and country:
        return country
    return ", ".join(parts) or None


class LockheedScraper(BaseScraper):
    SCRAPER_KEY = "lockheed"
    SITE_HINTS = ["lockheedmartinjobs.com"]
    PRIORITY = 10  # JSON-LD backed, no browser needed

    def scrape(
        self, url: str, company_name: Optional[str] = None, config: Optional[dict] = None
    ) -> Iterator[RawJob]:
        cfg = {**self.config, **(config or {})}
        max_pages = int(cfg.get("max_pages", 200))
        company = company_name or "Lockheed Martin"

        _oldest: Optional[_date] = None
        _oldest_raw = cfg.get("oldest_date")
        if _oldest_raw:
            try:
                _oldest = _date.fromisoformat(str(_oldest_raw))
            except ValueError:
                self.log.warning("Lockheed: invalid oldest_date %r — ignoring", _oldest_raw)
        break_after = int(cfg.get("break_after_old_pages", 2))
        consecutive_old_pages = 0
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        # Strip any existing ?p= param so we control pagination cleanly
        clean_url = f"{p.scheme}://{p.netloc}{p.path}"
        if p.query:
            from urllib.parse import parse_qs, urlencode
            qs = {k: v for k, v in parse_qs(p.query).items() if k != "p"}
            if qs:
                clean_url += "?" + urlencode(qs, doseq=True)

        seen_urls: set[str] = set()

        for pno in range(1, max_pages + 1):
            page_url = clean_url if pno == 1 else f"{clean_url}{'&' if '?' in clean_url else '?'}p={pno}"
            html = self.http.get_text(page_url)
            if not html:
                break
            cards = self._parse_cards(html, base)
            new_on_page = [c for c in cards if c["url"] not in seen_urls]
            if not new_on_page:
                break  # no new jobs → past the last page
            fresh_on_page = 0
            for card in new_on_page:
                seen_urls.add(card["url"])
                if _oldest and card.get("posted_date"):
                    try:
                        if _date.fromisoformat(card["posted_date"]) < _oldest:
                            continue  # too old — skip detail fetch entirely
                    except ValueError:
                        pass
                fresh_on_page += 1
                try:
                    if card["url"] in self.seen_urls:
                        yield RawJob(
                            source_url=card["url"],
                            raw_text="",
                            scraper_key=self.SCRAPER_KEY,
                            job_id=card["req_id"],
                            title=card["title"],
                            company=company,
                            location=card["location"],
                            posted_date=card["posted_date"],
                            already_seen=True,
                            platform="lockheed",
                        )
                        continue
                    job = self._detail(card, company)
                    if job:
                        yield job
                except Exception as exc:
                    self.log.warning("Lockheed detail failed for %s: %s", card["url"], exc)

            if _oldest:
                if fresh_on_page == 0:
                    consecutive_old_pages += 1
                    if consecutive_old_pages >= break_after:
                        self.log.info(
                            "Lockheed: %d consecutive pages all older than %s — stopping",
                            break_after, _oldest,
                        )
                        break
                else:
                    consecutive_old_pages = 0

    def _parse_cards(self, html: str, base: str) -> list[dict]:
        out: list[dict] = []
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            results = soup.select_one("#search-results")
            if not results:
                return out
            for a in results.find_all("a", href=re.compile(r"/job/")):
                href = a.get("href", "")
                full_url = urljoin(base, href)
                title_el = a.find("span", class_="job-title")
                loc_el = a.find("span", class_="job-location")
                date_el = a.find("span", class_="job-date-posted")
                id_el = a.find("span", class_="job-id")
                req_id = None
                if id_el:
                    raw_id = id_el.get_text(strip=True)
                    req_id = re.sub(r"^Job\s+ID\s*:\s*", "", raw_id, flags=re.I).strip() or None
                date_raw = date_el.get_text(strip=True) if date_el else None
                if date_raw:
                    # "Date Posted: 04/28/2026" → ISO
                    dm = re.search(r"(\d{2})/(\d{2})/(\d{4})", date_raw)
                    date_raw = f"{dm.group(3)}-{dm.group(1)}-{dm.group(2)}" if dm else None
                out.append({
                    "url": full_url,
                    "title": title_el.get_text(strip=True) if title_el else None,
                    "location": loc_el.get_text(strip=True) if loc_el else None,
                    "posted_date": date_raw,
                    "req_id": req_id,
                })
        except Exception as exc:
            self.log.warning("Lockheed listing parse failed: %s", exc)
        return out

    def _detail(self, card: dict, company: str) -> Optional[RawJob]:
        html = self.http.get_text(card["url"])
        if not html:
            return None
        ld = _jobld(html)
        if not ld:
            # Fall back to main-content text if JSON-LD is absent
            raw_text = html_to_text(html)
            if not raw_text:
                return None
            return RawJob(
                source_url=card["url"],
                raw_text=raw_text,
                scraper_key=self.SCRAPER_KEY,
                job_id=card["req_id"],
                title=card["title"],
                company=company,
                location=card["location"],
                posted_date=card["posted_date"],
                platform="lockheed",
            )

        desc_html = ld.get("description") or ""
        raw_text = html_to_text(desc_html) if desc_html else html_to_text(html)
        if not raw_text:
            return None

        # Locations: build primary + locations_all from the array of Place objects
        places = ld.get("jobLocation") or []
        if isinstance(places, dict):
            places = [places]
        loc_strings = [_location_str(pl) for pl in places if isinstance(pl, dict)]
        loc_strings = [s for s in loc_strings if s]
        primary_loc = card["location"] or (loc_strings[0] if loc_strings else None)
        locations_all = loc_strings if len(loc_strings) > 1 else None

        # identifier is the BR req number (e.g. "725731BR")
        ident = ld.get("identifier")
        if isinstance(ident, dict):
            ident = ident.get("value") or None
        req_id = (str(ident).strip() if ident else None) or card["req_id"]

        return RawJob(
            source_url=card["url"],
            raw_text=raw_text,
            scraper_key=self.SCRAPER_KEY,
            job_id=req_id,
            title=ld.get("title") or card["title"],
            company=company,
            location=primary_loc,
            locations_all=locations_all,
            posted_date=_iso_date(ld.get("datePosted")) or card["posted_date"],
            platform="lockheed",
        )
