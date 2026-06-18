"""Isolation test for the Leidos scraper: registration, recon routing, live scrape."""
from __future__ import annotations

URL = "https://careers.leidos.com/search/clearance/none-public-trust/jobs/in/country/united-states?q=software"

# 1) Registration
from scrapers.registry import registry
registry.discover()
print("[1] 'leidos' registered:", "leidos" in registry.all_keys())
cls = registry.get_class("leidos")
print("    class:", cls.__name__ if cls else None, "| SITE_HINTS:", cls.SITE_HINTS)

# 2) Recon routing (no Ollama needed — stage 1b matches SITE_HINTS on the URL)
from agent.recon_agent import ReconAgent
res = ReconAgent().investigate(URL, company_name="Leidos")
print("[2] recon status:", res.status, "| scraper_key:", res.scraper_key, "| conf:", res.confidence)

# 3) Live scrape — first listing page only, stop after a few detail fetches.
from scrapers.leidos_scraper import LeidosScraper
scraper = LeidosScraper(config={"max_pages": 1, "headless": False})
print("[3] scraping (headful window will open)…")
n = 0
for job in scraper.scrape(URL, company_name="Leidos"):
    n += 1
    print(f"\n--- job {n} ---")
    print("  url     :", job.source_url)
    print("  title   :", job.title)
    print("  location:", job.location)
    print("  job_id  :", job.job_id)
    print("  posted  :", job.posted_date)
    print("  raw_text:", len(job.raw_text), "chars |", repr((job.raw_text or "")[:120]))
    if n >= 4:
        break
print(f"\n[done] yielded {n} job(s)")
