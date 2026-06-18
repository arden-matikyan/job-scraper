"""Explore a Leidos detail page: JSON-LD? in-page fetch works? description container?"""
from __future__ import annotations
import json, os
DETAIL = "https://careers.leidos.com/jobs/17676078-software-developer"
LISTING = "https://careers.leidos.com/search/clearance/none-public-trust/jobs/in/country/united-states?q=software"
PROFILE = os.path.expanduser("~/.job-scraper/browser-profile")
from playwright.sync_api import sync_playwright

def apply_stealth(page):
    try:
        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page); return
    except Exception: pass

with sync_playwright() as pw:
    ctx = pw.chromium.launch_persistent_context(PROFILE, headless=False,
        args=["--disable-blink-features=AutomationControlled"])
    page = ctx.new_page(); apply_stealth(page)
    # Warm up on listing first to (re)solve challenge, then in-page fetch the detail.
    page.goto(LISTING, wait_until="domcontentloaded", timeout=60000)
    for _ in range(6):
        page.wait_for_timeout(2000)
        if "just a moment" not in page.title().lower(): break
    print("listing title:", repr(page.title()))

    # Try a same-origin in-page fetch of the detail page (light, carries cf cookie).
    fetched = page.evaluate("""async (u) => {
        try { const r = await fetch(u, {headers:{'X-Requested-With':'XMLHttpRequest'}});
              const t = await r.text(); return {status:r.status, len:t.length, html:t}; }
        catch(e){ return {error:String(e)}; }
    }""", DETAIL)
    print("in-page fetch status:", fetched.get("status"), "len:", fetched.get("len"))
    html = fetched.get("html") or ""

    if html:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # JSON-LD?
        lds = soup.find_all("script", type="application/ld+json")
        print("JSON-LD blocks:", len(lds))
        for s in lds:
            try:
                data = json.loads(s.string or "{}")
                t = data.get("@type") if isinstance(data, dict) else None
                print("  @type:", t)
                if t == "JobPosting":
                    print("  keys:", list(data.keys()))
                    for k in ("title","datePosted","employmentType","validThrough","jobLocation","hiringOrganization"):
                        v = data.get(k)
                        print(f"    {k}:", json.dumps(v, default=str)[:160])
                    desc = data.get("description","")
                    print("  description length:", len(desc), "(HTML)" if "<" in desc else "(plain)")
            except Exception as e:
                print("  ld parse err:", e)
        # Title + main description container guesses
        h1 = soup.find("h1")
        print("h1:", repr(h1.get_text(strip=True) if h1 else None))
        for sel in ["div.job-description","#job-description","div.jobdesc","section.job","article",
                    "div.description","div.ats-description","main",".job-details",".content"]:
            el = soup.select_one(sel)
            if el:
                txt = el.get_text(" ", strip=True)
                print(f"selector {sel!r}: {len(txt)} chars  -> {txt[:120]!r}")
        # Save for offline inspection
        open("_leidos_detail.html","w",encoding="utf-8").write(html)
        print("saved _leidos_detail.html")
    ctx.close()
