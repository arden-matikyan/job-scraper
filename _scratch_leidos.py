"""Throwaway exploration of careers.leidos.com (Cloudflare + Phenom People)."""
from __future__ import annotations

import json
import os
import sys

URL = "https://careers.leidos.com/search/clearance/none-public-trust/jobs/in/country/united-states?q=software"
PROFILE = os.path.expanduser("~/.job-scraper/browser-profile")
HEADLESS = os.environ.get("HEADLESS", "1") != "0"

os.makedirs(PROFILE, exist_ok=True)

from playwright.sync_api import sync_playwright


def apply_stealth(page):
    try:
        from playwright_stealth import stealth_sync
        stealth_sync(page)
        return "stealth_sync"
    except Exception:
        pass
    try:
        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page)
        return "Stealth().apply"
    except Exception as e:
        return f"no-stealth ({e})"


with sync_playwright() as pw:
    ctx = pw.chromium.launch_persistent_context(
        PROFILE, headless=HEADLESS,
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.new_page()
    print("stealth:", apply_stealth(page))
    print("navigating (headless=%s) ..." % HEADLESS)
    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        print("goto error:", e)

    # Give Cloudflare's managed challenge time to auto-resolve and the SPA to render.
    for i in range(6):
        page.wait_for_timeout(2500)
        t = page.title()
        if "just a moment" not in t.lower() and "attention required" not in t.lower():
            break
    print("title:", repr(page.title()))
    print("url  :", page.url)

    # Phenom app probes
    probe = page.evaluate(
        """() => {
            const out = {};
            out.hasPhApp = typeof window.phApp !== 'undefined';
            try { out.ddoKeys = Object.keys((window.phApp && window.phApp.ddo) || {}); } catch(e){ out.ddoKeys = String(e); }
            try {
                const rs = window.phApp && window.phApp.ddo && window.phApp.ddo.refineSearch;
                out.hasRefineSearch = !!rs;
                if (rs) {
                    out.totalHits = rs.totalHits;
                    out.dataKeys = Object.keys(rs.data || {});
                    const jobs = (rs.data && rs.data.jobs) || [];
                    out.jobCount = jobs.length;
                    out.sampleJob = jobs[0] || null;
                }
            } catch(e){ out.refineErr = String(e); }
            // anchors to job detail pages
            const anchors = [...document.querySelectorAll('a[href*="/job/"]')].slice(0,5)
                .map(a => ({href: a.href, text: (a.textContent||'').trim().slice(0,80)}));
            out.jobAnchors = anchors;
            out.jobAnchorCount = document.querySelectorAll('a[href*="/job/"]').length;
            return out;
        }"""
    )
    print("PROBE:")
    print(json.dumps(probe, indent=2, default=str)[:6000])

    # Try the Phenom widgets API from inside the page (carries cf cookie).
    api_result = page.evaluate(
        """async () => {
            const body = {
                lang: "en_us", deviceType: "desktop", country: "us",
                pageName: "search-results", ddoKey: "refineSearch",
                sortBy: "Relevance", subsearch: "", from: 0, jobs: true,
                counts: true, all_fields: ["category","country","state","city"],
                size: 10, clearAll: false, jdsource: "facets",
                isSliderEnable: false, pageId: "page12", siteType: "external",
                keywords: "software", global: true, selected_fields: {}, locationData: {}
            };
            try {
                const r = await fetch("/widgets", {
                    method: "POST",
                    headers: {"Content-Type": "application/json", "X-Requested-With":"XMLHttpRequest"},
                    body: JSON.stringify(body),
                });
                const text = await r.text();
                return {status: r.status, len: text.length, head: text.slice(0, 1500)};
            } catch(e) { return {error: String(e)}; }
        }"""
    )
    print("API /widgets:")
    print(json.dumps(api_result, indent=2, default=str)[:3000])

    try:
        page.screenshot(path="C:/Users/arden/agentic-job-scraper/_leidos.png", full_page=False)
        with open("C:/Users/arden/agentic-job-scraper/_leidos.html", "w", encoding="utf-8") as f:
            f.write(page.content())
        print("saved _leidos.png + _leidos.html")
    except Exception as e:
        print("save err:", e)

    ctx.close()
