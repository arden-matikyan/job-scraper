"""One-shot DB cleanup: remove jobs that pre-date the keyword/title-exclude filters.

Applies the exact same two-stage filter logic as pipeline/runner.py _ingest():
  1. title_exclude  — whole-word match (any term -> drop)
  2. keyword_filter — title must contain at least one term (none -> drop)

Dry-run by default. Pass --apply to commit the deletions.
Both jobs rows and their seen_hashes entries are removed so the dedup table
stays consistent (removed jobs will be re-evaluated on the next scrape run).
"""
import re
import sqlite3
import sys
import yaml

DRY_RUN = "--apply" not in sys.argv

con = sqlite3.connect("jobs.db")
con.row_factory = sqlite3.Row

cfg = yaml.safe_load(open("config/scraper_configs.yaml"))

# --- build filters (mirrors Runner.__init__) ---
title_exclude_terms = [str(t).strip() for t in (cfg.get("title_exclude") or []) if str(t).strip()]
title_exclude_re = (
    re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in title_exclude_terms) + r")\b",
        re.IGNORECASE,
    )
    if title_exclude_terms
    else None
)

keyword_filter = [str(k).lower() for k in (cfg.get("keyword_filter") or []) if str(k).strip()]


def _should_drop(title: str) -> tuple[bool, str]:
    t = title or ""
    if title_exclude_re and title_exclude_re.search(t):
        return True, "title_exclude"
    if keyword_filter:
        padded = f" {t.lower()} "
        if not any(kw in padded for kw in keyword_filter):
            return True, "keyword_filter"
    return False, ""


rows = con.execute("SELECT id, hash, company, title FROM jobs").fetchall()

to_delete: list[tuple[int, str, str, str, str]] = []  # (id, hash, company, title, reason)
for r in rows:
    drop, reason = _should_drop(r["title"] or "")
    if drop:
        to_delete.append((r["id"], r["hash"], r["company"] or "", r["title"] or "", reason))

# --- report ---
by_company: dict[str, dict[str, int]] = {}
for _, _, company, _, reason in to_delete:
    by_company.setdefault(company, {})
    by_company[company][reason] = by_company[company].get(reason, 0) + 1

print(f"\n{'DRY RUN — pass --apply to commit' if DRY_RUN else 'APPLYING DELETIONS'}")
print(f"\nJobs to remove: {len(to_delete)}")
for company in sorted(by_company):
    parts = ", ".join(f"{n} {r}" for r, n in sorted(by_company[company].items()))
    print(f"  {company}: {parts}")

if to_delete and not DRY_RUN:
    print("\nSample titles being removed (first 20):")
    for _, _, company, title, reason in to_delete[:20]:
        print(f"  [{reason}] {company} — {title}")

    ids = [r[0] for r in to_delete]
    hashes = [r[1] for r in to_delete]

    con.execute(f"DELETE FROM jobs WHERE id IN ({','.join('?' for _ in ids)})", ids)
    con.execute(
        f"DELETE FROM seen_hashes WHERE hash IN ({','.join('?' for _ in hashes)})", hashes
    )
    con.commit()
    print(f"\nDeleted {len(ids)} jobs and {len(ids)} seen_hashes entries.")
elif to_delete:
    print("\nSample titles that would be removed (first 20):")
    for _, _, company, title, reason in to_delete[:20]:
        print(f"  [{reason}] {company} — {title}")
    print("\nRun with --apply to commit.")
else:
    print("Nothing to remove — DB is already clean.")

con.close()
