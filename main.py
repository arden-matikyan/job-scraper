"""job-scraper CLI.

    python main.py run                      run the pipeline once for all tracked URLs
    python main.py watch                    run once, then schedule on an interval
    python main.py recon <url>              run the recon agent on a single URL
    python main.py add <url>                recon a URL, then save it to tracked_urls.yaml
    python main.py jobs [--search "..."]    print recent / matching jobs from the DB
    python main.py reextract --ids 6531     re-run LLM extraction on specific DB rows

Every command logs registered scrapers + tracked-URL count and checks LLM health
on startup. The LLM provider is selected in agent/llm.py (Claude or Ollama).
Pipeline commands exit if the LLM is down; `jobs` only warns (DB reads don't need
the model). File logging goes to logs/scraper.log; console uses rich.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import yaml
from rich.console import Console
from rich.table import Table

from agent.llm import LLMClient, get_llm_client, resolve_provider
from agent.recon_agent import PlatformKB, ReconAgent, ReconStatus, config_path, project_root
from filter.job_filter import JobFilter
from pipeline.runner import Runner, load_tracked_urls
from pipeline.scheduler import PipelineScheduler
from scrapers.registry import registry
from storage.job_store import JobStore

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logs_dir = os.path.join(project_root(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    handler = RotatingFileHandler(
        os.path.join(logs_dir, "scraper.log"),
        maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # File logging only; console output is via rich in runner + main.
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)


def banner(console: Console, require_llm: bool = True) -> LLMClient:
    registry.discover()
    keys = registry.all_keys()
    console.print(f"[bold]job-scraper[/] — {len(keys)} scrapers registered: {', '.join(keys)}")
    tracked = load_tracked_urls().get("companies", []) or []
    console.print(f"Tracked URLs: {len(tracked)}")

    provider = resolve_provider()
    llm = get_llm_client()
    if llm.check_health():
        embed = llm.embed_model or "off"
        console.print(f"[green]LLM OK[/] (provider={provider}, model={llm.model}, embed={embed})\n")
    else:
        if provider == "claude":
            console.print(
                "[red]Claude is not available[/].\n"
                "Set [bold]ANTHROPIC_API_KEY[/] and confirm the model id in agent/anthropic_client.py."
            )
        else:
            console.print(
                "[red]Ollama is not available[/] at http://localhost:11434.\n"
                "Start it with [bold]ollama serve[/] and pull the models:\n"
                "  ollama pull llama3.2\n  ollama pull nomic-embed-text"
            )
        if require_llm:
            sys.exit(1)
        console.print("[yellow]Continuing without the LLM (DB read only).[/]\n")
    return llm


def cmd_run(console: Console, no_embed: bool = False, workers: int | None = None,
            keywords: str | None = None, company: str | None = None,
            company_workers: int | None = None) -> None:
    ollama = banner(console, require_llm=True)
    entries = load_tracked_urls().get("companies", []) or []
    if not entries:
        console.print("[yellow]No tracked URLs. Add one with `python main.py add <url>`.[/]")
        return
    if company:
        entries = [e for e in entries if (e.get("name") or "").lower() == company.lower()]
        if not entries:
            console.print(f"[red]No tracked company named '{company}'. Check tracked_urls.yaml.[/]")
            return
    kw = None
    if keywords is not None:  # --keywords given (may be "" to disable)
        kw = [k.strip() for k in keywords.split(",") if k.strip()]
    runner = Runner(
        ollama=ollama,
        embed_jobs=(False if no_embed else None),
        extract_workers=workers,
        keyword_filter=kw,
        company_workers=company_workers,
    )
    console.print(
        f"[dim]run mode: embeddings {'OFF' if not runner.embed_jobs else 'on'}, "
        f"{runner.extract_workers} extraction worker(s), "
        f"{runner.company_workers} company worker(s), "
        f"keyword filter: {', '.join(runner.keyword_filter) if runner.keyword_filter else 'off'}[/]"
    )
    runner.run(entries)


def cmd_watch(console: Console) -> None:
    ollama = banner(console, require_llm=True)
    PipelineScheduler(Runner(ollama=ollama)).start(run_immediately=True)


def cmd_recon(console: Console, url: str) -> None:
    ollama = banner(console, require_llm=True)
    agent = ReconAgent(ollama=ollama)
    res = agent.investigate(url)
    _print_recon(console, res)


def cmd_recon_pending(console: Console) -> None:
    # Ollama is optional here — stage 3 reasoning degrades gracefully without it.
    ollama = banner(console, require_llm=False)
    entries = load_tracked_urls().get("companies", []) or []
    # "uncategorized" == no scraper built yet == no scraper_key pinned in the yaml.
    pending = [e for e in entries if not (e or {}).get("scraper_key")]
    if not pending:
        console.print("[green]Every tracked entry already has a scraper_key pinned.[/]")
        return
    console.print(f"Running recon (stages 1–3) on {len(pending)} uncategorized entr"
                  f"{'y' if len(pending) == 1 else 'ies'}…\n")
    agent = ReconAgent(ollama=ollama)
    results: list[tuple[str, object]] = []
    for entry in pending:
        res = agent.investigate(entry.get("url", ""), company_name=entry.get("name"))
        results.append((entry.get("name") or "?", res))
        _print_recon(console, res)

    table = Table(title="Recon-Pending Summary")
    table.add_column("Company", style="cyan")
    table.add_column("Status")
    table.add_column("Platform/Scraper")
    table.add_column("Conf.", justify="right")
    table.add_column("Notes", max_width=40)
    for name, res in results:
        color = "green" if res.status == ReconStatus.MAPPED else "yellow"
        table.add_row(
            name,
            f"[{color}]{res.status}[/]",
            res.scraper_key or res.platform or "-",
            f"{res.confidence:.2f}",
            res.notes,
        )
    console.print(table)

    needs = [name for name, res in results if res.status != ReconStatus.MAPPED]
    if needs:
        console.print(
            f"\n[yellow]{len(needs)} entr{'y' if len(needs) == 1 else 'ies'} need a "
            f"hand-built scraper:[/] {', '.join(needs)}"
        )


def cmd_add(console: Console, url: str) -> None:
    ollama = banner(console, require_llm=True)
    agent = ReconAgent(ollama=ollama)
    res = agent.investigate(url)
    _print_recon(console, res)

    from urllib.parse import urlparse

    default_name = urlparse(url).netloc
    try:
        name = input(f"Company name [{default_name}]: ").strip() or default_name
    except (EOFError, KeyboardInterrupt):
        name = default_name

    path = config_path("tracked_urls.yaml")
    data = load_tracked_urls() or {}
    companies = data.setdefault("companies", [])
    if any((c or {}).get("url") == url for c in companies):
        console.print(f"[yellow]{url} is already tracked.[/]")
        return
    note = res.notes if res.status == ReconStatus.MAPPED else f"({res.status})"
    companies.append({"name": name, "url": url, "notes": note})
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        console.print(f"[green]Added[/] {name} -> tracked_urls.yaml")
    except Exception as exc:
        console.print(f"[red]Could not write tracked_urls.yaml: {exc}[/]")


def cmd_filter_jobs(
    console: Console,
    company: str | None,
    limit: int | None,
    rerun: bool,
    purge: bool = False,
) -> None:
    ollama = banner(console, require_llm=True)
    store = JobStore()
    job_filter = JobFilter(store, ollama)
    results = job_filter.run(company=company, limit=limit, rerun=rerun)
    qualified = results["qualified"]
    not_qualified = results["not_qualified"]

    nq_table = Table(title=f"[bold red]Not Qualified ({len(not_qualified)})[/]")
    nq_table.add_column("Title", style="red", max_width=38)
    nq_table.add_column("Company", max_width=18)
    nq_table.add_column("Location", max_width=22)
    nq_table.add_column("Reason", max_width=46)
    for j in not_qualified:
        nq_table.add_row(
            j.get("title") or "-",
            j.get("company") or "-",
            j.get("location") or "-",
            j.get("_filter_reason") or "-",
        )
    console.print(nq_table)

    q_table = Table(title=f"[bold green]Qualified ({len(qualified)})[/]")
    q_table.add_column("Title", style="cyan", max_width=40)
    q_table.add_column("Company", max_width=18)
    q_table.add_column("Location", max_width=24)
    q_table.add_column("Posted")
    for j in qualified:
        q_table.add_row(
            j.get("title") or "-",
            j.get("company") or "-",
            j.get("location") or "-",
            (j.get("posted_date") or "-")[:10],
        )
    console.print(q_table)

    total = len(qualified) + len(not_qualified)
    console.print(
        f"\n[dim]Evaluated {total} job(s) — "
        f"[green]{len(qualified)} qualified[/], "
        f"[red]{len(not_qualified)} not qualified[/dim]"
    )

    if purge:
        deleted = job_filter.delete_not_qualified()
        console.print(f"[red]Purged {deleted} not-qualified job(s) from the DB.[/]")


def cmd_reextract(
    console: Console,
    ids: list[int] | None,
    all_rows: bool,
    dry_run: bool,
) -> None:
    import json as _json

    ollama = banner(console, require_llm=True)
    store = JobStore()

    if all_rows:
        rows = store.get_jobs()
    else:
        rows = [r for i in (ids or []) if (r := store.get_job_by_id(i)) is not None]
        missing = [i for i in (ids or []) if store.get_job_by_id(i) is None]
        if missing:
            console.print(f"[yellow]Row IDs not found: {missing}[/]")

    if not rows:
        console.print("[yellow]No rows to re-extract.[/]")
        return

    from agent.job_extractor import JobExtractor
    extractor = JobExtractor(ollama)

    # Fields to pass as hints from the existing row (scraper-authoritative values win).
    HINT_KEYS = ("job_id", "title", "company", "location", "locations_all", "posted_date")

    updated = failed = unchanged = 0
    for row in rows:
        row_id = row["id"]
        raw_text = row.get("description_full") or ""
        source_url = row.get("source_url") or ""
        hints = {k: row[k] for k in HINT_KEYS if row.get(k) not in (None, "", [])}

        try:
            new = extractor.extract(raw_text, source_url, hints=hints)
        except Exception as exc:
            console.print(f"[red]id={row_id}: extraction error: {exc}[/]")
            failed += 1
            continue

        old_req = row.get("required_qualifications") or []
        old_pref = row.get("preferred_qualifications") or []
        new_req = new.get("required_qualifications") or []
        new_pref = new.get("preferred_qualifications") or []

        changed = (old_req != new_req) or (old_pref != new_pref)

        if dry_run:
            if changed:
                console.print(f"\n[bold]id={row_id}[/] {row.get('title')} @ {row.get('company')}")
                console.print(f"  [red]required (old)[/]: {_json.dumps(old_req, indent=2)}")
                console.print(f"  [green]required (new)[/]: {_json.dumps(new_req, indent=2)}")
                console.print(f"  [red]preferred (old)[/]: {_json.dumps(old_pref, indent=2)}")
                console.print(f"  [green]preferred (new)[/]: {_json.dumps(new_pref, indent=2)}")
                updated += 1
            else:
                unchanged += 1
        else:
            if changed:
                store.update_extraction(row_id, new)
                updated += 1
            else:
                unchanged += 1

    label = "Would update" if dry_run else "Updated"
    console.print(
        f"\n[dim]{label} {updated}, unchanged {unchanged}, failed {failed}[/]"
        + (" [yellow](dry-run — no writes)[/]" if dry_run else "")
    )


def cmd_jobs(console: Console, search: str | None) -> None:
    banner(console, require_llm=False)
    store = JobStore()
    jobs = store.search_jobs(search) if search else store.get_recent_jobs(20)
    title = f'Jobs matching "{search}"' if search else "Recent jobs"
    table = Table(title=f"{title} ({len(jobs)})")
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Company", max_width=18)
    table.add_column("Location", max_width=24)
    table.add_column("Scraper")
    table.add_column("Posted")
    for j in jobs:
        table.add_row(
            j.get("title") or "-",
            j.get("company") or "-",
            j.get("location") or "-",
            j.get("scraper_key") or "-",
            (j.get("posted_date") or "-")[:10],
        )
    console.print(table)
    if not jobs:
        console.print("[yellow]No jobs found. Run `python main.py run` first.[/]")


def _print_recon(console: Console, res) -> None:
    color = "green" if res.status == ReconStatus.MAPPED else "yellow"
    console.print(f"[bold]Recon[/] {res.url}")
    console.print(f"  status    : [{color}]{res.status}[/]")
    console.print(f"  scraper   : {res.scraper_key}")
    console.print(f"  platform  : {res.platform}")
    console.print(f"  confidence: {res.confidence:.2f}")
    console.print(f"  notes     : {res.notes}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="job-scraper", description="Local-first agentic job scraper")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the pipeline once for all tracked URLs")
    p_run.add_argument("--no-embed", action="store_true",
                       help="skip per-job embeddings for a faster bulk run (they're unused today)")
    p_run.add_argument("--workers", type=int, default=None,
                       help="concurrent extraction workers (default from config; 1 = sequential)")
    p_run.add_argument("--company-workers", type=int, default=None,
                       help="parallel company scrapers (default from config; 1 = sequential)")
    p_run.add_argument("--keywords", default=None,
                       help='comma-separated keyword filter override (e.g. "software,engineer"); '
                            'pass an empty string to disable filtering')
    p_run.add_argument("--company", default=None,
                       help='run only this company (case-insensitive name match from tracked_urls.yaml)')
    sub.add_parser("watch", help="run once, then schedule on an interval")
    p_recon = sub.add_parser("recon", help="run the recon agent on a single URL")
    p_recon.add_argument("url")
    sub.add_parser("recon-pending",
                   help="run recon (stages 1–3) on every tracked entry without a scraper_key")
    p_add = sub.add_parser("add", help="recon a URL then save it to tracked_urls.yaml")
    p_add.add_argument("url")
    p_jobs = sub.add_parser("jobs", help="print recent / matching jobs from the DB")
    p_jobs.add_argument("--search", default=None, help="keyword to search in title + description")
    p_filter = sub.add_parser(
        "filter-jobs", help="classify saved jobs as qualified / not qualified"
    )
    p_filter.add_argument("--company", default=None, help="only filter jobs from this company")
    p_filter.add_argument("--limit", type=int, default=None, help="max number of jobs to evaluate")
    p_filter.add_argument(
        "--rerun", action="store_true", help="re-evaluate already-filtered jobs"
    )
    p_filter.add_argument(
        "--purge", action="store_true",
        help="delete not-qualified jobs from the DB after classifying",
    )
    p_reex = sub.add_parser(
        "reextract",
        help="re-run LLM extraction on existing DB rows using stored description_full",
    )
    p_reex_group = p_reex.add_mutually_exclusive_group(required=True)
    p_reex_group.add_argument(
        "--ids", type=int, nargs="+", metavar="ID",
        help="one or more row IDs to re-extract",
    )
    p_reex_group.add_argument(
        "--all", action="store_true", dest="all_rows",
        help="re-extract every row in the DB",
    )
    p_reex.add_argument(
        "--dry-run", action="store_true",
        help="print before/after diffs without writing to the DB",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    try:  # load ANTHROPIC_API_KEY / overrides from a local .env if present
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    setup_logging()
    console = Console()
    args = build_parser().parse_args(argv)

    if args.command == "run":
        cmd_run(console, no_embed=args.no_embed, workers=args.workers, keywords=args.keywords,
                company=args.company, company_workers=args.company_workers)
    elif args.command == "watch":
        cmd_watch(console)
    elif args.command == "recon":
        cmd_recon(console, args.url)
    elif args.command == "recon-pending":
        cmd_recon_pending(console)
    elif args.command == "add":
        cmd_add(console, args.url)
    elif args.command == "jobs":
        cmd_jobs(console, args.search)
    elif args.command == "filter-jobs":
        cmd_filter_jobs(console, args.company, args.limit, args.rerun, args.purge)
    elif args.command == "reextract":
        cmd_reextract(console, args.ids, args.all_rows, args.dry_run)


if __name__ == "__main__":
    main()
