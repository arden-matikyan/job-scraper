"""job-scraper CLI.

    python main.py run                      run the pipeline once for all tracked URLs
    python main.py watch                    run once, then schedule on an interval
    python main.py recon <url>              run the recon agent on a single URL
    python main.py add <url>                recon a URL, then save it to tracked_urls.yaml
    python main.py jobs [--search "..."]    print recent / matching jobs from the DB

Every command logs registered scrapers + tracked-URL count and checks Ollama health
on startup. Pipeline commands exit if Ollama is down; `jobs` only warns (DB reads
don't need the model). File logging goes to logs/scraper.log; console uses rich.
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

from agent.ollama_client import OllamaClient
from agent.recon_agent import PlatformKB, ReconAgent, ReconStatus, config_path, project_root
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


def banner(console: Console, require_ollama: bool = True) -> OllamaClient:
    registry.discover()
    keys = registry.all_keys()
    console.print(f"[bold]job-scraper[/] — {len(keys)} scrapers registered: {', '.join(keys)}")
    tracked = load_tracked_urls().get("companies", []) or []
    console.print(f"Tracked URLs: {len(tracked)}")

    ollama = OllamaClient()
    if ollama.check_health():
        console.print(f"[green]Ollama OK[/] (model={ollama.model}, embed={ollama.embed_model})\n")
    else:
        console.print(
            "[red]Ollama is not available[/] at http://localhost:11434.\n"
            "Start it with [bold]ollama serve[/] and pull the models:\n"
            "  ollama pull llama3.2\n  ollama pull nomic-embed-text"
        )
        if require_ollama:
            sys.exit(1)
        console.print("[yellow]Continuing without Ollama (DB read only).[/]\n")
    return ollama


def cmd_run(console: Console, no_embed: bool = False, workers: int | None = None,
            keywords: str | None = None, company: str | None = None) -> None:
    ollama = banner(console, require_ollama=True)
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
    )
    console.print(
        f"[dim]run mode: embeddings {'OFF' if not runner.embed_jobs else 'on'}, "
        f"{runner.extract_workers} extraction worker(s), "
        f"keyword filter: {', '.join(runner.keyword_filter) if runner.keyword_filter else 'off'}[/]"
    )
    runner.run(entries)


def cmd_watch(console: Console) -> None:
    ollama = banner(console, require_ollama=True)
    PipelineScheduler(Runner(ollama=ollama)).start(run_immediately=True)


def cmd_recon(console: Console, url: str) -> None:
    ollama = banner(console, require_ollama=True)
    agent = ReconAgent(ollama=ollama)
    res = agent.investigate(url)
    _print_recon(console, res)


def cmd_recon_pending(console: Console) -> None:
    # Ollama is optional here — stage 3 reasoning degrades gracefully without it.
    ollama = banner(console, require_ollama=False)
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
    ollama = banner(console, require_ollama=True)
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


def cmd_jobs(console: Console, search: str | None) -> None:
    banner(console, require_ollama=False)
    store = JobStore()
    jobs = store.search_jobs(search) if search else store.get_recent_jobs(20)
    title = f'Jobs matching "{search}"' if search else "Recent jobs"
    table = Table(title=f"{title} ({len(jobs)})")
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Company", max_width=18)
    table.add_column("Location", max_width=24)
    table.add_column("Type")
    table.add_column("Scraper")
    table.add_column("Posted")
    for j in jobs:
        table.add_row(
            j.get("title") or "-",
            j.get("company") or "-",
            j.get("location") or "-",
            j.get("employment_type") or "-",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    console = Console()
    args = build_parser().parse_args(argv)

    if args.command == "run":
        cmd_run(console, no_embed=args.no_embed, workers=args.workers, keywords=args.keywords,
                company=args.company)
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


if __name__ == "__main__":
    main()
