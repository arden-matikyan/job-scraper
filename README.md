# job-scraper

Vibe coded to build company specicific scrapers. 

Local-first job scraper. Uses Ollama (llama3.2 + nomic-embed-text) to extract and store job listings from tracked company career pages.

Includes filtering at scrape level and DB level. 

TODO: 
Deploy on AWS to have automatic daily runs. 
Further enable agentic capabiltiy so scraper can be built, tested and deployed automatically from single URL input of new job board. 

## Prerequisites

- Python 3.x
- [Ollama](https://ollama.com) running locally

```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Commands

### `run` — scrape all tracked companies once

```bash
python main.py run
```

Options:

| Flag | Description |
|------|-------------|
| `--no-embed` | Skip per-job embeddings (faster bulk run) |
| `--workers N` | Number of concurrent extraction workers (default from config; 1 = sequential) |
| `--keywords "kw1,kw2"` | Comma-separated keyword filter override; pass `""` to disable filtering |
| `--company NAME` | Run only one company (case-insensitive name match from `tracked_urls.yaml`) |

Examples:

```bash
python main.py run --no-embed
python main.py run --workers 4
python main.py run --keywords "software,engineer"
python main.py run --company "Acme Corp"
```

---

### `watch` — run continuously on a schedule

```bash
python main.py watch
```

Runs the pipeline immediately, then repeats on a configured interval.

---

### `recon <url>` — identify which scraper to use for a URL

```bash
python main.py recon https://jobs.lever.co/example
```

Runs the recon agent on a single URL and prints the detected platform, scraper key, confidence, and notes.

---

### `recon-pending` — recon all unclassified tracked URLs

```bash
python main.py recon-pending
```

Runs recon on every entry in `tracked_urls.yaml` that doesn't yet have a `scraper_key`. Ollama is optional — stage 3 reasoning degrades gracefully without it.

---

### `add <url>` — recon a URL and add it to tracking

```bash
python main.py add https://jobs.lever.co/example
```

Reconnaissances the URL, prompts for a company name, then saves the entry to `tracked_urls.yaml`.

---

### `filter-jobs` — classify saved jobs as qualified or not qualified

```bash
python main.py filter-jobs
```

Reads jobs from the database and disqualifies any that require an active security clearance or 3+ years of experience. Results are saved back to the database so re-runs skip already-evaluated jobs.

Options:

| Flag | Description |
|------|-------------|
| `--company NAME` | Only evaluate jobs from this company |
| `--limit N` | Max number of jobs to evaluate |
| `--rerun` | Re-evaluate jobs that were already filtered |

Examples:

```bash
python main.py filter-jobs
python main.py filter-jobs --company "Lockheed Martin"
python main.py filter-jobs --limit 50
python main.py filter-jobs --rerun
```

Requires Ollama — clearance detection uses `llama3.2` to avoid false positives from phrases like "active state bar membership."

---

### `jobs` — view scraped jobs from the database

```bash
python main.py jobs
python main.py jobs --search "backend engineer"
```

Options:

| Flag | Description |
|------|-------------|
| `--search "..."` | Keyword to search in job title and description |

Prints the 20 most recent jobs by default. Ollama is not required for this command.

---

## Logging

- Console output uses [rich](https://github.com/Textualize/rich)
- File logs go to `logs/scraper.log` (rotating, max 5 MB × 3 backups)
