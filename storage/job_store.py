"""SQLite job store (stdlib sqlite3, no ORM).

Three tables: jobs (full schema; list fields as JSON, embedding as a float32 BLOB),
seen_hashes (dedup), and recon_log (one row per recon run). Dedup is first-seen-
wins: a job whose hash is already in seen_hashes is skipped, never updated.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LIST_FIELDS = (
    "locations_all",
    "required_qualifications",
    "preferred_qualifications",
)

# Column order for the jobs table (excludes the autoincrement id).
_JOB_COLUMNS = (
    "job_id", "title", "company", "location", "locations_all",
    "description_full", "required_qualifications", "preferred_qualifications",
    "posted_date", "source_url", "scraper_key", "scraped_at", "platform",
    "embedding", "hash",
)


def _default_db_path() -> str:
    env = os.environ.get("JOB_SCRAPER_DB")
    if env:
        return env
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "jobs.db")


def compute_hash(company: Optional[str], title: Optional[str], source_url: Optional[str]) -> str:
    key = f"{company or ''}::{title or ''}::{source_url or ''}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _embedding_to_blob(embedding: Any) -> Optional[bytes]:
    if not embedding:
        return None
    try:
        import numpy as np

        return np.asarray(embedding, dtype=np.float32).tobytes()
    except Exception as exc:
        logger.warning("Could not serialize embedding: %s", exc)
        return None


class JobStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ----------------------------------------------------------------- schema
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT, title TEXT, company TEXT, location TEXT,
                    locations_all TEXT, description_full TEXT,
                    required_qualifications TEXT, preferred_qualifications TEXT,
                    posted_date TEXT, source_url TEXT, scraper_key TEXT,
                    scraped_at TEXT, platform TEXT, embedding BLOB,
                    hash TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS seen_hashes (
                    hash TEXT PRIMARY KEY,
                    first_seen TEXT
                );
                CREATE TABLE IF NOT EXISTS recon_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT, platform_detected TEXT, scraper_used TEXT,
                    jobs_found INTEGER, investigation_notes TEXT, timestamp TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
                CREATE INDEX IF NOT EXISTS idx_jobs_scraped_at ON jobs(scraped_at);
                """
            )
            # Safe migration: add filter columns if they don't exist yet.
            for col, ctype in [("filter_status", "TEXT"), ("filter_reason", "TEXT")]:
                try:
                    self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ctype}")
                    self._conn.commit()
                except sqlite3.OperationalError:
                    pass  # column already exists
            self._conn.commit()

    # ------------------------------------------------------------------- dedup
    def is_seen(self, hash_value: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM seen_hashes WHERE hash = ? LIMIT 1", (hash_value,)
            )
            return cur.fetchone() is not None

    # ------------------------------------------------------------------- saving
    @staticmethod
    def _bind_safe(col: str, val: Any):
        """Last-resort coercion so an unexpected list/dict can't break the INSERT."""
        if col == "embedding":
            return val  # bytes or None
        if isinstance(val, (list, dict)):
            return json.dumps(val, ensure_ascii=False)
        return val

    def save_job(self, record: dict) -> bool:
        """Insert a job (first-seen-wins). Returns True if newly inserted.

        Computes the dedup hash from company::title::source_url. Existing hashes
        are skipped and never updated.
        """
        from scrapers.base import scraped_at_stamp

        company = record.get("company")
        title = record.get("title")
        source_url = record.get("source_url")
        hash_value = record.get("hash") or compute_hash(company, title, source_url)

        if self.is_seen(hash_value):
            return False

        row = dict(record)
        row["hash"] = hash_value
        row.setdefault("scraped_at", scraped_at_stamp())
        for field in _LIST_FIELDS:
            row[field] = json.dumps(row.get(field) or [])
        row["embedding"] = _embedding_to_blob(record.get("embedding"))

        values = [self._bind_safe(col, row.get(col)) for col in _JOB_COLUMNS]
        placeholders = ", ".join("?" for _ in _JOB_COLUMNS)
        columns = ", ".join(_JOB_COLUMNS)

        with self._lock:
            try:
                cur = self._conn.execute(
                    f"INSERT OR IGNORE INTO jobs ({columns}) VALUES ({placeholders})",
                    values,
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO seen_hashes (hash, first_seen) VALUES (?, ?)",
                    (hash_value, row["scraped_at"]),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except Exception as exc:
                logger.error("save_job failed for %s: %s", source_url, exc)
                return False

    # ----------------------------------------------------------------- queries
    def _rows_to_dicts(self, rows) -> list[dict]:
        out = []
        for r in rows:
            d = dict(r)
            for field in _LIST_FIELDS:
                try:
                    d[field] = json.loads(d.get(field) or "[]")
                except Exception:
                    d[field] = []
            d["has_embedding"] = d.get("embedding") is not None
            d.pop("embedding", None)  # don't surface the raw blob
            out.append(d)
        return out

    def get_jobs(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM jobs ORDER BY id DESC")
            return self._rows_to_dicts(cur.fetchall())

    def get_recent_jobs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (int(limit),)
            )
            return self._rows_to_dicts(cur.fetchall())

    def search_jobs(self, keyword: str) -> list[dict]:
        like = f"%{keyword}%"
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE title LIKE ? OR description_full LIKE ? "
                "ORDER BY id DESC",
                (like, like),
            )
            return self._rows_to_dicts(cur.fetchall())

    def all_source_urls(self) -> set[str]:
        """Every source_url already stored — lets detail-fetch scrapers skip reruns."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT source_url FROM jobs WHERE source_url IS NOT NULL AND source_url != ''"
            )
            return {r["source_url"] for r in cur.fetchall()}

    def count_jobs(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) AS n FROM jobs")
            return int(cur.fetchone()["n"])

    # --------------------------------------------------------------- filtering

    def get_jobs_for_filter(
        self,
        company: Optional[str] = None,
        limit: Optional[int] = None,
        rerun: bool = False,
    ) -> list[dict]:
        """Return jobs to be evaluated by the filter.

        By default skips jobs that already have a filter_status so re-runs are
        fast.  Pass rerun=True to re-evaluate everything.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if not rerun:
            clauses.append("filter_status IS NULL")
        if company:
            clauses.append("LOWER(company) = LOWER(?)")
            params.append(company)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""

        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY id DESC {limit_clause}",
                params,
            )
            return self._rows_to_dicts(cur.fetchall())

    def update_filter_status(self, job_id: int, status: str, reason: str) -> None:
        """Persist the filter decision for a single job."""
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE jobs SET filter_status = ?, filter_reason = ? WHERE id = ?",
                    (status, reason, job_id),
                )
                self._conn.commit()
            except Exception as exc:
                logger.error("update_filter_status failed for id=%s: %s", job_id, exc)

    def delete_by_filter_status(self, status: str) -> int:
        """Delete all jobs with the given filter_status. Returns rows deleted."""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM jobs WHERE filter_status = ?", (status,)
                )
                self._conn.commit()
                return cur.rowcount
            except Exception as exc:
                logger.error("delete_by_filter_status failed for %r: %s", status, exc)
                return 0

    # --------------------------------------------------------------- recon log
    def log_recon(
        self,
        url: str,
        platform_detected: Optional[str],
        scraper_used: Optional[str],
        jobs_found: int,
        investigation_notes: str = "",
    ) -> None:
        from scrapers.base import utcnow_iso

        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO recon_log (url, platform_detected, scraper_used, "
                    "jobs_found, investigation_notes, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (url, platform_detected, scraper_used, int(jobs_found),
                     investigation_notes, utcnow_iso()),
                )
                self._conn.commit()
            except Exception as exc:
                logger.error("log_recon failed for %s: %s", url, exc)

    def get_recon_log(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM recon_log ORDER BY id DESC")
            return [dict(r) for r in cur.fetchall()]
