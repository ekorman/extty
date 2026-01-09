"""Database connection and query management using raw SQL."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

DATABASE_PATH = Path.home() / ".extty" / "server.db"


def get_database_path() -> Path:
    """Get database path and ensure directory exists."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DATABASE_PATH


class Database:
    """Database abstraction layer supporting sqlite3 and psycopg (future)."""

    def __init__(self, db_path: Path | None = None, use_postgres: bool = False):
        self.db_path = db_path or get_database_path()
        self.use_postgres = use_postgres

        if not use_postgres:
            # Enable foreign keys for SQLite
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Get a database connection with transaction management."""
        if self.use_postgres:
            # Future: import psycopg and create connection
            raise NotImplementedError("PostgreSQL support not yet implemented")
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def execute(self, conn: Any, query: str, params: tuple | dict = ()) -> Any:
        """Execute a query and return cursor."""
        return conn.execute(query, params)

    def fetchone(self, conn: Any, query: str, params: tuple | dict = ()) -> dict | None:
        """Fetch one row as a dictionary."""
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetchall(self, conn: Any, query: str, params: tuple | dict = ()) -> list[dict]:
        """Fetch all rows as list of dictionaries."""
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def lastrowid(self, cursor: Any) -> int:
        """Get last inserted row ID."""
        return cursor.lastrowid


# Schema definition
SCHEMA_SQL = """
-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    config TEXT,  -- JSON stored as text
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    status TEXT DEFAULT 'running',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_project_name ON runs(project_id, name);

-- Metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_run_name ON metrics(run_id, name);

-- Metric points table
CREATE TABLE IF NOT EXISTS metric_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    value REAL NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (metric_id) REFERENCES metrics(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_metric_points_metric ON metric_points(metric_id);
CREATE INDEX IF NOT EXISTS idx_metric_points_step ON metric_points(metric_id, step);

-- System points table
CREATE TABLE IF NOT EXISTS system_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    ram_used_gb REAL,
    ram_total_gb REAL,
    gpu_mem_used_gb REAL,
    gpu_mem_total_gb REAL,
    gpu_util_pct REAL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_system_points_run ON system_points(run_id);
"""


def init_db(db: "Database | None" = None):
    """Initialize database schema."""
    if db is None:
        db = Database()

    with db.get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


# Global database instance
_db: "Database | None" = None


def get_db_instance() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
        init_db(_db)
    return _db


def get_db() -> Generator[Any, None, None]:
    """FastAPI dependency for database connection."""
    db = get_db_instance()
    with db.get_connection() as conn:
        yield conn


# Helper functions for JSON handling (SQLite stores JSON as text)
def encode_json(data: Any | None) -> str | None:
    """Encode data as JSON string."""
    return json.dumps(data) if data is not None else None


def decode_json(data: str | None) -> Any | None:
    """Decode JSON string to Python object."""
    return json.loads(data) if data else None
