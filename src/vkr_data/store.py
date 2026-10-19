"""Small SQLite storage contract shared by downloading and dataset building."""

import json
import sqlite3
from pathlib import Path


def database_path(config):
    return Path(config["dataset_dir"]) / "market.sqlite3"


def connect(config):
    """Open the snapshot database; callers own transactions and the connection."""
    path = database_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    initialize(connection)
    return connection


def initialize(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL, interval TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open TEXT, high TEXT, low TEXT, close TEXT, volume TEXT,
            close_time INTEGER, quote_volume TEXT, trade_count INTEGER,
            taker_base_volume TEXT, taker_quote_volume TEXT,
            quality_flags TEXT NOT NULL,
            PRIMARY KEY(symbol, interval, open_time)
        );
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, interval TEXT NOT NULL,
            start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS pages (
            job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
            raw_path TEXT NOT NULL, meta_path TEXT NOT NULL,
            sha256 TEXT NOT NULL, fetched_at TEXT NOT NULL,
            row_count INTEGER NOT NULL, duplicate_rows INTEGER NOT NULL,
            rejected_rows INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rejected_rows (
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            row_index INTEGER NOT NULL, raw_json TEXT NOT NULL,
            reason TEXT NOT NULL, PRIMARY KEY(job_id, row_index)
        );
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
    """)


def set_metadata(connection, key, value):
    connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                       (key, json.dumps(value, sort_keys=True, ensure_ascii=False)))


def get_metadata(connection, key, default=None):
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row[0])
