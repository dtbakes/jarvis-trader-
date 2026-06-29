import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memory.db")

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            details TEXT
        );

        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            instrument TEXT,
            reasoning TEXT,
            outcome TEXT
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

# --- Events: short-term working memory ---

def log_event(event_type, summary, details=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO events (timestamp, event_type, summary, details) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), event_type, summary,
         json.dumps(details) if details else None)
    )
    conn.commit()
    conn.close()

def get_recent_events(limit=10):
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, event_type, summary FROM events ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

# --- Journal: trade decisions ---

def log_trade(action, instrument, reasoning, outcome=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO journal (timestamp, action, instrument, reasoning, outcome) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), action, instrument, reasoning, outcome)
    )
    conn.commit()
    conn.close()

def get_recent_trades(limit=5):
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, action, instrument, reasoning, outcome FROM journal ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

# --- Facts: long-term persistent memory ---

def set_fact(key, value):
    conn = get_db()
    conn.execute(
        "INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_fact(key):
    conn = get_db()
    row = conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else None

def get_all_facts():
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}

# Create tables on import
init_db()