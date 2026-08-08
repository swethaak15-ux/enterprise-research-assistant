"""
memory.py

Module 9 - Memory

* Short-term memory: the ongoing conversation within a chat turn, held by
  LangGraph's SqliteSaver checkpointer and keyed by ``thread_id``.
* Persistent memory: previous client conversations survive process restarts
  because the checkpointer is backed by a SQLite file on disk.
* Long-term / client-profile memory: a small structured store (separate
  SQLite database) that remembers the client's name, preferred report
  style and which industries/topics they research most often, plus a
  history of generated reports per session.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver

import config

# ---------------------------------------------------------------------------
# Short-term / persistent conversation memory (LangGraph checkpointer)
# ---------------------------------------------------------------------------


def get_checkpointer() -> SqliteSaver:
    """
    Returns a LangGraph SqliteSaver. Because it is backed by a SQLite file,
    conversation state (short-term memory) also persists across app restarts
    for a given thread_id (persistent memory).
    """
    conn = sqlite3.connect(config.AGENT_MEMORY_DB, check_same_thread=False)
    return SqliteSaver(conn)


# ---------------------------------------------------------------------------
# Long-term client-profile memory
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS client_profiles (
    thread_id TEXT PRIMARY KEY,
    client_name TEXT,
    preferred_report_style TEXT,
    industries_researched TEXT DEFAULT '{}',
    last_active TEXT
);

CREATE TABLE IF NOT EXISTS report_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT,
    topic TEXT,
    created_at TEXT,
    report_json TEXT
);
"""


def _connect():
    conn = sqlite3.connect(config.PROFILE_DB, check_same_thread=False)
    conn.executescript(_SCHEMA)
    return conn


def touch_session(thread_id: str) -> None:
    """Record that a thread/session was just used, for the 'Previous Sessions' list."""
    now = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO client_profiles (thread_id, last_active)
            VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET last_active=excluded.last_active
            """,
            (thread_id, now),
        )
        conn.commit()


def list_sessions(limit: int = 20) -> List[str]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT thread_id FROM client_profiles ORDER BY last_active DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row[0] for row in rows]


def get_profile(thread_id: str) -> Dict:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT client_name, preferred_report_style, industries_researched FROM client_profiles WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()

    if not row:
        return {"client_name": None, "preferred_report_style": None, "industries_researched": {}}

    client_name, preferred_style, industries_json = row
    try:
        industries = json.loads(industries_json) if industries_json else {}
    except json.JSONDecodeError:
        industries = {}

    return {
        "client_name": client_name,
        "preferred_report_style": preferred_style,
        "industries_researched": industries,
    }


def save_profile(thread_id: str, client_name: Optional[str] = None, preferred_report_style: Optional[str] = None) -> None:
    touch_session(thread_id)
    with closing(_connect()) as conn:
        if client_name is not None:
            conn.execute(
                "UPDATE client_profiles SET client_name = ? WHERE thread_id = ?",
                (client_name, thread_id),
            )
        if preferred_report_style is not None:
            conn.execute(
                "UPDATE client_profiles SET preferred_report_style = ? WHERE thread_id = ?",
                (preferred_report_style, thread_id),
            )
        conn.commit()


def record_industry(thread_id: str, industry: str) -> None:
    """Increment the research-frequency counter for an industry/topic for this client."""
    if not industry:
        return

    touch_session(thread_id)
    profile = get_profile(thread_id)
    industries = profile["industries_researched"]
    industries[industry] = industries.get(industry, 0) + 1

    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE client_profiles SET industries_researched = ? WHERE thread_id = ?",
            (json.dumps(industries), thread_id),
        )
        conn.commit()


def top_industries(thread_id: str, n: int = 3) -> List[str]:
    industries = get_profile(thread_id)["industries_researched"]
    ranked = sorted(industries.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:n]]


def save_report(thread_id: str, topic: str, report_json: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO report_history (thread_id, topic, created_at, report_json) VALUES (?, ?, ?, ?)",
            (thread_id, topic, now, report_json),
        )
        conn.commit()
    record_industry(thread_id, topic)


def get_report_history(thread_id: str, limit: int = 10) -> List[Dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT topic, created_at, report_json FROM report_history WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
    return [{"topic": r[0], "created_at": r[1], "report": json.loads(r[2])} for r in rows]
