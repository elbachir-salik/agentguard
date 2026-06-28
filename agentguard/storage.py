from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

from agentguard.models import BreakerEvent, SessionRecord, Turn

_DEFAULT_DB = os.path.join(Path.home(), ".agentguard", "agentguard.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    total_tokens INTEGER,
    total_cost_usd REAL,
    breaker_event_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    tool_calls_json TEXT,
    status TEXT NOT NULL,
    model TEXT,
    UNIQUE(session_id, turn_number)
);

CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_name);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
"""


class Storage:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_session(self, record: SessionRecord) -> None:
        breaker_json = json.dumps(asdict(record.breaker_event)) if record.breaker_event else None

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, agent_name, started_at, ended_at, status,
                    total_tokens, total_cost_usd, breaker_event_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.session_id, record.agent_name, record.started_at,
                    record.ended_at, record.status, record.total_tokens,
                    record.total_cost_usd, breaker_json,
                ),
            )

            conn.execute("DELETE FROM turns WHERE session_id = ?", (record.session_id,))

            for turn in record.turns:
                conn.execute(
                    """INSERT INTO turns
                       (session_id, turn_number, timestamp, input_json, output_json,
                        tokens_in, tokens_out, cost_usd, latency_ms,
                        tool_calls_json, status, model)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.session_id, turn.turn_number, turn.timestamp,
                        json.dumps(turn.input_messages), json.dumps(turn.output),
                        turn.tokens_in, turn.tokens_out, turn.cost_usd,
                        turn.latency_ms, json.dumps(turn.tool_calls),
                        turn.status, turn.model,
                    ),
                )

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None

            turn_rows = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number",
                (session_id,),
            ).fetchall()

            turns = [
                Turn(
                    turn_number=t["turn_number"],
                    timestamp=t["timestamp"],
                    input_messages=json.loads(t["input_json"]),
                    output=json.loads(t["output_json"]),
                    tokens_in=t["tokens_in"],
                    tokens_out=t["tokens_out"],
                    cost_usd=t["cost_usd"],
                    latency_ms=t["latency_ms"],
                    tool_calls=json.loads(t["tool_calls_json"]) if t["tool_calls_json"] else [],
                    status=t["status"],
                    model=t["model"],
                )
                for t in turn_rows
            ]

            breaker_event = None
            if row["breaker_event_json"]:
                be = json.loads(row["breaker_event_json"])
                breaker_event = BreakerEvent(**be)

            return SessionRecord(
                session_id=row["session_id"],
                agent_name=row["agent_name"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                status=row["status"],
                turns=turns,
                total_tokens=row["total_tokens"] or 0,
                total_cost_usd=row["total_cost_usd"] or 0.0,
                breaker_event=breaker_event,
            )

    def list_sessions(
        self,
        agent_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        query = "SELECT session_id, agent_name, started_at, ended_at, status, total_tokens, total_cost_usd FROM sessions WHERE 1=1"
        params: list = []

        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self, agent_name: str | None = None) -> dict:
        query = """SELECT
            COUNT(*) as total_sessions,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost_usd) as total_cost,
            SUM(CASE WHEN status = 'tripped' THEN 1 ELSE 0 END) as trips,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM sessions"""
        params: list = []

        if agent_name:
            query += " WHERE agent_name = ?"
            params.append(agent_name)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

            return {
                "total_sessions": row["total_sessions"],
                "total_tokens": row["total_tokens"] or 0,
                "total_cost_usd": row["total_cost"] or 0.0,
                "trips": row["trips"],
                "completed": row["completed"],
                "errors": row["errors"],
            }
