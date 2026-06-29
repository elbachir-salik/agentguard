from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from agentguard.models import (
    AncestorInfo,
    BreakerEvent,
    SessionRecord,
    SessionSummary,
    StatsResult,
    Turn,
)

_VALID_META_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.\-]*$")

_DEFAULT_DB = os.environ.get(
    "AGENTGUARD_DB_PATH",
    os.path.join(Path.home(), ".agentguard", "agentguard.db"),
)

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
    metadata_json TEXT,
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
        with self._connection() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "metadata_json" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN metadata_json TEXT")
        if "parent_session_id" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN parent_session_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id)"
        )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_session(self, record: SessionRecord) -> None:
        breaker_json = json.dumps(asdict(record.breaker_event)) if record.breaker_event else None
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        with self._connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, agent_name, started_at, ended_at, status,
                    total_tokens, total_cost_usd, breaker_event_json, metadata_json,
                    parent_session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.session_id, record.agent_name, record.started_at,
                    record.ended_at, record.status, record.total_tokens,
                    record.total_cost_usd, breaker_json, metadata_json,
                    record.parent_session_id,
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
        with self._connection() as conn:
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

            metadata = {}
            if row["metadata_json"]:
                metadata = json.loads(row["metadata_json"])

            return SessionRecord(
                session_id=row["session_id"],
                agent_name=row["agent_name"],
                parent_session_id=row["parent_session_id"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                status=row["status"],
                turns=turns,
                total_tokens=row["total_tokens"] or 0,
                total_cost_usd=row["total_cost_usd"] or 0.0,
                breaker_event=breaker_event,
                metadata=metadata,
            )

    def list_sessions(
        self,
        agent_name: str | None = None,
        status: str | None = None,
        metadata: dict[str, str] | None = None,
        parent_session_id: str | None = None,
        parent_session_id_prefix: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        query = """SELECT session_id, agent_name, started_at, ended_at, status,
                          total_tokens, total_cost_usd, metadata_json, parent_session_id
                   FROM sessions WHERE 1=1"""
        params: list = []

        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if status:
            query += " AND status = ?"
            params.append(status)
        if parent_session_id is not None:
            query += " AND parent_session_id = ?"
            params.append(parent_session_id)
        if parent_session_id_prefix is not None:
            query += " AND parent_session_id LIKE ?"
            params.append(f"{parent_session_id_prefix}%")
        if metadata:
            for key, value in metadata.items():
                if not _VALID_META_KEY.match(key):
                    raise ValueError(
                        f"Invalid metadata key: {key!r}. "
                        "Keys must be alphanumeric/underscore/dot/hyphen, "
                        "starting with a letter or underscore."
                    )
                query += " AND json_extract(metadata_json, '$.' || ?) = ?"
                params.extend([key, value])

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results: list[SessionSummary] = []
            for r in rows:
                results.append(SessionSummary(
                    session_id=r["session_id"],
                    agent_name=r["agent_name"],
                    started_at=r["started_at"],
                    ended_at=r["ended_at"],
                    status=r["status"],
                    total_tokens=r["total_tokens"],
                    total_cost_usd=r["total_cost_usd"],
                    metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                    parent_session_id=r["parent_session_id"],
                ))
            return results

    def list_child_sessions(self, parent_session_id: str, limit: int = 50) -> list[SessionSummary]:
        return self.list_sessions(parent_session_id=parent_session_id, limit=limit)

    def get_session_ancestors(self, session_id: str) -> list[AncestorInfo]:
        """Return ancestor sessions root-first (lightweight — no turn data loaded)."""
        ancestors: list[AncestorInfo] = []
        seen: set[str] = set()

        with self._connection() as conn:
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            current_parent = row["parent_session_id"] if row else None

            while current_parent:
                if current_parent in seen:
                    break
                seen.add(current_parent)

                parent_row = conn.execute(
                    """SELECT session_id, agent_name, status, parent_session_id
                       FROM sessions WHERE session_id = ?""",
                    (current_parent,),
                ).fetchone()

                if parent_row:
                    ancestors.insert(0, AncestorInfo(
                        session_id=parent_row["session_id"],
                        agent_name=parent_row["agent_name"],
                        status=parent_row["status"],
                        parent_session_id=parent_row["parent_session_id"],
                    ))
                    current_parent = parent_row["parent_session_id"]
                else:
                    ancestors.insert(0, AncestorInfo(
                        session_id=current_parent,
                        agent_name="?",
                        status="unknown",
                        parent_session_id=None,
                    ))
                    break

        return ancestors

    def find_sessions_by_prefix(self, session_id_prefix: str) -> list[SessionSummary]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT session_id, agent_name, started_at, ended_at, status,
                          total_tokens, total_cost_usd, metadata_json, parent_session_id
                   FROM sessions
                   WHERE session_id LIKE ?
                   ORDER BY started_at DESC""",
                (f"{session_id_prefix}%",),
            ).fetchall()
            return [
                SessionSummary(
                    session_id=r["session_id"],
                    agent_name=r["agent_name"],
                    started_at=r["started_at"],
                    ended_at=r["ended_at"],
                    status=r["status"],
                    total_tokens=r["total_tokens"],
                    total_cost_usd=r["total_cost_usd"],
                    metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                    parent_session_id=r["parent_session_id"],
                )
                for r in rows
            ]

    def get_cost_by_agent(self) -> dict[str, float]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT agent_name, SUM(total_cost_usd) as total_cost
                   FROM sessions
                   GROUP BY agent_name
                   ORDER BY total_cost DESC"""
            ).fetchall()
            return {row["agent_name"]: row["total_cost"] or 0.0 for row in rows}

    def get_daily_costs(self) -> tuple[list[str], list[float]]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT substr(started_at, 1, 10) as day, SUM(total_cost_usd) as total_cost
                   FROM sessions
                   GROUP BY day
                   ORDER BY day"""
            ).fetchall()
            labels = [row["day"] for row in rows]
            values = [row["total_cost"] or 0.0 for row in rows]
            return labels, values

    def get_stats(self, agent_name: str | None = None) -> StatsResult:
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

        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()

            return StatsResult(
                total_sessions=row["total_sessions"],
                total_tokens=row["total_tokens"] or 0,
                total_cost_usd=row["total_cost"] or 0.0,
                trips=row["trips"],
                completed=row["completed"],
                errors=row["errors"],
            )
