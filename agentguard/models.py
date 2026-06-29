from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Turn:
    turn_number: int
    timestamp: str
    input_messages: list[dict]
    output: dict
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    tool_calls: list[dict]
    status: str  # "success" | "error"
    model: str | None = None


@dataclass
class BreakerEvent:
    rule: str
    trigger: str
    turn: int
    details: dict = field(default_factory=dict)


@dataclass
class WarnEvent:
    rule: str
    trigger: str
    turn: int
    details: dict = field(default_factory=dict)


@dataclass
class SessionRecord:
    session_id: str = field(default_factory=_new_id)
    agent_name: str = ""
    parent_session_id: str | None = None
    started_at: str = field(default_factory=_now)
    ended_at: str | None = None
    status: str = "running"  # "completed" | "tripped" | "error"
    turns: list[Turn] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    breaker_event: BreakerEvent | None = None
    metadata: dict = field(default_factory=dict)

    def finalize(self, status: str = "completed") -> None:
        self.ended_at = _now()
        self.status = status
        self.total_tokens = sum(t.tokens_in + t.tokens_out for t in self.turns)
        self.total_cost_usd = sum(t.cost_usd for t in self.turns)


class SessionSummary(TypedDict):
    session_id: str
    agent_name: str
    started_at: str
    ended_at: str | None
    status: str
    total_tokens: int | None
    total_cost_usd: float | None
    metadata: dict
    parent_session_id: str | None


class AncestorInfo(TypedDict):
    session_id: str
    agent_name: str
    status: str
    parent_session_id: str | None


class StatsResult(TypedDict):
    total_sessions: int
    total_tokens: int
    total_cost_usd: float
    trips: int
    completed: int
    errors: int
