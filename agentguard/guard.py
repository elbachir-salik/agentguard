from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator

from agentguard.breaker import CircuitBreaker
from agentguard.exceptions import CircuitBreakerTripped
from agentguard.models import BreakerEvent, SessionRecord, Turn, WarnEvent
from agentguard.rules import BudgetRule, LoopRule, ScopeRule, TimeoutRule, TurnsRule
from agentguard.rules.base import BaseRule
from agentguard.session import Session
from agentguard.storage import Storage

OnTripCallback = Callable[[BreakerEvent, SessionRecord], None]
OnTurnCallback = Callable[[Turn, SessionRecord], None]
OnWarnCallback = Callable[[WarnEvent, SessionRecord], None]

_CONVENIENCE_KWARGS = (
    "max_cost",
    "max_tokens",
    "max_turns",
    "max_tool_retries",
    "timeout",
    "allowed_tools",
    "blocked_tools",
)


class Guard:
    def __init__(
        self,
        agent_name: str = "default",
        db_path: str | None = None,
        max_cost: float | None = None,
        max_tokens: int | None = None,
        max_turns: int | None = None,
        max_tool_retries: int | None = None,
        similarity_threshold: float = 0.85,
        timeout: float | None = None,
        allowed_tools: list[str] | None = None,
        blocked_tools: list[str] | None = None,
        rules: list[BaseRule] | None = None,
        on_trip: OnTripCallback | None = None,
        on_turn: OnTurnCallback | None = None,
        warn_cost: float | None = None,
        warn_pct: float | None = None,
        on_warn: OnWarnCallback | None = None,
    ):
        self.agent_name = agent_name
        self._storage = Storage(db_path=db_path)
        self._on_trip = on_trip
        self._on_turn = on_turn
        self._warn_cost = warn_cost
        self._warn_pct = warn_pct
        self._on_warn = on_warn

        convenience = {
            "max_cost": max_cost,
            "max_tokens": max_tokens,
            "max_turns": max_turns,
            "max_tool_retries": max_tool_retries,
            "timeout": timeout,
            "allowed_tools": allowed_tools,
            "blocked_tools": blocked_tools,
        }
        if rules is not None and any(convenience[k] is not None for k in _CONVENIENCE_KWARGS):
            raise ValueError(
                "Cannot pass both `rules` and convenience kwargs "
                f"({_CONVENIENCE_KWARGS}). Use one or the other."
            )

        self._rules = rules or self._build_default_rules(
            max_cost=max_cost,
            max_tokens=max_tokens,
            max_turns=max_turns,
            max_tool_retries=max_tool_retries,
            similarity_threshold=similarity_threshold,
            timeout=timeout,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
        )

        effective_max_cost = max_cost if max_cost is not None else self._max_cost_from_rules(self._rules)
        if warn_pct is not None and effective_max_cost is None:
            raise ValueError("warn_pct requires max_cost to be set (directly or via BudgetRule).")
        self._max_cost = effective_max_cost

    @staticmethod
    def _max_cost_from_rules(rules: list[BaseRule]) -> float | None:
        for rule in rules:
            if isinstance(rule, BudgetRule) and rule.max_cost_usd is not None:
                return rule.max_cost_usd
        return None

    def _build_default_rules(self, **kwargs) -> list[BaseRule]:
        rules: list[BaseRule] = []

        if kwargs["max_cost"] is not None or kwargs["max_tokens"] is not None:
            rules.append(BudgetRule(
                max_cost_usd=kwargs["max_cost"],
                max_tokens=kwargs["max_tokens"],
            ))
        if kwargs["max_turns"] is not None:
            rules.append(TurnsRule(max_turns=kwargs["max_turns"]))
        if kwargs["max_tool_retries"] is not None:
            rules.append(LoopRule(
                max_retries=kwargs["max_tool_retries"],
                similarity_threshold=kwargs["similarity_threshold"],
            ))
        if kwargs["timeout"] is not None:
            rules.append(TimeoutRule(timeout_seconds=kwargs["timeout"]))
        if kwargs["allowed_tools"] is not None or kwargs["blocked_tools"]:
            rules.append(ScopeRule(
                allowed_tools=kwargs["allowed_tools"],
                blocked_tools=kwargs["blocked_tools"],
            ))

        return rules

    @contextmanager
    def session(self, metadata: dict | None = None) -> Generator[Session, None, None]:
        record = SessionRecord(agent_name=self.agent_name, metadata=metadata or {})
        breaker = CircuitBreaker(rules=self._rules)
        session = Session(
            record=record,
            storage=self._storage,
            breaker=breaker,
            on_trip=self._on_trip,
            on_turn=self._on_turn,
            on_warn=self._on_warn,
            warn_cost=self._warn_cost,
            warn_pct=self._warn_pct,
            max_cost=self._max_cost,
        )

        try:
            yield session
            if record.status == "running":
                record.finalize("completed")
        except CircuitBreakerTripped:
            raise
        except Exception:
            record.finalize("error")
            raise
        finally:
            session._save()
