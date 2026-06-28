from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentguard.models import Turn


@dataclass
class RuleResult:
    tripped: bool
    reason: str = ""
    details: dict = field(default_factory=dict)

    @classmethod
    def passed(cls) -> RuleResult:
        return cls(tripped=False)

    @classmethod
    def trip(cls, reason: str, **details: object) -> RuleResult:
        return cls(tripped=True, reason=reason, details=details)


class BaseRule(ABC):
    name: str = "base"

    @abstractmethod
    def check(self, state: SessionState) -> RuleResult:
        ...


class SessionState:
    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.start_time: float = 0.0

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.total_cost += turn.cost_usd
        self.total_tokens += turn.tokens_in + turn.tokens_out

    def recent_tool_calls(self, window: int = 5) -> list[dict]:
        calls: list[dict] = []
        for turn in self.turns[-window:]:
            for tc in turn.tool_calls:
                calls.append(tc)
        return calls
