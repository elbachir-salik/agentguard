from __future__ import annotations

from agentguard.models import BreakerEvent
from agentguard.rules.base import BaseRule, SessionState


class CircuitBreaker:
    def __init__(self, rules: list[BaseRule] | None = None):
        self.rules = rules or []

    def evaluate(self, state: SessionState) -> BreakerEvent | None:
        for rule in self.rules:
            result = rule.check(state)
            if result.tripped:
                return BreakerEvent(
                    rule=rule.name,
                    trigger=result.reason,
                    turn=len(state.turns),
                    details=result.details,
                )
        return None
