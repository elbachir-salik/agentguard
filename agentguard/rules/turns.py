from __future__ import annotations

from agentguard.rules.base import BaseRule, RuleResult, SessionState


class TurnsRule(BaseRule):
    name = "turns"

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns

    def check(self, state: SessionState) -> RuleResult:
        count = len(state.turns)
        if count >= self.max_turns:
            return RuleResult.trip(
                f"Max turns exceeded: {count} >= {self.max_turns}",
                current_turns=count,
                limit=self.max_turns,
            )
        return RuleResult.passed()
