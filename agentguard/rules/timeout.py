from __future__ import annotations

import time

from agentguard.rules.base import BaseRule, RuleResult, SessionState


class TimeoutRule(BaseRule):
    name = "timeout"

    def __init__(self, timeout_seconds: float = 300):
        self.timeout_seconds = timeout_seconds

    def check(self, state: SessionState) -> RuleResult:
        if state.start_time <= 0:
            return RuleResult.passed()
        elapsed = time.time() - state.start_time
        if elapsed >= self.timeout_seconds:
            return RuleResult.trip(
                f"Timeout: {elapsed:.1f}s >= {self.timeout_seconds}s",
                elapsed=elapsed,
                limit=self.timeout_seconds,
            )
        return RuleResult.passed()
