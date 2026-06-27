from __future__ import annotations

from agentguard.rules.base import BaseRule, RuleResult, SessionState


class BudgetRule(BaseRule):
    name = "budget"

    def __init__(
        self,
        max_cost_usd: float | None = None,
        max_tokens: int | None = None,
    ):
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens

    def check(self, state: SessionState) -> RuleResult:
        if self.max_cost_usd is not None and state.total_cost >= self.max_cost_usd:
            return RuleResult.trip(
                f"Budget exceeded: ${state.total_cost:.4f} >= ${self.max_cost_usd:.4f}",
                current_cost=state.total_cost,
                limit=self.max_cost_usd,
            )
        if self.max_tokens is not None and state.total_tokens >= self.max_tokens:
            return RuleResult.trip(
                f"Token limit exceeded: {state.total_tokens} >= {self.max_tokens}",
                current_tokens=state.total_tokens,
                limit=self.max_tokens,
            )
        return RuleResult.passed()
