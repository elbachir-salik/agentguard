from __future__ import annotations

from agentguard.rules.base import BaseRule, RuleResult, SessionState


class ScopeRule(BaseRule):
    name = "scope"

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        blocked_tools: list[str] | None = None,
    ):
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.blocked_tools = set(blocked_tools) if blocked_tools else set()

    def check(self, state: SessionState) -> RuleResult:
        if not state.turns:
            return RuleResult.passed()

        last_turn = state.turns[-1]
        for tc in last_turn.tool_calls:
            tool_name = tc.get("name", "")
            if not tool_name:
                continue
            if self.blocked_tools and tool_name in self.blocked_tools:
                return RuleResult.trip(
                    f"Blocked tool used: {tool_name}",
                    tool=tool_name,
                )
            if self.allowed_tools is not None and tool_name not in self.allowed_tools:
                return RuleResult.trip(
                    f"Tool not in allowlist: {tool_name}",
                    tool=tool_name,
                    allowed=list(self.allowed_tools),
                )
        return RuleResult.passed()
