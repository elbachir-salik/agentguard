from __future__ import annotations

from agentguard.rules.base import BaseRule, RuleResult, SessionState
from agentguard.similarity import inputs_are_similar


class LoopRule(BaseRule):
    name = "loop_detection"

    def __init__(
        self,
        max_retries: int = 3,
        similarity_threshold: float = 0.85,
        window: int = 5,
    ):
        self.max_retries = max_retries
        self.similarity_threshold = similarity_threshold
        self.window = window

    def check(self, state: SessionState) -> RuleResult:
        calls = state.recent_tool_calls(window=self.window)
        if len(calls) < self.max_retries:
            return RuleResult.passed()

        groups: dict[str, list[dict]] = {}
        for tc in calls:
            name = tc.get("name", "")
            if name:
                groups.setdefault(name, []).append(tc)

        for tool_name, tool_calls in groups.items():
            if len(tool_calls) < self.max_retries:
                continue

            similar_count = 1
            for i in range(1, len(tool_calls)):
                if inputs_are_similar(
                    tool_calls[i - 1], tool_calls[i], self.similarity_threshold
                ):
                    similar_count += 1
                else:
                    similar_count = 1

                if similar_count >= self.max_retries:
                    return RuleResult.trip(
                        f"{tool_name} called {similar_count}x with similar input",
                        tool=tool_name,
                        count=similar_count,
                        threshold=self.similarity_threshold,
                    )

        return RuleResult.passed()
