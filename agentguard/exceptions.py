from __future__ import annotations

from agentguard.models import BreakerEvent


class CircuitBreakerTripped(Exception):
    def __init__(self, event: BreakerEvent):
        self.event = event
        super().__init__(f"Circuit breaker tripped: [{event.rule}] {event.trigger}")
