from __future__ import annotations

import time
from typing import Any, Callable

from agentguard.breaker import CircuitBreaker
from agentguard.exceptions import CircuitBreakerTripped
from agentguard.extractors import AnthropicExtractor, GenericExtractor, OpenAIExtractor
from agentguard.models import BreakerEvent, SessionRecord, Turn
from agentguard.recorder import Recorder
from agentguard.rules.base import SessionState
from agentguard.storage import Storage

OnTripCallback = Callable[[BreakerEvent, SessionRecord], None]
OnTurnCallback = Callable[[Turn, SessionRecord], None]


class Session:
    def __init__(
        self,
        record: SessionRecord,
        storage: Storage,
        breaker: CircuitBreaker | None = None,
        on_trip: OnTripCallback | None = None,
        on_turn: OnTurnCallback | None = None,
    ):
        self._record = record
        self._storage = storage
        self._recorder = Recorder(record)
        self._breaker = breaker or CircuitBreaker()
        self._state = SessionState()
        self._state.start_time = time.time()
        self._on_trip = on_trip
        self._on_turn = on_turn

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        event = self._breaker.evaluate(self._state)
        if event:
            self._trip(event)
            raise CircuitBreakerTripped(event)

        input_data = kwargs.get("messages", list(args[:1]))
        if not isinstance(input_data, list):
            input_data = [{"raw": str(input_data)}]

        start = time.perf_counter()
        try:
            response = fn(*args, **kwargs)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            turn = self._recorder.record_error(input_data, e, latency_ms)
            self._state.add_turn(turn)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        extractor = self._detect_extractor(response)
        turn = self._recorder.record_turn(input_data, response, extractor, latency_ms)

        if self._on_turn:
            self._on_turn(turn, self._record)

        self._state.add_turn(turn)

        event = self._breaker.evaluate(self._state)
        if event:
            self._trip(event)
            raise CircuitBreakerTripped(event)

        return response

    def _trip(self, event: BreakerEvent) -> None:
        self._record.breaker_event = event
        self._record.finalize("tripped")
        if self._on_trip:
            self._on_trip(event, self._record)

    def _detect_extractor(self, response: Any):
        module = type(response).__module__ or ""
        if "openai" in module:
            return OpenAIExtractor()
        if "anthropic" in module:
            return AnthropicExtractor()
        return GenericExtractor()

    def summary(self) -> dict:
        total_tokens = sum(t.tokens_in + t.tokens_out for t in self._record.turns)
        total_cost = sum(t.cost_usd for t in self._record.turns)
        return {
            "session_id": self._record.session_id,
            "agent_name": self._record.agent_name,
            "turns": len(self._record.turns),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "status": self._record.status,
        }

    def _save(self) -> None:
        self._storage.save_session(self._record)

    @property
    def record(self) -> SessionRecord:
        return self._record
