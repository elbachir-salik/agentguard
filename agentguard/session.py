from __future__ import annotations

import time
from typing import Any, Callable

from agentguard.breaker import CircuitBreaker
from agentguard.exceptions import CircuitBreakerTripped
from agentguard.extractors import AnthropicExtractor, GenericExtractor, OpenAIExtractor
from agentguard.models import SessionRecord
from agentguard.recorder import Recorder
from agentguard.rules.base import SessionState
from agentguard.storage import Storage


class Session:
    def __init__(
        self,
        record: SessionRecord,
        storage: Storage,
        breaker: CircuitBreaker | None = None,
    ):
        self._record = record
        self._storage = storage
        self._recorder = Recorder(record)
        self._breaker = breaker or CircuitBreaker()
        self._state = SessionState()
        self._state.start_time = time.time()
        self._tripped = False

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
            self._recorder.record_error(input_data, e, latency_ms)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        extractor = self._detect_extractor(response)
        turn = self._recorder.record_turn(input_data, response, extractor, latency_ms)

        self._state.add_turn(turn)

        event = self._breaker.evaluate(self._state)
        if event:
            self._trip(event)
            raise CircuitBreakerTripped(event)

        return response

    def _trip(self, event) -> None:
        self._record.breaker_event = event
        self._record.finalize("tripped")
        self._tripped = True

    def _detect_extractor(self, response: Any):
        module = type(response).__module__ or ""
        if "openai" in module:
            return OpenAIExtractor()
        if "anthropic" in module:
            return AnthropicExtractor()
        return GenericExtractor()

    def summary(self) -> dict:
        self._record.finalize(
            self._record.status if self._record.status != "running" else "completed"
        )
        return {
            "session_id": self._record.session_id,
            "agent_name": self._record.agent_name,
            "turns": len(self._record.turns),
            "total_tokens": self._record.total_tokens,
            "total_cost_usd": round(self._record.total_cost_usd, 6),
            "status": self._record.status,
        }

    def _save(self) -> None:
        self._storage.save_session(self._record)

    @property
    def record(self) -> SessionRecord:
        return self._record
