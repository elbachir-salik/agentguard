from __future__ import annotations

import time
from typing import Any, Callable, Iterator

from agentguard.breaker import CircuitBreaker
from agentguard.exceptions import CircuitBreakerTripped
from agentguard.extractors import AnthropicExtractor, GenericExtractor, OpenAIExtractor
from agentguard.models import BreakerEvent, SessionRecord, Turn
from agentguard.recorder import Recorder
from agentguard.rules.base import SessionState
from agentguard.storage import Storage
from agentguard.streaming import GuardedStream, OpenAIStreamAccumulator

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

        input_data = self._input_data_from_kwargs(args, kwargs)

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

    def stream(self, fn: Callable, *args: Any, **kwargs: Any) -> GuardedStream:
        event = self._breaker.evaluate(self._state)
        if event:
            self._trip(event)
            raise CircuitBreakerTripped(event)

        input_data = self._input_data_from_kwargs(args, kwargs)
        start = time.perf_counter()
        accumulator = OpenAIStreamAccumulator()

        try:
            raw_stream = fn(*args, **kwargs)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            turn = self._recorder.record_error(input_data, e, latency_ms)
            self._state.add_turn(turn)
            raise

        if isinstance(raw_stream, (dict, str, bytes)):
            raise TypeError(
                "stream() expects fn() to return an iterator of chunks; "
                f"got {type(raw_stream).__name__}. Use session.call() for non-streaming responses."
            )
        try:
            stream_iter = iter(raw_stream)
        except TypeError as exc:
            raise TypeError(
                "stream() expects fn() to return an iterator of chunks. "
                "Use session.call() for non-streaming responses."
            ) from exc

        def on_complete(_: Any) -> None:
            latency_ms = (time.perf_counter() - start) * 1000
            response = accumulator.as_response()
            turn = self._recorder.record_turn(
                input_data, response, OpenAIExtractor(), latency_ms
            )
            if self._on_turn:
                self._on_turn(turn, self._record)
            self._state.add_turn(turn)
            event = self._breaker.evaluate(self._state)
            if event:
                self._trip(event)
                raise CircuitBreakerTripped(event)

        def on_error(exc: Exception) -> None:
            latency_ms = (time.perf_counter() - start) * 1000
            turn = self._recorder.record_error(input_data, exc, latency_ms)
            self._state.add_turn(turn)

        def tracking_stream() -> Iterator[Any]:
            for chunk in stream_iter:
                accumulator.consume(chunk)
                yield chunk

        return GuardedStream(
            tracking_stream(),
            on_complete=on_complete,
            on_error=on_error,
        )

    def _input_data_from_kwargs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list:
        input_data = kwargs.get("messages", list(args[:1]))
        if not isinstance(input_data, list):
            input_data = [{"raw": str(input_data)}]
        return input_data

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
            "metadata": self._record.metadata,
        }

    def _save(self) -> None:
        self._storage.save_session(self._record)

    @property
    def record(self) -> SessionRecord:
        return self._record
