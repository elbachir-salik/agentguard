from __future__ import annotations

import inspect
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable

from agentguard.breaker import CircuitBreaker
from agentguard.exceptions import CircuitBreakerTripped
from agentguard.extractors import (
    AnthropicExtractor,
    BaseExtractor,
    GenericExtractor,
    OpenAIExtractor,
)
from agentguard.models import BreakerEvent, SessionRecord, Turn, WarnEvent
from agentguard.recorder import Recorder
from agentguard.rules.base import SessionState
from agentguard.storage import Storage
from agentguard.streaming import GuardedAsyncStream, GuardedStream, OpenAIStreamAccumulator

OnTripCallback = Callable[[BreakerEvent, SessionRecord], None]
OnTurnCallback = Callable[[Turn, SessionRecord], None]
OnWarnCallback = Callable[[WarnEvent, SessionRecord], None]


class Session:
    """Wraps LLM calls with recording and circuit-breaker evaluation.

    Thread safety: a Session is bound to a single agent loop. If you share a
    Session across threads, all calls are serialized by an internal lock. For
    best performance, create one Session per thread/task instead.
    """

    def __init__(
        self,
        record: SessionRecord,
        storage: Storage,
        breaker: CircuitBreaker | None = None,
        on_trip: OnTripCallback | None = None,
        on_turn: OnTurnCallback | None = None,
        on_warn: OnWarnCallback | None = None,
        warn_cost: float | None = None,
        warn_pct: float | None = None,
        max_cost: float | None = None,
    ):
        self._record = record
        self._storage = storage
        self._recorder = Recorder(record)
        self._breaker = breaker or CircuitBreaker()
        self._state = SessionState()
        self._state.start_time = time.time()
        self._on_trip = on_trip
        self._on_turn = on_turn
        self._on_warn = on_warn
        self._warn_cost = warn_cost
        self._warn_pct = warn_pct
        self._max_cost = max_cost
        self._warned = False
        self._lock = threading.Lock()

    def call(
        self, fn: Callable, *args: Any, extractor: BaseExtractor | None = None, **kwargs: Any
    ) -> Any:
        input_data = self._pre_check(args, kwargs)

        start = time.perf_counter()
        try:
            response = fn(*args, **kwargs)
        except Exception as e:
            self._record_error(input_data, e, time.perf_counter() - start)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        ext = extractor or self._detect_extractor(response)
        self._record_success(input_data, response, ext, latency_ms)
        return response

    async def acall(
        self, fn: Callable, *args: Any, extractor: BaseExtractor | None = None, **kwargs: Any
    ) -> Any:
        input_data = self._pre_check(args, kwargs)

        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                response = await result
            else:
                response = result
        except Exception as e:
            self._record_error(input_data, e, time.perf_counter() - start)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        ext = extractor or self._detect_extractor(response)
        self._record_success(input_data, response, ext, latency_ms)
        return response

    def stream(self, fn: Callable, *args: Any, **kwargs: Any) -> GuardedStream:
        input_data = self._pre_check(args, kwargs)

        start = time.perf_counter()
        accumulator = OpenAIStreamAccumulator()

        try:
            raw_stream = fn(*args, **kwargs)
        except Exception as e:
            self._record_error(input_data, e, time.perf_counter() - start)
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
            self._record_success(input_data, response, OpenAIExtractor(), latency_ms)

        def on_error(exc: Exception) -> None:
            self._record_error(input_data, exc, time.perf_counter() - start)

        def tracking_stream() -> Iterator[Any]:
            for chunk in stream_iter:
                accumulator.consume(chunk)
                yield chunk

        return GuardedStream(
            tracking_stream(),
            on_complete=on_complete,
            on_error=on_error,
        )

    async def astream(self, fn: Callable, *args: Any, **kwargs: Any) -> GuardedAsyncStream:
        input_data = self._pre_check(args, kwargs)

        start = time.perf_counter()
        accumulator = OpenAIStreamAccumulator()

        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                raw_stream = await result
            else:
                raw_stream = result
        except Exception as e:
            self._record_error(input_data, e, time.perf_counter() - start)
            raise

        if isinstance(raw_stream, (dict, str, bytes)):
            raise TypeError(
                "astream() expects fn() to return an async iterator of chunks; "
                f"got {type(raw_stream).__name__}. Use session.acall() for non-streaming responses."
            )
        if not hasattr(raw_stream, "__aiter__"):
            raise TypeError(
                "astream() expects fn() to return an async iterator of chunks. "
                "Use session.stream() for synchronous streaming responses."
            )

        def on_complete(_: Any) -> None:
            latency_ms = (time.perf_counter() - start) * 1000
            response = accumulator.as_response()
            self._record_success(input_data, response, OpenAIExtractor(), latency_ms)

        def on_error(exc: Exception) -> None:
            self._record_error(input_data, exc, time.perf_counter() - start)

        async def tracking_stream() -> AsyncIterator[Any]:
            async for chunk in raw_stream:
                accumulator.consume(chunk)
                yield chunk

        return GuardedAsyncStream(
            tracking_stream(),
            on_complete=on_complete,
            on_error=on_error,
        )

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

    @property
    def record(self) -> SessionRecord:
        return self._record

    # ------------------------------------------------------------------
    # Internal helpers (shared by call/acall/stream/astream)
    # ------------------------------------------------------------------

    def _pre_check(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list:
        """Evaluate breaker and extract input data. Raises on trip."""
        with self._lock:
            event = self._breaker.evaluate(self._state)
            if event:
                self._trip(event)
                raise CircuitBreakerTripped(event)
            return self._input_data_from_kwargs(args, kwargs)

    def _record_success(
        self, input_data: list, response: Any, extractor: BaseExtractor, latency_ms: float
    ) -> None:
        """Record a successful turn, fire callbacks, check warnings/trip."""
        with self._lock:
            turn = self._recorder.record_turn(input_data, response, extractor, latency_ms)
            self._state.add_turn(turn)

            if self._on_turn:
                self._on_turn(turn, self._record)

            self._check_warnings()
            self._check_post_trip()

    def _record_error(self, input_data: list, exc: Exception, elapsed_s: float) -> None:
        """Record a failed turn (no callbacks, no trip check)."""
        latency_ms = elapsed_s * 1000
        with self._lock:
            turn = self._recorder.record_error(input_data, exc, latency_ms)
            self._state.add_turn(turn)

    def _check_post_trip(self) -> None:
        event = self._breaker.evaluate(self._state)
        if event:
            self._trip(event)
            raise CircuitBreakerTripped(event)

    def _check_warnings(self) -> None:
        if self._warned or not self._on_warn:
            return

        cost = self._state.total_cost
        event: WarnEvent | None = None

        if self._warn_cost is not None and cost >= self._warn_cost:
            event = WarnEvent(
                rule="budget",
                trigger=f"Cost warning: ${cost:.4f} >= ${self._warn_cost:.4f} (warn_cost)",
                turn=len(self._state.turns),
                details={
                    "kind": "warn_cost",
                    "current_cost": cost,
                    "warn_cost": self._warn_cost,
                },
            )
        elif self._warn_pct is not None and self._max_cost is not None:
            threshold = self._max_cost * self._warn_pct
            if cost >= threshold:
                event = WarnEvent(
                    rule="budget",
                    trigger=(
                        f"Cost warning: ${cost:.4f} >= {self._warn_pct:.0%} "
                        f"of ${self._max_cost:.4f} budget"
                    ),
                    turn=len(self._state.turns),
                    details={
                        "kind": "warn_pct",
                        "current_cost": cost,
                        "warn_pct": self._warn_pct,
                        "max_cost": self._max_cost,
                        "threshold": threshold,
                    },
                )

        if event:
            self._warned = True
            self._on_warn(event, self._record)

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

    def _detect_extractor(self, response: Any) -> BaseExtractor:
        module = type(response).__module__ or ""
        if "openai" in module:
            return OpenAIExtractor()
        if "anthropic" in module:
            return AnthropicExtractor()
        return GenericExtractor()

    def _save(self) -> None:
        self._storage.save_session(self._record)
