from __future__ import annotations

import time
from typing import Any, Callable

from agentguard.extractors import GenericExtractor, OpenAIExtractor
from agentguard.models import SessionRecord
from agentguard.recorder import Recorder
from agentguard.storage import Storage


class Session:
    def __init__(self, record: SessionRecord, storage: Storage):
        self._record = record
        self._storage = storage
        self._recorder = Recorder(record)

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
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
        self._recorder.record_turn(input_data, response, extractor, latency_ms)

        return response

    def _detect_extractor(self, response: Any):
        module = type(response).__module__ or ""
        if "openai" in module:
            return OpenAIExtractor()
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
