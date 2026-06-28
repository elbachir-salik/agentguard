from __future__ import annotations

from typing import Any

from agentguard.extractors.base import BaseExtractor
from agentguard.models import SessionRecord, Turn, _now
from agentguard.pricing import estimate_cost


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


class Recorder:
    def __init__(self, session_record: SessionRecord):
        self._record = session_record
        self._turn_count = 0

    def record_turn(
        self,
        input_data: list,
        response: object,
        extractor: BaseExtractor,
        latency_ms: float,
    ) -> Turn:
        self._turn_count += 1
        extracted = extractor.extract(response)

        model = extracted.get("model")
        tokens_in = extracted.get("tokens_in", 0)
        tokens_out = extracted.get("tokens_out", 0)
        cost = estimate_cost(model or "", tokens_in, tokens_out)

        turn = Turn(
            turn_number=self._turn_count,
            timestamp=_now(),
            input_messages=_make_serializable(input_data),
            output={"content": extracted.get("content")},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            tool_calls=_make_serializable(extracted.get("tool_calls", [])),
            status="success",
            model=model,
        )

        self._record.turns.append(turn)
        return turn

    def record_error(
        self,
        input_data: list,
        error: Exception,
        latency_ms: float,
    ) -> Turn:
        self._turn_count += 1

        turn = Turn(
            turn_number=self._turn_count,
            timestamp=_now(),
            input_messages=_make_serializable(input_data),
            output={"error": str(error), "type": type(error).__name__},
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            latency_ms=latency_ms,
            tool_calls=[],
            status="error",
            model=None,
        )

        self._record.turns.append(turn)
        return turn
