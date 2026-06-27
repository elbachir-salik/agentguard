from __future__ import annotations

from agentguard.extractors.base import BaseExtractor
from agentguard.models import SessionRecord, Turn, _now
from agentguard.pricing import estimate_cost


class Recorder:
    def __init__(self, session_record: SessionRecord):
        self._record = session_record
        self._turn_count = 0

    def record_turn(
        self,
        input_data: list[dict],
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
            input_messages=input_data,
            output={"content": extracted.get("content")},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            tool_calls=extracted.get("tool_calls", []),
            status="success",
            model=model,
        )

        self._record.turns.append(turn)
        return turn

    def record_error(
        self,
        input_data: list[dict],
        error: Exception,
        latency_ms: float,
    ) -> Turn:
        self._turn_count += 1

        turn = Turn(
            turn_number=self._turn_count,
            timestamp=_now(),
            input_messages=input_data,
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
