"""LangChain / LangGraph integration for AgentGuard."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator
from uuid import UUID

from agentguard.extractors import GenericExtractor
from agentguard.guard import Guard
from agentguard.session import Session

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
    from langchain_core.outputs import LLMResult as _LLMResult

    _HAS_LANGCHAIN = True
except ImportError:
    _BaseCallbackHandler = object  # type: ignore[misc, assignment]
    _LLMResult = Any  # type: ignore[misc, assignment]
    _HAS_LANGCHAIN = False


def is_available() -> bool:
    return _HAS_LANGCHAIN


def _messages_to_dicts(messages: Any) -> list:
    if not messages:
        return []
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    result: list = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        elif hasattr(msg, "model_dump"):
            result.append(msg.model_dump())
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
            result.append({"role": role, "content": msg.content})
        else:
            result.append({"raw": str(msg)})
    return result


def _extract_tool_calls(message: Any) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None) or []
    result: list[dict] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("args") or tc.get("arguments", {})
            tc_id = tc.get("id", "")
        else:
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", {})
            tc_id = getattr(tc, "id", "")
        result.append({
            "id": tc_id,
            "name": name,
            "arguments": args if isinstance(args, str) else json.dumps(args),
        })
    return result


def _llm_result_to_response(output: LLMResult, model_name: str | None) -> dict:
    llm_output = output.llm_output or {}
    token_usage = llm_output.get("token_usage") or {}

    content = ""
    tool_calls: list[dict] = []
    resolved_model = llm_output.get("model_name") or model_name

    if output.generations:
        for gen_list in output.generations:
            if not gen_list:
                continue
            gen = gen_list[0]
            if hasattr(gen, "message") and gen.message is not None:
                content = getattr(gen.message, "content", "") or content
                tool_calls = _extract_tool_calls(gen.message) or tool_calls
            elif hasattr(gen, "text"):
                content = gen.text or content

    return {
        "usage": {
            "prompt_tokens": token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0),
        },
        "content": content,
        "tool_calls": tool_calls,
        "model": resolved_model,
    }


class AgentGuardCallbackHandler(_BaseCallbackHandler):
    """LangChain callback that records turns and enforces circuit-breaker rules."""

    def __init__(self, session: Session) -> None:
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for LangChain integration. "
                "Install with: pip install agentguard[langchain]"
            )
        super().__init__()
        self._session = session
        self._starts: dict[UUID, tuple[list, float, str | None]] = {}

    def _handle_start(
        self, run_id: UUID, messages: Any, *, serialized: dict | None = None
    ) -> None:
        input_messages = _messages_to_dicts(messages)
        model_name = None
        if serialized:
            model_name = serialized.get("kwargs", {}).get("model") or serialized.get("name")
        self._session.begin_turn(input_messages)
        self._starts[run_id] = (input_messages, time.perf_counter(), model_name)

    def _handle_end(self, run_id: UUID, response: LLMResult) -> None:
        start = self._starts.pop(run_id, None)
        if start is None:
            return
        input_messages, t0, model_name = start
        latency_ms = (time.perf_counter() - t0) * 1000
        payload = _llm_result_to_response(response, model_name)
        self._session.record_response(
            input_messages,
            payload,
            extractor=GenericExtractor(),
            latency_ms=latency_ms,
        )

    def _handle_error(self, run_id: UUID, error: BaseException) -> None:
        start = self._starts.pop(run_id, None)
        if start is None:
            return
        input_messages, t0, _ = start
        latency_ms = (time.perf_counter() - t0) * 1000
        self._session.record_failure(input_messages, error, latency_ms=latency_ms)

    # Chat models (LangChain 0.2+)
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        flat = messages[0] if messages else []
        self._handle_start(run_id, flat, serialized=serialized)

    def on_chat_model_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_end(run_id, response)

    def on_chat_model_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_error(run_id, error)

    # Legacy LLM interface
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        messages = [{"role": "user", "content": p} for p in prompts]
        self._handle_start(run_id, messages, serialized=serialized)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_end(run_id, response)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_error(run_id, error)


@contextmanager
def guard_session(
    guard: Guard,
    metadata: dict | None = None,
    parent_session_id: str | None = None,
) -> Generator[tuple[Session, list[AgentGuardCallbackHandler]], None, None]:
    """Open a Guard session and return ``(session, [callback_handler])`` for LangChain models."""
    with guard.session(metadata=metadata, parent_session_id=parent_session_id) as session:
        handler = AgentGuardCallbackHandler(session)
        yield session, [handler]
