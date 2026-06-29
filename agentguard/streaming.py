from __future__ import annotations

from typing import Any, Callable, Iterator


class OpenAIStreamAccumulator:
    """Accumulate OpenAI-compatible chat completion stream chunks."""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.model: str | None = None
        self.usage: Any = None

    def consume(self, chunk: Any) -> None:
        model = getattr(chunk, "model", None)
        if model:
            self.model = model

        usage = getattr(chunk, "usage", None)
        if usage:
            self.usage = usage

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return

        delta = getattr(choices[0], "delta", None)
        if not delta:
            return

        content = getattr(delta, "content", None)
        if content:
            self.content_parts.append(content)

        for tc_delta in getattr(delta, "tool_calls", None) or []:
            index = getattr(tc_delta, "index", 0) or 0
            entry = self.tool_calls.setdefault(
                index, {"id": None, "name": None, "arguments": ""}
            )
            tc_id = getattr(tc_delta, "id", None)
            if tc_id:
                entry["id"] = tc_id
            fn = getattr(tc_delta, "function", None)
            if fn:
                name = getattr(fn, "name", None)
                if name:
                    entry["name"] = name
                args = getattr(fn, "arguments", None)
                if args:
                    entry["arguments"] = f"{entry['arguments']}{args}"

    @property
    def content(self) -> str | None:
        if not self.content_parts:
            return None
        return "".join(self.content_parts)

    def as_response(self) -> _AssembledOpenAIResponse:
        tool_calls = [
            _AssembledToolCall(
                tc_id=entry["id"],
                name=entry["name"],
                arguments=entry["arguments"] or None,
            )
            for _, entry in sorted(self.tool_calls.items())
            if entry.get("name")
        ]
        return _AssembledOpenAIResponse(
            model=self.model,
            usage=self.usage or _AssembledUsage(),
            content=self.content,
            tool_calls=tool_calls,
        )


class _AssembledUsage:
    prompt_tokens = 0
    completion_tokens = 0

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _AssembledFunction:
    def __init__(self, name: str | None, arguments: str | None) -> None:
        self.name = name
        self.arguments = arguments


class _AssembledToolCall:
    def __init__(self, tc_id: str | None, name: str | None, arguments: str | None) -> None:
        self.id = tc_id
        self.function = _AssembledFunction(name, arguments)


class _AssembledMessage:
    def __init__(self, content: str | None, tool_calls: list[_AssembledToolCall]) -> None:
        self.content = content
        self.tool_calls = tool_calls or None


class _AssembledChoice:
    def __init__(self, content: str | None, tool_calls: list[_AssembledToolCall]) -> None:
        self.message = _AssembledMessage(content, tool_calls)


class _AssembledOpenAIResponse:
    __module__ = "openai.types.chat"

    def __init__(
        self,
        model: str | None,
        usage: Any,
        content: str | None,
        tool_calls: list[_AssembledToolCall],
    ) -> None:
        self.model = model
        self.usage = usage
        self.choices = [_AssembledChoice(content, tool_calls)]


class GuardedStream:
    """Transparent iterator wrapper that records after the stream finishes."""

    def __init__(
        self,
        stream: Iterator[Any],
        *,
        on_complete: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self._stream = stream
        self._on_complete = on_complete
        self._on_error = on_error
        self._finalized = False

    def __iter__(self) -> GuardedStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finalize_success()
            raise
        except Exception as exc:
            self._finalize_error(exc)
            raise

        return chunk

    def _finalize_success(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._on_complete(None)

    def _finalize_error(self, exc: Exception) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._on_error(exc)
