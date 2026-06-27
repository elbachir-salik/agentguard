from __future__ import annotations

from typing import Any

from agentguard.extractors.base import BaseExtractor


class OpenAIExtractor(BaseExtractor):
    def extract(self, response: Any) -> dict:
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0

        model = getattr(response, "model", None)

        tool_calls: list[dict] = []
        content: str | None = None

        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            if message:
                content = getattr(message, "content", None)
                raw_tool_calls = getattr(message, "tool_calls", None) or []
                for tc in raw_tool_calls:
                    fn = getattr(tc, "function", None)
                    tool_calls.append({
                        "id": getattr(tc, "id", None),
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    })

        return {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tool_calls": tool_calls,
            "content": content,
            "model": model,
        }
