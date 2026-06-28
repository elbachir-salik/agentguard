from __future__ import annotations

from typing import Any

from agentguard.extractors.base import BaseExtractor


class AnthropicExtractor(BaseExtractor):
    def extract(self, response: Any) -> dict:
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0

        model = getattr(response, "model", None)

        tool_calls: list[dict] = []
        content: str | None = None

        content_blocks = getattr(response, "content", None) or []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "")
                content = text if content is None else content + text
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "arguments": getattr(block, "input", None),
                })

        return {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tool_calls": tool_calls,
            "content": content,
            "model": model,
        }
