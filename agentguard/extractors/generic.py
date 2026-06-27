from __future__ import annotations

from typing import Any

from agentguard.extractors.base import BaseExtractor


class GenericExtractor(BaseExtractor):
    def extract(self, response: Any) -> dict:
        if isinstance(response, dict):
            return self._from_dict(response)
        return self._from_object(response)

    def _from_dict(self, resp: dict) -> dict:
        usage = resp.get("usage", {})
        return {
            "tokens_in": usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
            "tool_calls": resp.get("tool_calls", []),
            "content": resp.get("content", str(resp)),
            "model": resp.get("model"),
        }

    def _from_object(self, resp: Any) -> dict:
        usage = getattr(resp, "usage", None)
        return {
            "tokens_in": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "tokens_out": getattr(usage, "completion_tokens", 0) if usage else 0,
            "tool_calls": [],
            "content": str(resp),
            "model": getattr(resp, "model", None),
        }
