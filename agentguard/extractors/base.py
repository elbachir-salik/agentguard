from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, response: Any) -> dict:
        """Extract tokens, cost, tool_calls, content from an LLM response."""
        ...
