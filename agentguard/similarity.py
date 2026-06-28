from __future__ import annotations

import json
from difflib import SequenceMatcher


def _normalize_arguments(arguments: object) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        try:
            return json.dumps(json.loads(arguments), sort_keys=True, default=str)
        except (json.JSONDecodeError, TypeError):
            return arguments
    return json.dumps(arguments, sort_keys=True, default=str)


def inputs_are_similar(input_a: dict, input_b: dict, threshold: float = 0.85) -> bool:
    str_a = _normalize_arguments(input_a.get("arguments"))
    str_b = _normalize_arguments(input_b.get("arguments"))
    if not str_a and not str_b:
        return True
    return SequenceMatcher(None, str_a, str_b).ratio() >= threshold
