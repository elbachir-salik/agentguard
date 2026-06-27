from __future__ import annotations

import json
from difflib import SequenceMatcher


def inputs_are_similar(input_a: dict, input_b: dict, threshold: float = 0.85) -> bool:
    str_a = json.dumps(input_a, sort_keys=True, default=str)
    str_b = json.dumps(input_b, sort_keys=True, default=str)
    return SequenceMatcher(None, str_a, str_b).ratio() >= threshold
