from __future__ import annotations

import json
from dataclasses import asdict

from agentguard.models import SessionRecord


def session_to_dict(record: SessionRecord) -> dict:
    return asdict(record)


def session_to_json(record: SessionRecord, indent: int = 2) -> str:
    return json.dumps(session_to_dict(record), indent=indent)
