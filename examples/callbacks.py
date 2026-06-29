"""Callbacks example: react to every turn and circuit breaker trips.

Set SLACK_WEBHOOK_URL to also post trip alerts to Slack (optional).

    export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
    python examples/callbacks.py
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from agentguard import CircuitBreakerTripped, Guard
from agentguard.models import BreakerEvent, SessionRecord, Turn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("agentguard.demo")


def on_turn(turn: Turn, record: SessionRecord) -> None:
    tok = turn.tokens_in + turn.tokens_out
    logger.info(
        "Turn %s | %s tok | $%.4f | session=%s",
        turn.turn_number,
        tok,
        turn.cost_usd,
        record.session_id,
    )


def _post_slack(webhook_url: str, text: str) -> None:
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def on_trip(event: BreakerEvent, record: SessionRecord) -> None:
    message = (
        f"Circuit breaker tripped for {record.agent_name}\n"
        f"Session: {record.session_id}\n"
        f"Rule: {event.rule}\n"
        f"Cause: {event.trigger}"
    )
    logger.warning(message)

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return

    try:
        _post_slack(webhook_url, message)
        logger.info("Slack alert sent")
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.error("Failed to send Slack alert: %s", exc)


class _MockUsage:
    prompt_tokens = 50
    completion_tokens = 30


class _MockMessage:
    content = "Done"
    tool_calls = None


class _MockChoice:
    message = _MockMessage()


class _MockResponse:
    __module__ = "openai.types.chat"
    usage = _MockUsage()
    model = "gpt-4o"
    choices = [_MockChoice()]


guard = Guard(
    agent_name="support-bot",
    max_turns=2,
    on_turn=on_turn,
    on_trip=on_trip,
)

try:
    with guard.session(metadata={"env": "staging", "ticket": "1234"}) as session:
        session.call(lambda: _MockResponse())
        session.call(lambda: _MockResponse())
        session.call(lambda: _MockResponse())  # trips on turn 3
except CircuitBreakerTripped as exc:
    logger.info("Caught: %s", exc)
