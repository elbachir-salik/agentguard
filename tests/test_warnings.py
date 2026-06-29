from unittest.mock import patch

import pytest

from agentguard import Guard, CircuitBreakerTripped
from agentguard.models import WarnEvent
from agentguard.rules.budget import BudgetRule


class _MockUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _MockMessage:
    content = "hello"
    tool_calls = None


class _MockChoice:
    message = _MockMessage()


class _MockResponse:
    __module__ = "openai.types.chat"
    usage = _MockUsage()
    model = "gpt-4o"
    choices = [_MockChoice()]


@patch("agentguard.recorder.estimate_cost", return_value=0.50)
def test_on_warn_fires_at_warn_cost(mock_cost, tmp_db):
    warnings: list[WarnEvent] = []

    def on_warn(event: WarnEvent, record) -> None:
        warnings.append(event)

    guard = Guard(
        agent_name="test",
        max_cost=2.00,
        warn_cost=0.50,
        db_path=tmp_db,
        on_warn=on_warn,
    )
    with guard.session() as session:
        session.call(lambda: _MockResponse())
        session.call(lambda: _MockResponse())

    assert len(warnings) == 1
    assert warnings[0].rule == "budget"
    assert warnings[0].details["kind"] == "warn_cost"
    assert warnings[0].turn == 1


@patch("agentguard.recorder.estimate_cost", return_value=0.50)
def test_on_warn_fires_once(mock_cost, tmp_db):
    warnings: list[WarnEvent] = []

    guard = Guard(
        agent_name="test",
        max_cost=5.00,
        warn_cost=0.50,
        db_path=tmp_db,
        on_warn=lambda event, _: warnings.append(event),
    )
    with guard.session() as session:
        for _ in range(4):
            session.call(lambda: _MockResponse())

    assert len(warnings) == 1


@patch("agentguard.recorder.estimate_cost", return_value=0.40)
def test_on_warn_fires_at_warn_pct(mock_cost, tmp_db):
    warnings: list[WarnEvent] = []

    guard = Guard(
        agent_name="test",
        max_cost=1.00,
        warn_pct=0.8,
        db_path=tmp_db,
        on_warn=lambda event, _: warnings.append(event),
    )
    with guard.session() as session:
        session.call(lambda: _MockResponse())  # $0.40 — no warn
        session.call(lambda: _MockResponse())  # $0.80 — warn at 80%

    assert len(warnings) == 1
    assert warnings[0].details["kind"] == "warn_pct"
    assert warnings[0].details["threshold"] == pytest.approx(0.8)


@patch("agentguard.recorder.estimate_cost", return_value=0.50)
def test_warn_does_not_trip_session(mock_cost, tmp_db):
    guard = Guard(
        agent_name="test",
        max_cost=2.00,
        warn_cost=0.50,
        db_path=tmp_db,
        on_warn=lambda *_: None,
    )
    with guard.session() as session:
        session.call(lambda: _MockResponse())
        session.call(lambda: _MockResponse())

    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session.record.session_id)
    assert record.status == "completed"
    assert len(record.turns) == 2


@patch("agentguard.recorder.estimate_cost", return_value=0.50)
def test_warn_then_trip(mock_cost, tmp_db):
    warnings: list[WarnEvent] = []

    guard = Guard(
        agent_name="test",
        max_cost=1.00,
        warn_pct=0.5,
        db_path=tmp_db,
        on_warn=lambda event, _: warnings.append(event),
    )
    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            session.call(lambda: _MockResponse())  # $0.50 — warn
            session.call(lambda: _MockResponse())  # $1.00 — trip

    assert len(warnings) == 1


def test_warn_pct_requires_max_cost():
    with pytest.raises(ValueError, match="warn_pct requires max_cost"):
        Guard(agent_name="test", warn_pct=0.8)


def test_warn_pct_with_budget_rule():
    guard = Guard(
        agent_name="test",
        rules=[BudgetRule(max_cost_usd=5.0)],
        warn_pct=0.8,
    )
    assert guard._max_cost == 5.0
