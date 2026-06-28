import pytest

from agentguard import Guard
from agentguard.exceptions import CircuitBreakerTripped


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


def test_session_completes_and_saves(tmp_db):
    guard = Guard(agent_name="bot", db_path=tmp_db)
    with guard.session() as session:
        session.call(lambda: _MockResponse())
    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session.record.session_id)
    assert record is not None
    assert record.status == "completed"
    assert len(record.turns) == 1


def test_session_trips_and_saves(tmp_db):
    guard = Guard(agent_name="bot", max_turns=1, db_path=tmp_db)
    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            session.call(lambda: _MockResponse())
            session.call(lambda: _MockResponse())

    from agentguard.storage import Storage

    sessions = Storage(tmp_db).list_sessions()
    assert sessions[0]["status"] == "tripped"


def test_error_turn_recorded(tmp_db):
    guard = Guard(agent_name="bot", db_path=tmp_db)

    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        with guard.session() as session:
            session.call(fail)

    from agentguard.storage import Storage

    sessions = Storage(tmp_db).list_sessions()
    record = Storage(tmp_db).get_session(sessions[0]["session_id"])
    assert record.turns[0].status == "error"
    assert record.status == "error"
