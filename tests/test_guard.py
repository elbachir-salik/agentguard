import pytest

from agentguard import Guard
from agentguard.exceptions import CircuitBreakerTripped
from agentguard.rules.base import BaseRule, RuleResult, SessionState


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


def _mock_call():
    return _MockResponse()


def test_guard_convenience_kwargs():
    guard = Guard(agent_name="test", max_turns=5)
    assert len(guard._rules) == 1


def test_guard_rules_and_kwargs_conflict():
    class DummyRule(BaseRule):
        name = "dummy"

        def check(self, state: SessionState) -> RuleResult:
            return RuleResult.passed()

    with pytest.raises(ValueError, match="Cannot pass both"):
        Guard(agent_name="test", rules=[DummyRule()], max_cost=1.0)


def test_session_trips_on_max_turns(tmp_db):
    guard = Guard(agent_name="test", max_turns=2, db_path=tmp_db)
    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            session.call(_mock_call)
            session.call(_mock_call)
            session.call(_mock_call)


def test_summary_does_not_finalize(tmp_db):
    guard = Guard(agent_name="test", db_path=tmp_db)
    with guard.session() as session:
        session.call(_mock_call)
        summary = session.summary()
        assert summary["status"] == "running"
        assert summary["turns"] == 1
        session.call(_mock_call)
        assert session.record.status == "running"


def test_error_turns_count_toward_max_turns(tmp_db):
    guard = Guard(agent_name="test", max_turns=2, db_path=tmp_db)

    def fail():
        raise RuntimeError("api error")

    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            with pytest.raises(RuntimeError):
                session.call(fail)
            with pytest.raises(RuntimeError):
                session.call(fail)
            session.call(_mock_call)
