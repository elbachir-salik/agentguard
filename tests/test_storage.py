from agentguard.models import BreakerEvent, SessionRecord, Turn
from agentguard.storage import Storage


def _sample_record(session_id: str = "abc123") -> SessionRecord:
    record = SessionRecord(session_id=session_id, agent_name="test-bot")
    record.turns.append(
        Turn(
            turn_number=1,
            timestamp="2026-01-01T00:00:00+00:00",
            input_messages=[{"role": "user", "content": "hi"}],
            output={"content": "hello"},
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.001,
            latency_ms=100.0,
            tool_calls=[],
            status="success",
            model="gpt-4o",
        )
    )
    record.finalize("completed")
    return record


def test_save_and_load_round_trip(tmp_db):
    storage = Storage(tmp_db)
    record = _sample_record()
    storage.save_session(record)

    loaded = storage.get_session("abc123")
    assert loaded is not None
    assert loaded.agent_name == "test-bot"
    assert len(loaded.turns) == 1
    assert loaded.total_tokens == 15


def test_list_sessions_with_filters(tmp_db):
    storage = Storage(tmp_db)
    storage.save_session(_sample_record("aaa111"))
    tripped = _sample_record("bbb222")
    tripped.status = "tripped"
    tripped.breaker_event = BreakerEvent(rule="turns", trigger="limit", turn=1)
    storage.save_session(tripped)

    all_sessions = storage.list_sessions()
    assert len(all_sessions) == 2

    filtered = storage.list_sessions(status="tripped")
    assert len(filtered) == 1
    assert filtered[0]["session_id"] == "bbb222"


def test_find_sessions_by_prefix(tmp_db):
    storage = Storage(tmp_db)
    storage.save_session(_sample_record("abc123def456"))
    storage.save_session(_sample_record("abc789"))

    matches = storage.find_sessions_by_prefix("abc")
    assert len(matches) == 2

    single = storage.find_sessions_by_prefix("abc123")
    assert len(single) == 1


def test_get_stats(tmp_db):
    storage = Storage(tmp_db)
    storage.save_session(_sample_record("s1"))
    stats = storage.get_stats()
    assert stats["total_sessions"] == 1
    assert stats["completed"] == 1
    assert stats["total_tokens"] == 15


def test_chart_aggregates(tmp_db):
    storage = Storage(tmp_db)
    storage.save_session(_sample_record("s1"))
    cost_by_agent = storage.get_cost_by_agent()
    labels, values = storage.get_daily_costs()
    assert "test-bot" in cost_by_agent
    assert len(labels) == 1
    assert values[0] == 0.001
