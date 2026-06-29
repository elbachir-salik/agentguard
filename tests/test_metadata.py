from agentguard import Guard
from agentguard.models import SessionRecord
from agentguard.storage import Storage


def test_metadata_save_and_load(tmp_db):
    record = SessionRecord(
        session_id="meta001",
        agent_name="support-bot",
        metadata={"customer_id": "4521", "env": "staging", "ticket": "1234"},
    )
    record.finalize("completed")

    storage = Storage(tmp_db)
    storage.save_session(record)

    loaded = storage.get_session("meta001")
    assert loaded is not None
    assert loaded.metadata == {
        "customer_id": "4521",
        "env": "staging",
        "ticket": "1234",
    }


def test_metadata_in_list_sessions(tmp_db):
    record = SessionRecord(
        session_id="meta002",
        agent_name="support-bot",
        metadata={"env": "production"},
    )
    record.finalize("completed")
    Storage(tmp_db).save_session(record)

    sessions = Storage(tmp_db).list_sessions()
    assert sessions[0]["metadata"] == {"env": "production"}


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


def test_guard_session_metadata(tmp_db):
    guard = Guard(agent_name="support-bot", db_path=tmp_db)
    metadata = {"customer_id": "99", "env": "staging"}

    with guard.session(metadata=metadata) as session:
        session.call(lambda: _MockResponse())
        summary = session.summary()
        assert summary["metadata"] == metadata

    loaded = Storage(tmp_db).get_session(session.record.session_id)
    assert loaded is not None
    assert loaded.metadata == metadata
