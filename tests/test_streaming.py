import pytest

from agentguard import Guard, CircuitBreakerTripped
from agentguard.streaming import OpenAIStreamAccumulator


class _MockDelta:
    def __init__(self, content: str | None = None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _MockStreamChoice:
    def __init__(self, delta):
        self.delta = delta


class _MockStreamChunk:
    __module__ = "openai.types.chat"

    def __init__(self, content: str | None = None, model: str | None = None, usage=None):
        self.model = model
        self.usage = usage
        self.choices = [_MockStreamChoice(_MockDelta(content=content))] if content is not None else []


class _MockUsage:
    prompt_tokens = 12
    completion_tokens = 8


def _mock_stream():
    yield _MockStreamChunk(content="Hel", model="gpt-4o")
    yield _MockStreamChunk(content="lo")
    yield _MockStreamChunk(usage=_MockUsage(), model="gpt-4o")


def test_stream_accumulator_assembles_content():
    acc = OpenAIStreamAccumulator()
    for chunk in _mock_stream():
        acc.consume(chunk)

    response = acc.as_response()
    assert response.choices[0].message.content == "Hello"
    assert response.usage.prompt_tokens == 12
    assert response.model == "gpt-4o"


def test_session_stream_records_turn(tmp_db):
    guard = Guard(agent_name="bot", db_path=tmp_db)
    with guard.session() as session:
        stream = session.stream(lambda: _mock_stream())
        chunks = list(stream)

    assert len(chunks) == 3
    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session.record.session_id)
    assert record is not None
    assert len(record.turns) == 1
    assert record.turns[0].output["content"] == "Hello"
    assert record.turns[0].tokens_in == 12
    assert record.turns[0].tokens_out == 8


def test_session_stream_trips_after_completion(tmp_db):
    guard = Guard(agent_name="bot", max_turns=2, db_path=tmp_db)
    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            session.call(lambda: _make_response("first"))
            list(session.stream(lambda: _mock_stream()))

    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session.record.session_id)
    assert record.status == "tripped"
    assert len(record.turns) == 2


def test_session_stream_error_mid_iteration(tmp_db):
    def bad_stream():
        yield _MockStreamChunk(content="partial", model="gpt-4o")
        raise RuntimeError("stream failed")

    guard = Guard(agent_name="bot", db_path=tmp_db)
    with pytest.raises(RuntimeError, match="stream failed"):
        with guard.session() as session:
            stream = session.stream(bad_stream)
            list(stream)

    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session.record.session_id)
    assert record.turns[0].status == "error"


def test_session_stream_rejects_non_iterator():
    guard = Guard(agent_name="bot")
    with guard.session() as session:
        with pytest.raises(TypeError, match="iterator"):
            session.stream(lambda: {"not": "a stream"})


class _MockUsageStatic:
    prompt_tokens = 1
    completion_tokens = 1


class _MockMessageStatic:
    content = "ok"
    tool_calls = None


class _MockChoiceStatic:
    message = _MockMessageStatic()


class _MockResponseStatic:
    __module__ = "openai.types.chat"
    usage = _MockUsageStatic()
    model = "gpt-4o"
    choices = [_MockChoiceStatic()]


def _make_response(content: str):
    class _Msg:
        pass

    msg = _Msg()
    msg.content = content
    msg.tool_calls = None

    class _Choice:
        message = msg

    class _Resp:
        __module__ = "openai.types.chat"
        usage = _MockUsageStatic()
        model = "gpt-4o"
        choices = [_Choice()]

    return _Resp()
