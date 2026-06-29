import pytest

from agentguard import Guard, CircuitBreakerTripped


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


class _MockDelta:
    def __init__(self, content: str | None = None):
        self.content = content


class _MockStreamChoice:
    def __init__(self, delta):
        self.delta = delta


class _MockStreamChunk:
    def __init__(self, content: str | None = None, model: str | None = None, usage=None):
        self.model = model
        self.usage = usage
        self.choices = [_MockStreamChoice(_MockDelta(content=content))] if content is not None else []


async def _mock_acall():
    return _MockResponse()


async def _mock_async_stream():
    yield _MockStreamChunk(content="Hel", model="gpt-4o")
    yield _MockStreamChunk(content="lo")
    yield _MockStreamChunk(usage=_MockUsage(), model="gpt-4o")


@pytest.mark.asyncio
async def test_acall_records_turn(tmp_db):
    guard = Guard(agent_name="bot", db_path=tmp_db)
    with guard.session() as session:
        response = await session.acall(_mock_acall)
        assert response.choices[0].message.content == "hello"
        session_id = session.record.session_id

    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session_id)
    assert record is not None
    assert len(record.turns) == 1
    assert record.turns[0].output["content"] == "hello"


@pytest.mark.asyncio
async def test_acall_awaitable_client_method(tmp_db):
    async def create(**_kwargs):
        return _MockResponse()

    guard = Guard(agent_name="bot", db_path=tmp_db)
    with guard.session() as session:
        response = await session.acall(create, model="gpt-4o", messages=[])
        assert response.choices[0].message.content == "hello"
        session_id = session.record.session_id

    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session_id)
    assert len(record.turns) == 1


@pytest.mark.asyncio
async def test_acall_trips_on_breaker(tmp_db):
    guard = Guard(agent_name="bot", max_turns=1, db_path=tmp_db)
    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            await session.acall(_mock_acall)
            await session.acall(_mock_acall)


@pytest.mark.asyncio
async def test_astream_records_turn(tmp_db):
    guard = Guard(agent_name="bot", db_path=tmp_db)
    with guard.session() as session:
        stream = await session.astream(_mock_async_stream)
        chunks = [chunk async for chunk in stream]
        session_id = session.record.session_id

    assert len(chunks) == 3
    from agentguard.storage import Storage

    record = Storage(tmp_db).get_session(session_id)
    assert record.turns[0].output["content"] == "Hello"


@pytest.mark.asyncio
async def test_astream_awaitable_stream_factory(tmp_db):
    async def create(**_kwargs):
        return _mock_async_stream()

    guard = Guard(agent_name="bot", db_path=tmp_db)
    with guard.session() as session:
        stream = await session.astream(create, stream=True)
        chunks = [chunk async for chunk in stream]

    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_astream_trips_after_completion(tmp_db):
    guard = Guard(agent_name="bot", max_turns=2, db_path=tmp_db)
    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            await session.acall(_mock_acall)
            stream = await session.astream(_mock_async_stream)
            async for _ in stream:
                pass
            await session.acall(_mock_acall)


@pytest.mark.asyncio
async def test_astream_rejects_sync_iterator():
    guard = Guard(agent_name="bot")

    def sync_stream():
        yield _MockStreamChunk(content="x", model="gpt-4o")

    with guard.session() as session:
        with pytest.raises(TypeError, match="async iterator"):
            await session.astream(sync_stream)
