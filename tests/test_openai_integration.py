"""Tests for OpenAI SDK integration (uses mocks — no API key required)."""

from __future__ import annotations

import pytest

from agentguard import Guard, CircuitBreakerTripped
from agentguard.integrations.openai import (
    GuardedOpenAIClient,
    guard_openai,
    guarded_client,
)
from agentguard.storage import Storage


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


def _mock_sync_client():
    class _Completions:
        def create(self, **kwargs):
            return _MockResponse()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def _mock_async_client():
    class _Completions:
        async def create(self, **kwargs):
            return _MockResponse()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def test_guarded_client_records_turn(tmp_db):
    guard = Guard(agent_name="openai-bot", max_turns=10, db_path=tmp_db)
    raw = _mock_sync_client()

    with guard.session() as session:
        client = guarded_client(raw, session)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response.choices[0].message.content == "hello"
    assert session.summary()["turns"] == 1


def test_guard_openai_helper(tmp_db):
    guard = Guard(agent_name="openai-helper", max_turns=10, db_path=tmp_db)
    raw = _mock_sync_client()

    with guard_openai(guard, raw, metadata={"env": "test"}) as (session, client):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

    stored = Storage(tmp_db).get_session(session.record.session_id)
    assert stored is not None
    assert stored.metadata.get("env") == "test"
    assert len(stored.turns) == 1


def test_guarded_client_trips_on_max_turns(tmp_db):
    guard = Guard(agent_name="openai-trip", max_turns=2, db_path=tmp_db)
    raw = _mock_sync_client()

    with pytest.raises(CircuitBreakerTripped):
        with guard.session() as session:
            client = guarded_client(raw, session)
            for _ in range(3):
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )


def test_guarded_client_stream(tmp_db):
    class _MockDelta:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None

    class _MockStreamChoice:
        def __init__(self, delta):
            self.delta = delta

    class _MockStreamChunk:
        __module__ = "openai.types.chat"

        def __init__(self, content=None, model=None, usage=None):
            self.model = model
            self.usage = usage
            self.choices = [_MockStreamChoice(_MockDelta(content))] if content else []

    def _mock_stream():
        yield _MockStreamChunk(content="Hel", model="gpt-4o")
        yield _MockStreamChunk(content="lo")
        yield _MockStreamChunk(usage=_MockUsage(), model="gpt-4o")

    guard = Guard(agent_name="openai-stream", max_turns=10, db_path=tmp_db)

    class _Completions:
        def create(self, **kwargs):
            return _mock_stream()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    with guard.session() as session:
        client = guarded_client(_Client(), session)
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        chunks = list(stream)

    assert len(chunks) == 3
    assert session.summary()["turns"] == 1


@pytest.mark.asyncio
async def test_guarded_async_client(tmp_db):
    guard = Guard(agent_name="openai-async", max_turns=10, db_path=tmp_db)
    raw = _mock_async_client()

    with guard.session() as session:
        client = guarded_client(raw, session)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response.choices[0].message.content == "hello"
    assert session.summary()["turns"] == 1


def test_other_client_attrs_pass_through():
    raw = _mock_sync_client()
    raw.models = "models-resource"

    guard = Guard(agent_name="pass-through")
    with guard.session() as session:
        client = guarded_client(raw, session)
        assert client.models == "models-resource"
        assert isinstance(client, GuardedOpenAIClient)
