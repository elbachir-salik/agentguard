"""Tests for FastAPI integration."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from agentguard import Guard
from agentguard.integrations.fastapi import (
    create_session_dependency,
    metadata_from_request,
    register_trip_handler,
    setup_agentguard,
)
from agentguard.storage import Storage


class _MockUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _MockMessage:
    content = "api response"
    tool_calls = None


class _MockChoice:
    message = _MockMessage()


class _MockResponse:
    __module__ = "openai.types.chat"
    usage = _MockUsage()
    model = "gpt-4o"
    choices = [_MockChoice()]


@pytest.fixture
def app_and_client(tmp_db):
    guard = Guard(agent_name="api-bot", max_turns=10, db_path=tmp_db)
    app = FastAPI()
    setup_agentguard(app, guard)
    SessionDep = create_session_dependency()

    @app.post("/chat")
    async def chat(session=Depends(SessionDep)):
        session.call(lambda: _MockResponse())
        return {"turns": session.summary()["turns"], "session_id": session.record.session_id}

    return app, TestClient(app), tmp_db


def test_session_dependency_records_turn(app_and_client):
    _, client, tmp_db = app_and_client
    resp = client.post("/chat")
    assert resp.status_code == 200
    data = resp.json()
    assert data["turns"] == 1

    stored = Storage(tmp_db).get_session(data["session_id"])
    assert stored is not None
    assert stored.metadata.get("path") == "/chat"
    assert stored.metadata.get("method") == "POST"


def test_trip_handler_returns_429(tmp_db):
    guard = Guard(agent_name="api-trip", max_turns=1, db_path=tmp_db)
    app = FastAPI()
    setup_agentguard(app, guard)
    SessionDep = create_session_dependency()

    @app.post("/loop")
    async def loop(session=Depends(SessionDep)):
        session.call(lambda: _MockResponse())
        session.call(lambda: _MockResponse())
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/loop")
    assert resp.status_code == 429
    body = resp.json()
    assert body["rule"] == "turns"
    assert "detail" in body


def test_register_trip_handler_standalone(tmp_db):
    guard = Guard(agent_name="standalone", max_turns=1, db_path=tmp_db)
    app = FastAPI()
    app.state.agentguard = guard
    register_trip_handler(app)
    SessionDep = create_session_dependency()

    @app.post("/trip")
    async def trip(session=Depends(SessionDep)):
        for _ in range(2):
            session.call(lambda: _MockResponse())
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/trip").status_code == 429


def test_metadata_from_request():
    from unittest.mock import MagicMock

    request = MagicMock()
    request.url.path = "/meta"
    request.method = "GET"
    request.client.host = "127.0.0.1"

    meta = metadata_from_request(request)
    assert meta["path"] == "/meta"
    assert meta["method"] == "GET"
    assert meta["client_host"] == "127.0.0.1"
