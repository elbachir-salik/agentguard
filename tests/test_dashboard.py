"""Integration tests for the FastAPI dashboard endpoints."""

import pytest

from agentguard.models import SessionRecord, Turn
from agentguard.storage import Storage


@pytest.fixture
def client(tmp_db):
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    with patch("agentguard.dashboard.app.storage", Storage(tmp_db)):
        from agentguard.dashboard.app import app

        yield TestClient(app)


def _seed(tmp_db: str) -> str:
    storage = Storage(tmp_db)
    record = SessionRecord(session_id="dash001", agent_name="dash-bot")
    record.turns.append(
        Turn(
            turn_number=1,
            timestamp="2026-06-29T12:00:00+00:00",
            input_messages=[{"role": "user", "content": "hello"}],
            output={"content": "hi"},
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.001,
            latency_ms=100.0,
            tool_calls=[],
            status="success",
            model="gpt-4o",
        )
    )
    record.metadata = {"env": "staging"}
    record.finalize("completed")
    storage.save_session(record)
    return record.session_id


def test_index_page(client, tmp_db):
    _seed(tmp_db)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "dash001" in resp.text
    assert "dash-bot" in resp.text


def test_index_filter_by_agent(client, tmp_db):
    _seed(tmp_db)
    resp = client.get("/?agent=dash-bot")
    assert resp.status_code == 200
    assert "dash001" in resp.text

    resp = client.get("/?agent=other")
    assert resp.status_code == 200
    assert "dash001" not in resp.text


def test_session_detail_page(client, tmp_db):
    _seed(tmp_db)
    resp = client.get("/session/dash001")
    assert resp.status_code == 200
    assert "Turn 1" in resp.text
    assert "dash-bot" in resp.text


def test_session_not_found(client, tmp_db):
    resp = client.get("/session/nonexistent")
    assert resp.status_code == 404


def test_stats_page(client, tmp_db):
    _seed(tmp_db)
    resp = client.get("/stats")
    assert resp.status_code == 200
