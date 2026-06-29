"""Tests for LangChain integration (requires langchain-core)."""

from __future__ import annotations

import pytest

from agentguard import Guard
from agentguard.integrations.langchain import (
    AgentGuardCallbackHandler,
    guard_session,
    is_available,
)
from agentguard.storage import Storage

pytestmark = pytest.mark.skipif(not is_available(), reason="langchain-core not installed")


def test_callback_records_turns(tmp_db):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    guard = Guard(agent_name="lc-bot", max_turns=10, db_path=tmp_db)

    with guard.session() as session:
        handler = AgentGuardCallbackHandler(session)
        llm = FakeListChatModel(responses=["hello", "world"], callbacks=[handler])
        llm.invoke("hi")
        llm.invoke("again")

    assert session.summary()["turns"] == 2


def test_guard_session_helper(tmp_db):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    guard = Guard(agent_name="lc-helper", max_turns=10, db_path=tmp_db)

    with guard_session(guard, metadata={"env": "test"}) as (session, callbacks):
        llm = FakeListChatModel(responses=["ok"], callbacks=callbacks)
        llm.invoke("ping")

    assert session.summary()["turns"] == 1
    stored = Storage(tmp_db).get_session(session.record.session_id)
    assert stored is not None
    assert stored.metadata.get("env") == "test"


def test_callback_trips_on_max_turns(tmp_db):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    guard = Guard(agent_name="lc-trip", max_turns=2, db_path=tmp_db)

    with guard.session() as session:
        handler = AgentGuardCallbackHandler(session)
        llm = FakeListChatModel(responses=["a", "b", "c"], callbacks=[handler])
        llm.invoke("1")
        llm.invoke("2")
        llm.invoke("3")  # LangChain swallows breaker errors in callbacks

    assert session.record.status == "tripped"
    assert session.summary()["turns"] == 2
