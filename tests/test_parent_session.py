from agentguard import Guard
from agentguard.models import SessionRecord
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


def test_parent_session_id_round_trip(tmp_db):
    parent = SessionRecord(session_id="parent001", agent_name="orchestrator")
    child = SessionRecord(
        session_id="child001",
        agent_name="worker",
        parent_session_id="parent001",
    )
    parent.finalize("completed")
    child.finalize("completed")

    storage = Storage(tmp_db)
    storage.save_session(parent)
    storage.save_session(child)

    loaded = storage.get_session("child001")
    assert loaded is not None
    assert loaded.parent_session_id == "parent001"


def test_list_sessions_filter_by_parent(tmp_db):
    storage = Storage(tmp_db)
    storage.save_session(SessionRecord(session_id="p1", agent_name="parent"))
    storage.save_session(
        SessionRecord(session_id="c1", agent_name="child-a", parent_session_id="p1")
    )
    storage.save_session(
        SessionRecord(session_id="c2", agent_name="child-b", parent_session_id="p1")
    )
    storage.save_session(SessionRecord(session_id="solo", agent_name="solo"))

    children = storage.list_sessions(parent_session_id="p1")
    assert {s["session_id"] for s in children} == {"c1", "c2"}

    by_prefix = storage.list_sessions(parent_session_id_prefix="p")
    assert {s["session_id"] for s in by_prefix} == {"c1", "c2"}


def test_list_child_sessions(tmp_db):
    storage = Storage(tmp_db)
    storage.save_session(SessionRecord(session_id="p1", agent_name="parent"))
    storage.save_session(
        SessionRecord(session_id="c1", agent_name="child", parent_session_id="p1")
    )

    children = storage.list_child_sessions("p1")
    assert len(children) == 1
    assert children[0]["session_id"] == "c1"


def test_get_session_ancestors(tmp_db):
    storage = Storage(tmp_db)
    for sid, agent, parent in [
        ("root", "root-agent", None),
        ("mid", "mid-agent", "root"),
        ("leaf", "leaf-agent", "mid"),
    ]:
        record = SessionRecord(session_id=sid, agent_name=agent, parent_session_id=parent)
        record.finalize("completed")
        storage.save_session(record)

    ancestors = storage.get_session_ancestors("leaf")
    assert [a["session_id"] for a in ancestors] == ["root", "mid"]
    assert [a["agent_name"] for a in ancestors] == ["root-agent", "mid-agent"]


def test_guard_session_links_parent(tmp_db):
    parent_guard = Guard(agent_name="orchestrator", db_path=tmp_db)
    child_guard = Guard(agent_name="worker", db_path=tmp_db)

    with parent_guard.session() as parent:
        parent.call(lambda: _MockResponse())
        parent_id = parent.record.session_id

        with child_guard.session(parent_session_id=parent_id) as child:
            child.call(lambda: _MockResponse())
            child_id = child.record.session_id

    storage = Storage(tmp_db)
    loaded_child = storage.get_session(child_id)
    assert loaded_child.parent_session_id == parent_id

    children = storage.list_child_sessions(parent_id)
    assert len(children) == 1
    assert children[0]["session_id"] == child_id


def test_nested_three_level_chain(tmp_db):
    guards = {
        "root": Guard(agent_name="root", db_path=tmp_db),
        "planner": Guard(agent_name="planner", db_path=tmp_db),
        "executor": Guard(agent_name="executor", db_path=tmp_db),
    }

    with guards["root"].session() as root:
        root_id = root.record.session_id
        with guards["planner"].session(parent_session_id=root_id) as planner:
            planner_id = planner.record.session_id
            with guards["executor"].session(parent_session_id=planner_id) as executor:
                executor_id = executor.record.session_id

    storage = Storage(tmp_db)
    ancestors = storage.get_session_ancestors(executor_id)
    assert [a["session_id"] for a in ancestors] == [root_id, planner_id]
