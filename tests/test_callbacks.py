from agentguard import Guard, CircuitBreakerTripped
from agentguard.models import BreakerEvent, SessionRecord, Turn


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


def test_on_turn_callback(tmp_db):
    turns_seen: list[Turn] = []

    def on_turn(turn: Turn, record: SessionRecord) -> None:
        turns_seen.append(turn)

    guard = Guard(agent_name="test", db_path=tmp_db, on_turn=on_turn)
    with guard.session() as session:
        session.call(lambda: _MockResponse())

    assert len(turns_seen) == 1
    assert turns_seen[0].turn_number == 1


def test_on_trip_callback(tmp_db):
    trips_seen: list[BreakerEvent] = []

    def on_trip(event: BreakerEvent, record: SessionRecord) -> None:
        trips_seen.append(event)

    guard = Guard(agent_name="test", max_turns=1, db_path=tmp_db, on_trip=on_trip)

    # max_turns=1: first call succeeds, second trips
    try:
        with guard.session() as session:
            session.call(lambda: _MockResponse())
            session.call(lambda: _MockResponse())
    except CircuitBreakerTripped:
        pass

    assert len(trips_seen) == 1
    assert trips_seen[0].rule == "turns"
