import json

from click.testing import CliRunner

from agentguard.cli import main
from agentguard.export import session_to_dict, session_to_json
from agentguard.models import BreakerEvent, SessionRecord, Turn
from agentguard.storage import Storage


def _sample_record(session_id: str = "export001") -> SessionRecord:
    record = SessionRecord(
        session_id=session_id,
        agent_name="support-bot",
        metadata={"env": "staging", "ticket": "99"},
    )
    record.turns.append(
        Turn(
            turn_number=1,
            timestamp="2026-01-01T00:00:00+00:00",
            input_messages=[{"role": "user", "content": "hello"}],
            output={"content": "hi"},
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.001,
            latency_ms=120.0,
            tool_calls=[],
            status="success",
            model="gpt-4o",
        )
    )
    record.breaker_event = BreakerEvent(rule="turns", trigger="limit", turn=1)
    record.finalize("tripped")
    return record


def test_session_to_json_round_trip(tmp_db):
    record = _sample_record()
    Storage(tmp_db).save_session(record)

    loaded = Storage(tmp_db).get_session("export001")
    assert loaded is not None

    data = session_to_dict(loaded)
    assert data["session_id"] == "export001"
    assert data["metadata"]["ticket"] == "99"
    assert data["turns"][0]["output"]["content"] == "hi"
    assert data["breaker_event"]["rule"] == "turns"

    parsed = json.loads(session_to_json(loaded))
    assert parsed["status"] == "tripped"


def test_export_cli_stdout(tmp_db, monkeypatch):
    record = _sample_record("cliexport01")
    storage = Storage(tmp_db)
    storage.save_session(record)

    monkeypatch.setattr("agentguard.cli.Storage", lambda db_path=None: Storage(tmp_db))

    runner = CliRunner()
    result = runner.invoke(main, ["export", "cliexport01"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "cliexport01"
    assert payload["agent_name"] == "support-bot"


def test_export_cli_to_file(tmp_db, monkeypatch, tmp_path):
    record = _sample_record("cliexport02")
    storage = Storage(tmp_db)
    storage.save_session(record)

    monkeypatch.setattr("agentguard.cli.Storage", lambda db_path=None: Storage(tmp_db))

    out_file = tmp_path / "session.json"
    runner = CliRunner()
    result = runner.invoke(main, ["export", "cliexport02", "-o", str(out_file)])

    assert result.exit_code == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["session_id"] == "cliexport02"
