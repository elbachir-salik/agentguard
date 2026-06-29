"""Integration tests for the CLI commands."""

from unittest.mock import patch

from click.testing import CliRunner

from agentguard.cli import main
from agentguard.models import SessionRecord, Turn
from agentguard.storage import Storage


def _seed_db(db_path: str) -> str:
    """Insert a test session and return its session_id."""
    storage = Storage(db_path)
    record = SessionRecord(session_id="cli_test_001", agent_name="test-bot")
    record.turns.append(
        Turn(
            turn_number=1,
            timestamp="2026-06-29T12:00:00+00:00",
            input_messages=[{"role": "user", "content": "hello"}],
            output={"content": "hi there"},
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.001,
            latency_ms=120.0,
            tool_calls=[],
            status="success",
            model="gpt-4o",
        )
    )
    record.metadata = {"env": "test", "ticket": "42"}
    record.finalize("completed")
    storage.save_session(record)
    return record.session_id


def test_sessions_command(tmp_db):
    _seed_db(tmp_db)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["sessions"])

    assert result.exit_code == 0
    assert "cli_test_001" in result.output


def test_sessions_filter_by_agent(tmp_db):
    _seed_db(tmp_db)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["sessions", "--agent", "nonexistent"])

    assert result.exit_code == 0
    assert "cli_test_001" not in result.output


def test_replay_command(tmp_db):
    _seed_db(tmp_db)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["replay", "cli_test_001"])

    assert result.exit_code == 0
    assert "Turn 1" in result.output
    assert "test-bot" in result.output


def test_replay_not_found(tmp_db):
    _seed_db(tmp_db)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["replay", "nonexistent"])

    assert result.exit_code == 0
    assert "not found" in result.output.lower()


def test_export_command(tmp_db):
    _seed_db(tmp_db)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["export", "cli_test_001"])

    assert result.exit_code == 0
    assert '"session_id": "cli_test_001"' in result.output
    assert '"agent_name": "test-bot"' in result.output


def test_stats_command(tmp_db):
    _seed_db(tmp_db)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["stats"])

    assert result.exit_code == 0
    assert "1" in result.output  # total sessions


def test_sessions_with_meta_filter(tmp_db):
    _seed_db(tmp_db)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["sessions", "--meta", "env=test"])

    assert result.exit_code == 0
    assert "cli_test_001" in result.output


def test_sessions_with_parent_filter(tmp_db):
    storage = Storage(tmp_db)
    storage.save_session(SessionRecord(session_id="parent01", agent_name="orchestrator"))
    child = SessionRecord(
        session_id="child01", agent_name="worker", parent_session_id="parent01"
    )
    child.finalize("completed")
    storage.save_session(child)

    runner = CliRunner()
    with patch("agentguard.cli.Storage", return_value=Storage(tmp_db)):
        result = runner.invoke(main, ["sessions", "--parent", "parent01"])

    assert result.exit_code == 0
    assert "child01" in result.output
