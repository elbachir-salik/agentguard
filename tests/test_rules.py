import time

from agentguard.models import Turn
from agentguard.rules.base import SessionState
from agentguard.rules.budget import BudgetRule
from agentguard.rules.loop import LoopRule
from agentguard.rules.scope import ScopeRule
from agentguard.rules.timeout import TimeoutRule
from agentguard.rules.turns import TurnsRule


def _turn(cost: float = 0.01, tokens_in: int = 10, tokens_out: int = 10, tool_calls=None) -> Turn:
    return Turn(
        turn_number=1,
        timestamp="2026-01-01T00:00:00+00:00",
        input_messages=[{"role": "user", "content": "hi"}],
        output={"content": "hello"},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        latency_ms=100.0,
        tool_calls=tool_calls or [],
        status="success",
        model="gpt-4o",
    )


def test_budget_rule_trips_on_cost():
    state = SessionState()
    state.add_turn(_turn(cost=0.06))
    result = BudgetRule(max_cost_usd=0.05).check(state)
    assert result.tripped is True


def test_budget_rule_passes_under_limit():
    state = SessionState()
    state.add_turn(_turn(cost=0.01))
    result = BudgetRule(max_cost_usd=0.05).check(state)
    assert result.tripped is False


def test_turns_rule_trips_at_limit():
    state = SessionState()
    state.turns = [_turn() for _ in range(2)]
    state.total_cost = 0.02
    state.total_tokens = 40
    result = TurnsRule(max_turns=2).check(state)
    assert result.tripped is True


def test_scope_empty_allowlist_blocks_all():
    state = SessionState()
    state.add_turn(_turn(tool_calls=[{"name": "dangerous_tool", "arguments": "{}"}]))
    result = ScopeRule(allowed_tools=[]).check(state)
    assert result.tripped is True


def test_scope_blocked_tool():
    state = SessionState()
    state.add_turn(_turn(tool_calls=[{"name": "delete_all", "arguments": "{}"}]))
    result = ScopeRule(blocked_tools=["delete_all"]).check(state)
    assert result.tripped is True


def test_timeout_rule_trips():
    state = SessionState()
    state.start_time = time.time() - 61
    result = TimeoutRule(timeout_seconds=60).check(state)
    assert result.tripped is True


def test_loop_rule_trips_on_repeated_similar_calls():
    state = SessionState()
    for i in range(3):
        state.add_turn(_turn(tool_calls=[{
            "id": f"call_{i}",
            "name": "search_kb",
            "arguments": '{"query": "refund"}',
        }]))
    result = LoopRule(max_retries=3).check(state)
    assert result.tripped is True
