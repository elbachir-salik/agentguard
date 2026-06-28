# AgentGuard

**The black box + circuit breaker for AI agents.**

Wrap any LLM call in 3 lines. Record every turn. Catch loops, budget overruns, and timeouts before they cause damage. Replay any session. Browse it all in a local dashboard. No cloud. No API keys. Just `pip install agentguard`.

```python
from agentguard import Guard

guard = Guard(agent_name="my-agent", max_cost=5.00, max_turns=20, max_tool_retries=3, timeout=60)

with guard.session() as session:
    response = session.call(client.chat.completions.create, model="gpt-4o", messages=messages)
```

That's it. Every call is recorded, costs are tracked, and if your agent loops or burns through budget — the circuit breaker stops it.

---

## Why

Agents in production fail silently. They don't crash — they loop, hallucinate, burn tokens, and go off-scope. By the time you notice, the damage is done.

Existing tools are either enterprise SaaS (Datadog, AgentOps — cloud, API keys, $$$) or narrowly focused (just a breaker, no recording). Nobody built a self-hosted SDK that combines **recording + protection + dashboard** in one `pip install`.

---

## Install

```bash
pip install agentguard
```

Or from source:

```bash
git clone https://github.com/salikelbachir/agentguard.git
cd agentguard
pip install -e .
```

---

## What You Get

### 1. The Black Box (Flight Recorder)

Every LLM call is recorded: input, output, tokens, cost, tool calls, latency. Stored locally in SQLite at `~/.agentguard/agentguard.db`.

### 2. The Circuit Breaker

Real-time protection with composable rules:

| Rule | What it catches | Config |
|------|----------------|--------|
| **Budget** | Cost or token limit exceeded | `max_cost=5.00`, `max_tokens=100000` |
| **Turns** | Too many LLM calls in one session | `max_turns=20` |
| **Loop Detection** | Same tool called repeatedly with similar input | `max_tool_retries=3` |
| **Timeout** | Session running too long | `timeout=60` |
| **Scope** | Unauthorized tool usage | `allowed_tools=[...]`, `blocked_tools=[...]` |

When a rule trips, AgentGuard raises `CircuitBreakerTripped` and saves the session with full context.

### 3. CLI

```bash
agentguard sessions              # list all recorded sessions
agentguard replay <session_id>   # replay a session turn by turn
agentguard stats                 # cost and usage summary
agentguard dashboard             # local web UI at localhost:8585
```

### 4. Local Dashboard

A dark-themed web UI to browse sessions, replay turns, and view cost/usage charts. FastAPI + Jinja2 + Chart.js. No React, no build step.

```bash
agentguard dashboard
# open http://localhost:8585
```

---

## Usage

### Basic

```python
from agentguard import Guard

guard = Guard(agent_name="support-bot", max_cost=1.00, max_turns=15)

with guard.session() as session:
    response = session.call(
        client.chat.completions.create,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Refund order #4521"}],
    )
    print(session.summary())
```

### Catching a Trip

```python
from agentguard import Guard, CircuitBreakerTripped

guard = Guard(agent_name="my-agent", max_cost=0.10, max_tool_retries=3)

try:
    with guard.session() as session:
        while True:
            response = session.call(client.chat.completions.create, ...)
            # ... handle tool calls, loop back ...
except CircuitBreakerTripped as e:
    print(f"Stopped: {e}")
    # Session is saved automatically with status="tripped"
```

### Custom Rules

```python
from agentguard.rules.base import BaseRule, RuleResult, SessionState

class NoPIIRule(BaseRule):
    name = "no_pii"

    def check(self, state: SessionState) -> RuleResult:
        if state.turns:
            content = str(state.turns[-1].output.get("content", ""))
            if "SSN" in content or "social security" in content.lower():
                return RuleResult.trip("PII detected in output")
        return RuleResult.passed()

guard = Guard(agent_name="safe-bot", rules=[NoPIIRule()])
```

---

## Terminal Output

**Successful session:**

```
+---------------- agentguard -- support-bot -- fdf2f69c9301 -----------------+
|   Turn 1  OK  510 tok  $0.0008  1200ms                                     |
|     -> user: "Refund order #4521"                                           |
|     <- tool_call: lookup_order({"order_id": "4521"})                        |
|                                                                             |
|   Turn 2  OK  380 tok  $0.0006  800ms                                      |
|     <- tool_call: process_refund({"order_id": "4521"})                      |
|                                                                             |
|   Turn 3  OK  290 tok  $0.0004  600ms                                      |
|     <- "Your refund of $49.99 has been processed."                          |
|                                                                             |
|   Completed -- 3 turns -- 1180 tok -- $0.0018                               |
+-----------------------------------------------------------------------------+
```

**Circuit breaker tripped:**

```
+------------ agentguard -- support-bot -- c699836f5882 ---------------------+
|   Turn 1  OK  150 tok  $0.0008  1100ms                                     |
|   Turn 2  OK  150 tok  $0.0008  900ms                                      |
|                                                                            |
|   == CIRCUIT BREAKER =========================                             |
|     Rule:    loop_detection                                                |
|     Cause:   search_kb called 3x with similar input                        |
|   ============================================                             |
|                                                                            |
|   Tripped -- 2 turns -- 300 tok -- $0.0015                                 |
+----------------------------------------------------------------------------+
```

---

## Supported Providers

AgentGuard has **zero LLM dependencies**. It wraps whatever client you already use:

- **OpenAI** — full extraction (tokens, cost, tool calls)
- **Anthropic** — full extraction (tokens, cost, tool calls)
- **Any provider** — generic extractor works with any dict-like or object response
- **NVIDIA NIM / DeepSeek** — works via OpenAI-compatible base_url

---

## Architecture

```
Your Agent Code
      |
      v
+-------------------------------------------+
|              Guard (wrapper)              |
|                                           |
|  +-------------+    +------------------+  |
|  |  Recorder   |    | Circuit Breaker  |  |
|  |  (black box)|    | (protection)     |  |
|  +------+------+    +--------+---------+  |
|         |                    |            |
|         v                    v            |
|  +-----------------------------------+   |
|  |       Storage (SQLite)             |   |
|  |  ~/.agentguard/agentguard.db       |   |
|  +----------------+------------------+   |
|                    |                      |
+--------------------+----------------------+
                     |
                     v
         +-----------------------+
         |   Dashboard (local)   |
         |   localhost:8585      |
         +-----------------------+
```

---

## License

MIT
