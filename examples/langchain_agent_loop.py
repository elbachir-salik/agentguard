"""LangChain example: circuit breaker trips on max turns.

Install: pip install agentguard[langchain]
"""

from agentguard import Guard, CircuitBreakerTripped
from agentguard.integrations.langchain import AgentGuardCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel

guard = Guard(agent_name="langchain-loop", max_turns=3)

try:
    with guard.session() as session:
        handler = AgentGuardCallbackHandler(session)
        llm = FakeListChatModel(
            responses=["turn 1", "turn 2", "turn 3", "turn 4"],
            callbacks=[handler],
        )
        for i in range(4):
            print(f"Invoke {i + 1}:", llm.invoke(f"message {i + 1}").content)
except CircuitBreakerTripped as e:
    print(f"Stopped: {e}")
