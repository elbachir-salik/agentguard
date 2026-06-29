"""LangChain integration example (no API key required).

Install: pip install agentguard[langchain]
"""

from agentguard import Guard
from agentguard.integrations.langchain import guard_session
from langchain_core.language_models.fake_chat_models import FakeListChatModel

guard = Guard(agent_name="langchain-demo", max_turns=10)

with guard_session(guard, metadata={"env": "example"}) as (session, callbacks):
    llm = FakeListChatModel(responses=["Hello from LangChain!"], callbacks=callbacks)
    result = llm.invoke("Hi there")
    print("Response:", result.content)
    print("Summary:", session.summary())
