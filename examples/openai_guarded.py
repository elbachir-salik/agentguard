"""OpenAI SDK wrapper example (no API key — uses a mock client).

With a real client:

    from openai import OpenAI
    from agentguard import Guard
    from agentguard.integrations.openai import guard_openai

    guard = Guard(agent_name="my-agent", max_cost=1.00, max_turns=15)
    client = OpenAI()

    with guard_openai(guard, client) as (session, guarded):
        response = guarded.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.choices[0].message.content)
        print(session.summary())
"""

from agentguard import Guard
from agentguard.integrations.openai import guard_openai


class _MockUsage:
    prompt_tokens = 25
    completion_tokens = 10


class _MockMessage:
    content = "Hello! How can I help?"
    tool_calls = None


class _MockChoice:
    message = _MockMessage()


class _MockResponse:
    __module__ = "openai.types.chat"
    usage = _MockUsage()
    model = "gpt-4o"
    choices = [_MockChoice()]


class _MockClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return _MockResponse()


guard = Guard(agent_name="openai-demo", max_turns=10)

with guard_openai(guard) as (session, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print("Response:", response.choices[0].message.content)
    print("Summary:", session.summary())
