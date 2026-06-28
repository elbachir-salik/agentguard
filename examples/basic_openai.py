"""Basic example: wrap an OpenAI call with AgentGuard.

Replace the mock with a real OpenAI client to use with a live API:

    from openai import OpenAI
    client = OpenAI()

    with guard.session() as session:
        response = session.call(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )
"""

from agentguard import Guard


class _MockUsage:
    prompt_tokens = 25
    completion_tokens = 40


class _MockMessage:
    content = "Hello! How can I help you today?"
    tool_calls = None


class _MockChoice:
    message = _MockMessage()


class _MockResponse:
    __module__ = "openai.types.chat"
    usage = _MockUsage()
    model = "gpt-4o"
    choices = [_MockChoice()]


guard = Guard(agent_name="support-bot")

with guard.session() as session:
    response = session.call(lambda: _MockResponse())
    print("Response:", response.choices[0].message.content)
    print("Summary:", session.summary())
