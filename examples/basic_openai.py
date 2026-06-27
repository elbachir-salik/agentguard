"""Basic example: wrap an OpenAI call with AgentGuard."""

from agentguard import Guard

# from openai import OpenAI
# client = OpenAI()

guard = Guard(agent_name="support-bot")

with guard.session() as session:
    # Replace with a real OpenAI call:
    # response = session.call(
    #     client.chat.completions.create,
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": "Hello!"}],
    # )

    # For testing without an API key, use a mock:
    class MockUsage:
        prompt_tokens = 25
        completion_tokens = 40

    class MockMessage:
        content = "Hello! How can I help you today?"
        tool_calls = None

    class MockChoice:
        message = MockMessage()

    class MockResponse:
        __module__ = "openai.types.chat"
        usage = MockUsage()
        model = "gpt-4o"
        choices = [MockChoice()]

    response = session.call(lambda: MockResponse())

    print("Response:", MockResponse.choices[0].message.content)
    print("Summary:", session.summary())
