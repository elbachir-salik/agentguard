from agentguard.extractors.anthropic import AnthropicExtractor
from agentguard.extractors.generic import GenericExtractor
from agentguard.extractors.openai import OpenAIExtractor


class _OpenAIUsage:
    prompt_tokens = 100
    completion_tokens = 50


class _OpenAIFn:
    name = "get_weather"
    arguments = '{"city": "Paris"}'


class _OpenAIToolCall:
    id = "call_1"
    function = _OpenAIFn()


class _OpenAIMessage:
    content = "Checking weather"
    tool_calls = [_OpenAIToolCall()]


class _OpenAIChoice:
    message = _OpenAIMessage()


class _OpenAIResponse:
    usage = _OpenAIUsage()
    model = "gpt-4o"
    choices = [_OpenAIChoice()]


class _AnthropicUsage:
    input_tokens = 80
    output_tokens = 40


class _AnthropicTextBlock:
    type = "text"
    text = "Hello"


class _AnthropicToolBlock:
    type = "tool_use"
    id = "toolu_1"
    name = "search"
    input = {"q": "test"}


class _AnthropicResponse:
    usage = _AnthropicUsage()
    model = "claude-sonnet-4-6"
    content = [_AnthropicTextBlock(), _AnthropicToolBlock()]


def test_openai_extractor():
    result = OpenAIExtractor().extract(_OpenAIResponse())
    assert result["tokens_in"] == 100
    assert result["tokens_out"] == 50
    assert result["model"] == "gpt-4o"
    assert result["content"] == "Checking weather"
    assert result["tool_calls"][0]["name"] == "get_weather"


def test_anthropic_extractor():
    result = AnthropicExtractor().extract(_AnthropicResponse())
    assert result["tokens_in"] == 80
    assert result["content"] == "Hello"
    assert result["tool_calls"][0]["name"] == "search"
    assert result["tool_calls"][0]["arguments"] == {"q": "test"}


def test_generic_extractor_dict():
    resp = {
        "model": "custom-model",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        "content": "ok",
    }
    result = GenericExtractor().extract(resp)
    assert result["tokens_in"] == 5
    assert result["content"] == "ok"
