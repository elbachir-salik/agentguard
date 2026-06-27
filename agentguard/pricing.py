PRICING: dict[str, dict[str, float]] = {
    # OpenAI - per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic - per 1M tokens
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
}

_FALLBACK = {"input": 1.00, "output": 3.00}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    prices = PRICING.get(model, _FALLBACK)
    return (tokens_in * prices["input"] + tokens_out * prices["output"]) / 1_000_000
