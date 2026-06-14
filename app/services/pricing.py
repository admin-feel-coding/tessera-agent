_PRICES_PER_M = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}
_DEFAULT = {"input": 3.00, "output": 15.00}


def cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    prices = _PRICES_PER_M.get(model, _DEFAULT)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
