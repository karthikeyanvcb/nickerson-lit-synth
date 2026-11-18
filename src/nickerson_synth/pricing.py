"""Token-cost estimation for ``--dry-run`` and for reporting real runs.

Prices are USD per million tokens, taken from Anthropic's public pricing page
(https://platform.claude.com/docs/en/about-claude/pricing) in November 2026.
Prices change: pass ``--price-input`` / ``--price-output`` (and optionally the
cache prices) on the CLI to override, or edit ``MODEL_PRICES``.

Token counts in a dry run are *estimates* (about 4 characters per token for
English prose); real runs report the exact usage returned by the API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nickerson_synth.models import TokenUsage

CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_write: float  # 5-minute ephemeral cache write
    cache_read: float
    min_cacheable_tokens: int = 1024


MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(4.0, 20.0, 5.0, 0.20, 512),
    "claude-opus-5": ModelPrice(5.0, 25.0, 6.25, 0.50, 512),
    "claude-sonnet-5-5": ModelPrice(2.0, 10.0, 2.50, 0.10, 512),
    "claude-sonnet-5": ModelPrice(2.0, 10.0, 2.50, 0.20, 1024),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 3.75, 0.30, 1024),
    # Older Sonnet models. Anthropic lists these as retired on its own API, so
    # requests may fail; check model availability for your account first.
    "claude-sonnet-4-20250514": ModelPrice(3.0, 15.0, 3.75, 0.30, 1024),
    "claude-3-7-sonnet-latest": ModelPrice(3.0, 15.0, 3.75, 0.30, 1024),
    # Haiku 5.5 is priced higher above 100k-token prompts; this row is the <=100k tier.
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.125, 0.01, 512),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 1.25, 0.10, 4096),
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"


def get_price(model: str, overrides: dict[str, float | None] | None = None) -> ModelPrice | None:
    """Return the price for ``model`` with any non-``None`` overrides applied.

    Dated model ids (``claude-haiku-4-5-20251001``) fall back to their alias.
    If the model is unknown, input and output overrides are required; cache
    prices then default to 1.25x / 0.1x input.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    base = MODEL_PRICES.get(model)
    if base is None:
        base = next((p for k, p in MODEL_PRICES.items() if model.startswith(k + "-")), None)
    if base is None:
        if "input" not in overrides or "output" not in overrides:
            return None
        base = ModelPrice(
            overrides["input"],
            overrides["output"],
            overrides["input"] * 1.25,
            overrides["input"] * 0.10,
        )
    fields = {
        "input": base.input,
        "output": base.output,
        "cache_write": base.cache_write,
        "cache_read": base.cache_read,
        "min_cacheable_tokens": base.min_cacheable_tokens,
    }
    fields.update(overrides)
    return ModelPrice(**fields)


def cost_usd(usage: TokenUsage, price: ModelPrice) -> float:
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_creation_input_tokens * price.cache_write
        + usage.cache_read_input_tokens * price.cache_read
    ) / 1_000_000


def estimate_tokens(text_or_chars: str | int) -> int:
    n = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars)
    return math.ceil(n / CHARS_PER_TOKEN)


@dataclass
class RunEstimate:
    n_papers: int
    prefix_tokens: int
    document_tokens: int
    output_tokens: int
    usage: TokenUsage
    cost_usd: float | None
    cache_used: bool


def estimate_run(
    prefix_tokens: int,
    document_tokens_per_paper: list[int],
    output_tokens_per_paper: int,
    price: ModelPrice | None,
) -> RunEstimate:
    """Estimate a run where a shared prefix (tools + system rubric) is cached.

    The first request writes the prefix to the cache; later requests read it.
    If the prefix is below the model's minimum cacheable length, it is billed
    as ordinary input on every request.
    """
    n = len(document_tokens_per_paper)
    min_cache = price.min_cacheable_tokens if price else 1024
    cache_used = prefix_tokens >= min_cache and n > 0
    docs = sum(document_tokens_per_paper)
    if cache_used:
        usage = TokenUsage(
            input_tokens=docs,
            output_tokens=output_tokens_per_paper * n,
            cache_creation_input_tokens=prefix_tokens,
            cache_read_input_tokens=prefix_tokens * (n - 1),
        )
    else:
        usage = TokenUsage(
            input_tokens=docs + prefix_tokens * n,
            output_tokens=output_tokens_per_paper * n,
        )
    return RunEstimate(
        n_papers=n,
        prefix_tokens=prefix_tokens,
        document_tokens=docs,
        output_tokens=output_tokens_per_paper * n,
        usage=usage,
        cost_usd=cost_usd(usage, price) if price else None,
        cache_used=cache_used,
    )
