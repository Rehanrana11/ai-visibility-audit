"""
Runner module — thin dispatcher to provider implementations.

Backward compatible: defaults to Anthropic provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    raw_text: str
    raw_json: dict | None
    request_payload: dict
    model_version_hint: str | None = None
    usage: dict = field(default_factory=dict)


def run_once(
    prompt: str,
    *,
    provider_name: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> RunResult:
    """
    Execute a single prompt via the named provider.

    Defaults to Anthropic for backward compatibility.
    """
    from aivis.providers.registry import get_provider

    provider = get_provider(provider_name)

    kwargs: dict = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model is not None:
        kwargs["model"] = model

    return provider.run(prompt, **kwargs)


def run_once_stub(
    prompt: str,
    *,
    provider_name: str = "anthropic",
) -> RunResult:
    """
    Return a stub response from the named provider.

    Defaults to Anthropic for backward compatibility.
    """
    from aivis.providers.registry import get_provider

    provider = get_provider(provider_name)
    return provider.run_stub(prompt)