"""Provider registry — maps provider names to implementations."""
from __future__ import annotations

from aivis.providers.anthropic_provider import AnthropicProvider
from aivis.providers.openai_provider import OpenAIProvider
from aivis.providers.google_provider import GoogleProvider
from aivis.providers.grok_provider import GrokProvider

PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
    "grok": GrokProvider,
}

PROVIDER_NAMES = list(PROVIDERS.keys())


def get_provider(name: str):
    """Return an instance of the named provider."""
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name}. "
            f"Available: {PROVIDER_NAMES}"
        )
    return PROVIDERS[name]()


def get_stub_runner(name: str):
    """Return the run_stub method for the named provider."""
    return get_provider(name).run_stub


def get_live_runner(name: str):
    """Return the run method for the named provider."""
    return get_provider(name).run