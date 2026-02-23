"""Base protocol for AI providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aivis.runner import RunResult


@runtime_checkable
class AIProvider(Protocol):
    """Protocol that all AI providers must implement."""

    name: str

    def run(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> RunResult:
        """Execute a single prompt against the provider API."""
        ...

    def run_stub(self, prompt: str) -> RunResult:
        """Return a realistic stub response for testing."""
        ...

    def check_api_key(self) -> bool:
        """Return True if the API key is configured."""
        ...