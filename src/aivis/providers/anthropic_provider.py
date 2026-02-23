"""Anthropic Claude provider implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from aivis.runner import RunResult
from aivis.settings import settings

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-20250514"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
RETRYABLE = {429, 500, 502, 503, 529}


class AnthropicProvider:
    """Anthropic Claude API provider."""

    name: str = "anthropic"

    def check_api_key(self) -> bool:
        return settings.has_key("anthropic")

    def run(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RunResult:
        api_key = settings.get_key("anthropic")

        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        request_payload = {
            "url": ANTHROPIC_URL,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "prompt_text": prompt,
        }

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=TIMEOUT) as client:
                    resp = client.post(
                        ANTHROPIC_URL,
                        json=request_body,
                        headers=headers,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = ""
                    for block in data.get("content", []):
                        if block.get("type") == "text":
                            raw_text += block.get("text", "")

                    return RunResult(
                        raw_text=raw_text,
                        raw_json=data,
                        request_payload=request_payload,
                        model_version_hint=data.get("model"),
                        usage=data.get("usage", {}),
                    )

                if resp.status_code in RETRYABLE:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    _backoff(attempt)
                    continue

                error_body = ""
                try:
                    error_body = resp.text
                except Exception:
                    pass
                raise RuntimeError(
                    f"Anthropic API returned {resp.status_code}: {error_body}"
                )

            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
            ) as e:
                last_error = e
                _backoff(attempt)
                continue

        raise RuntimeError(
            f"Anthropic API failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    def run_stub(self, prompt: str) -> RunResult:
        fake = (
            "1. Asana - Excellent task management with timeline views"
            " and team collaboration features (no citation)\n"
            "2. Monday.com - Visual project tracking with"
            " automations (no citation)\n"
            "3. Trello - Simple Kanban boards for small"
            " teams (no citation)\n"
            "4. Jira - Best for agile software development"
            " teams (no citation)\n"
            "5. ClickUp - All-in-one productivity platform"
            " with docs and goals (no citation)\n"
            "6. Notion - Flexible workspace combining notes"
            " and project tracking (no citation)\n"
            "7. Smartsheet - Spreadsheet-style project"
            " management for enterprises (no citation)\n"
            "8. Wrike - Robust work management with"
            " real-time collaboration (no citation)\n"
            "9. Basecamp - Straightforward project"
            " communication tool (no citation)\n"
            "10. Linear - Fast issue tracking designed"
            " for software teams (no citation)\n"
        )
        return RunResult(
            raw_text=fake,
            raw_json=None,
            request_payload={
                "url": "STUB",
                "model": "stub-anthropic",
                "temperature": 0.0,
                "max_tokens": 2048,
                "prompt_length": len(prompt),
                "prompt_text": prompt,
            },
            model_version_hint=None,
            usage={},
        )


def _backoff(attempt: int) -> None:
    time.sleep(RETRY_BACKOFF_BASE ** attempt)