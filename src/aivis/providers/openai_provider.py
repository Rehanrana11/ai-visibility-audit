"""OpenAI ChatGPT provider implementation."""

from __future__ import annotations

import time

import httpx

from aivis.runner import RunResult
from aivis.settings import settings

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
RETRYABLE = {429, 500, 502, 503}


class OpenAIProvider:
    """OpenAI ChatGPT API provider."""

    name: str = "openai"

    def check_api_key(self) -> bool:
        return settings.has_key("openai")

    def run(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RunResult:
        api_key = settings.get_key("openai")

        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_payload = {
            "url": OPENAI_URL,
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
                        OPENAI_URL,
                        json=request_body,
                        headers=headers,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = ""
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        raw_text = msg.get("content", "")

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
                    f"OpenAI API returned {resp.status_code}: {error_body}"
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
            f"OpenAI API failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    def run_stub(self, prompt: str) -> RunResult:
        fake = (
            "1. Monday.com - Intuitive visual project"
            " management with powerful automations (no citation)\n"
            "2. Asana - Comprehensive task and project"
            " tracking for teams of all sizes (no citation)\n"
            "3. ClickUp - Feature-rich all-in-one"
            " productivity and project management (no citation)\n"
            "4. Trello - Simple and flexible Kanban-style"
            " boards for organizing work (no citation)\n"
            "5. Notion - Versatile workspace combining"
            " docs, wikis, and project management (no citation)\n"
            "6. Jira - Industry-standard tool for agile"
            " software development teams (no citation)\n"
            "7. Smartsheet - Enterprise-grade work"
            " management with spreadsheet interface (no citation)\n"
            "8. Wrike - Collaborative work management"
            " with advanced reporting (no citation)\n"
            "9. Basecamp - Simplified project communication"
            " and task management (no citation)\n"
            "10. Linear - Fast issue tracking for"
            " modern software teams (no citation)"
        )
        return RunResult(
            raw_text=fake,
            raw_json=None,
            request_payload={
                "url": "STUB",
                "model": "stub-openai",
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