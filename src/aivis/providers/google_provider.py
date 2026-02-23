"""Google Gemini provider implementation."""

from __future__ import annotations

import time

import httpx

from aivis.runner import RunResult
from aivis.settings import settings

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
RETRYABLE = {429, 500, 503}


class GoogleProvider:
    """Google Gemini API provider."""

    name: str = "google"

    def check_api_key(self) -> bool:
        return settings.has_key("google")

    def run(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RunResult:
        api_key = settings.get_key("google")

        url = f"{GEMINI_BASE}/models/{model}:generateContent"

        request_body = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        request_payload = {
            "url": url,
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
                        url,
                        json=request_body,
                        headers=headers,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = ""
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            raw_text = parts[0].get("text", "")

                    usage_meta = data.get("usageMetadata", {})
                    usage = {
                        "prompt_tokens": usage_meta.get(
                            "promptTokenCount", 0
                        ),
                        "completion_tokens": usage_meta.get(
                            "candidatesTokenCount", 0
                        ),
                        "total_tokens": usage_meta.get(
                            "totalTokenCount", 0
                        ),
                    }

                    return RunResult(
                        raw_text=raw_text,
                        raw_json=data,
                        request_payload=request_payload,
                        model_version_hint=data.get("modelVersion"),
                        usage=usage,
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
                    f"Gemini API returned {resp.status_code}: {error_body}"
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
            f"Gemini API failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    def run_stub(self, prompt: str) -> RunResult:
        fake = (
            "1. ClickUp - All-in-one project management"
            " with docs, goals, and time tracking (no citation)\n"
            "2. Asana - Popular task management platform"
            " with timeline and portfolio views (no citation)\n"
            "3. Monday.com - Visual work operating system"
            " for teams and organizations (no citation)\n"
            "4. Notion - Flexible all-in-one workspace"
            " for notes, docs, and projects (no citation)\n"
            "5. Trello - Lightweight Kanban board tool"
            " ideal for simple workflows (no citation)\n"
            "6. Jira - Powerful issue and project tracker"
            " for software teams (no citation)\n"
            "7. Linear - Modern issue tracking with"
            " keyboard-first design (no citation)\n"
            "8. Wrike - Enterprise work management"
            " with Gantt charts and dashboards (no citation)\n"
            "9. Smartsheet - Spreadsheet-based project"
            " management for large teams (no citation)\n"
            "10. Basecamp - Simple project communication"
            " and to-do management (no citation)\n"
        )
        return RunResult(
            raw_text=fake,
            raw_json=None,
            request_payload={
                "url": "STUB",
                "model": "stub-google",
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