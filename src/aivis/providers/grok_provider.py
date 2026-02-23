"""xAI Grok API provider."""
import os
import time

import httpx

from aivis.runner import RunResult
from aivis.settings import settings

XAI_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-3-fast"
MAX_RETRIES = 3
RETRYABLE = {429, 500, 502, 503}
TIMEOUT = 60


class GrokProvider:
    """xAI Grok API provider."""

    name: str = "grok"

    def check_api_key(self) -> bool:
        return settings.has_key("grok")

    def run(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RunResult:
        api_key = settings.get_key("grok")

        request_body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        request_payload = {
            "url": XAI_URL,
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
                        XAI_URL,
                        json=request_body,
                        headers=headers,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    model_hint = data.get("model", model)
                    return RunResult(
                        raw_text=raw_text,
                        raw_json=data,
                        request_payload=request_payload,
                        model_version_hint=model_hint,
                        usage=usage,
                    )

                if resp.status_code in RETRYABLE and attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt + 0.5)
                    last_error = RuntimeError(f"Grok API {resp.status_code}")
                    continue

                error_body = ""
                try:
                    error_body = resp.text
                except Exception:
                    pass
                raise RuntimeError(
                    f"Grok API returned {resp.status_code}: {error_body}"
                )

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt + 0.5)
                    continue
                raise

        raise last_error or RuntimeError("Grok API: max retries exceeded")

    def run_stub(self, prompt: str, **kwargs) -> RunResult:
        stub_text = (
            "1. Asana - Project management\n"
            "2. Monday.com - Work OS\n"
            "3. Trello - Kanban boards\n"
            "4. Jira - Issue tracking\n"
            "5. ClickUp - All-in-one\n"
            "6. Basecamp - Simplicity\n"
            "7. Notion - Docs and projects\n"
            "8. Wrike - Enterprise PM\n"
            "9. Smartsheet - Spreadsheet PM\n"
            "10. Teamwork - Client work"
        )
        return RunResult(
            raw_text=stub_text,
            raw_json={},
            request_payload={},
            model_version_hint="grok-3-fast-stub",
            usage={},
        )