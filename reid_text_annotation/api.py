from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _completion_endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("REID_API_BASE_URL must be a valid HTTP(S) URL")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def _response_text(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API response does not contain choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    raise ValueError("API response content is not textual")


def parse_json_content(text: str) -> dict[str, Any]:
    cleaned = _FENCE_PATTERN.sub("", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response does not contain a JSON object")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Model response contains invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")
    return payload


@dataclass(frozen=True)
class CompatibleAPI:
    base_url: str
    api_key: str
    model: str
    timeout_sec: float = 120.0
    retries: int = 3
    max_output_tokens: int = 1800
    use_json_response_format: bool = True

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("REID_API_KEY is empty")
        if not self.model.strip():
            raise ValueError("REID_API_MODEL is empty")
        _completion_endpoint(self.base_url)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_url in image_urls or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                }
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _completion_endpoint(self.base_url),
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        attempts = max(1, self.retries + 1)
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(response_payload, dict):
                    raise ValueError("API response is not a JSON object")
                return parse_json_content(_response_text(response_payload))
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt + 1 >= attempts:
                    raise RuntimeError(f"API request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt + 1 >= attempts:
                    raise RuntimeError(f"API request failed: {type(exc).__name__}") from None

            delay = min(20.0, (2.0 ** attempt) + random.random())
            time.sleep(delay)

        raise RuntimeError("API request failed after retries")
