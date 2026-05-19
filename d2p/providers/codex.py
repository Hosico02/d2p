"""OpenAI ('Codex') provider — GPT-4o and GPT-4o-mini.

Default wiring:
  - gpt-4o-mini → executor / fix / self_heal
  - gpt-4o      → analyzer / planner / qa

Web-search: OpenAI's hosted web-search tool lives on gpt-4o-search-preview /
the Responses API. We honor `web_search=True` by switching to that model when
needed — non-search calls stay on the configured model.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ._json import extract_json

log = logging.getLogger("d2p.providers.codex")


class CodexProvider:
    SEARCH_MODEL = "gpt-4o-search-preview"

    def __init__(self, *, api_key: str, model: str,
                 base_url: str | None = None,
                 timeout: int = 240, role: str = "default") -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "CodexProvider requires `pip install openai`"
            ) from e
        self.role = role
        self.model = model
        self.name = f"codex:{model}@{role}"
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(self, system: str, user: str, *,
             web_search: bool = False, json_mode: bool = False,
             temperature: float = 0.4, max_tokens: int = 4096) -> str:
        if json_mode:
            user += "\n\nReturn ONLY a single JSON object/array. No prose, no markdown fences."
        chat_model = self.SEARCH_MODEL if web_search else self.model
        kwargs: dict[str, Any] = {
            "model": chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode and not web_search:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        return (msg.content or "").strip()

    def chat_json(self, system: str, user: str, *,
                  web_search: bool = False, temperature: float = 0.3,
                  max_tokens: int = 4096, retries: int = 2) -> Any:
        last_raw = ""
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            raw = self.chat(system, user, web_search=web_search, json_mode=True,
                            temperature=temperature + 0.1 * attempt,
                            max_tokens=max_tokens)
            last_raw = raw
            try:
                return extract_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("chat_json parse fail %d/%d: %s | head=%r",
                            attempt + 1, retries + 1, e, raw[:200])
                user += ("\n\nIMPORTANT: previous reply was not valid JSON. "
                         "Return ONLY a single JSON object/array, no prose.")
        raise RuntimeError(
            f"chat_json failed after {retries + 1} attempts: {last_err}; "
            f"last raw head: {last_raw[:300]!r}"
        )
