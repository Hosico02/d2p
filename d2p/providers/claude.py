"""Anthropic Claude provider.

Uses the official Anthropic API directly. By default the orchestrator wires:
  - Haiku (cheap+fast)  → executor / fix / self_heal
  - Opus (smart+deep)   → analyzer / planner / qa
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from ._json import extract_json
from .base import UsageAccumulator

log = logging.getLogger("d2p.providers.claude")


WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}


class ClaudeProvider:
    def __init__(self, *, api_key: str, model: str,
                 base_url: str | None = None,
                 timeout: int = 240, role: str = "default",
                 usage: UsageAccumulator | None = None) -> None:
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is empty")
        self.role = role
        self.model = model
        self.name = f"claude:{model}@{role}"
        self.usage = usage
        if base_url:
            self._client = anthropic.Anthropic(
                api_key=api_key, base_url=base_url, timeout=timeout,
            )
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key, timeout=timeout,
            )

    def chat(self, system: str, user: str, *,
             web_search: bool = False, json_mode: bool = False,
             temperature: float = 0.4, max_tokens: int = 4096) -> str:
        if json_mode:
            user += "\n\nReturn ONLY a single JSON object/array. No prose, no markdown fences."
        kwargs: dict[str, Any] = {
            "model": self.model, "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature,
        }
        if web_search:
            kwargs["tools"] = [WEB_SEARCH_TOOL]
        resp = self._client.messages.create(**kwargs)
        self._record_usage(resp)
        parts: list[str] = []
        for block in resp.content or []:
            # The Anthropic SDK content union has many block types; only
            # text blocks expose .text. mypy can't narrow on `block.type`
            # alone, so hop through getattr.
            if getattr(block, "type", None) == "text":
                txt = getattr(block, "text", "")
                if txt:
                    parts.append(txt)
        return "".join(parts).strip()

    def _record_usage(self, resp: Any) -> None:
        if self.usage is None:
            return
        u = getattr(resp, "usage", None)
        if u is None:
            return
        try:
            self.usage.add(
                role=self.role, model=self.model,
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                cache_creation_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            )
        except Exception as e:
            log.debug("usage record failed (%s): %s", self.name, e)

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
