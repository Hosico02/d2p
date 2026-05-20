"""MiniMax Token-Plan provider (Anthropic-compatible protocol).

Wraps api.minimaxi.com/anthropic via the `anthropic` SDK with a custom base_url.
This is the default for users with `sk-cp-` keys.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from ._json import extract_json
from .base import UsageAccumulator

log = logging.getLogger("d2p.providers.minimax")


class MiniMaxProvider:
    def __init__(self, *, api_key: str, model: str,
                 base_url: str = "https://api.minimaxi.com/anthropic",
                 timeout: int = 240, role: str = "default",
                 usage: UsageAccumulator | None = None) -> None:
        if not api_key:
            raise RuntimeError("MiniMax API key is empty")
        self.role = role
        self.model = model
        self.name = f"minimax:{model}@{role}"
        self.usage = usage
        self._client = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, timeout=timeout,
        )

    def chat(self, system: str, user: str, *,
             web_search: bool = False, json_mode: bool = False,
             temperature: float = 0.4, max_tokens: int = 4096) -> str:
        if web_search:
            user = ("You have live web access. Use it to gather up-to-date "
                    "facts about real products before answering.\n\n" + user)
        if json_mode:
            user += "\n\nReturn ONLY a single JSON object/array. No prose, no markdown fences."
        resp = self._client.messages.create(
            model=self.model, system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature,
        )
        self._record_usage(resp)
        parts: list[str] = []
        for block in resp.content or []:
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
