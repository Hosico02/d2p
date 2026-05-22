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


# USD per token. Source: cloudprice.net / pricepertoken.com (May 2026).
# Unknown models fall back to (0, 0, 0) so cost shows 0 instead of crashing.
_PRICING: dict[str, tuple[float, float, float]] = {
    # (input_per_token, output_per_token, cache_read_per_token)
    "MiniMax-M2.7-highspeed": (0.6 / 1_000_000, 2.4 / 1_000_000, 0.06 / 1_000_000),
    "MiniMax-M2.7":           (0.279 / 1_000_000, 1.2 / 1_000_000, 0.0279 / 1_000_000),
    "MiniMax-M2.5-highspeed": (0.6 / 1_000_000, 2.4 / 1_000_000, 0.06 / 1_000_000),
    "MiniMax-M2.5":           (0.279 / 1_000_000, 1.2 / 1_000_000, 0.0279 / 1_000_000),
    "MiniMax-M2":             (0.3 / 1_000_000, 1.2 / 1_000_000, 0.03 / 1_000_000),
}


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
            in_t = getattr(u, "input_tokens", 0) or 0
            out_t = getattr(u, "output_tokens", 0) or 0
            cc_t = getattr(u, "cache_creation_input_tokens", 0) or 0
            cr_t = getattr(u, "cache_read_input_tokens", 0) or 0
            # Estimate USD cost from the per-token rate table. MiniMax's
            # Anthropic-compat response carries no cost field, so without
            # this we'd always log $0 — making model comparisons impossible.
            rates = _PRICING.get(self.model, (0.0, 0.0, 0.0))
            cost = in_t * rates[0] + out_t * rates[1] + cr_t * rates[2]
            self.usage.add(
                role=self.role, model=self.model,
                input_tokens=in_t, output_tokens=out_t,
                cache_creation_tokens=cc_t, cache_read_tokens=cr_t,
                cost_usd=cost,
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
