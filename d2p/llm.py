"""Backward-compat shim. New code should use `d2p.providers` directly.

`MiniMaxClient` is preserved as an alias for `MiniMaxProvider` so older
imports keep working. The `_extract_json` helper lives in providers/_json.py.
"""
from __future__ import annotations

from typing import Any, Optional

from .config import Config
from .providers._json import extract_json as _extract_json   # noqa: F401
from .providers.minimax import MiniMaxProvider


class MiniMaxClient(MiniMaxProvider):
    """Legacy alias: construct a MiniMaxProvider from a Config object."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        c = cfg or Config()
        c.require_key()
        super().__init__(
            api_key=c.api_key,
            model=c.model,
            base_url=c.base_url,
            timeout=c.request_timeout,
        )


__all__ = ["MiniMaxClient", "_extract_json"]
