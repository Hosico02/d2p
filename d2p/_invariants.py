"""Runtime invariants for d2p.

Three helpers — `require`, `ensure`, `invariant` — encode preconditions,
postconditions, and internal-consistency checks. On failure each one
logs at ERROR with structured context (the kwargs you pass) and then
raises `InvariantError` (a subclass of AssertionError so existing
`except AssertionError` blocks still catch).

Env knob:
  D2P_INVARIANTS=strict  (default) — failures raise after logging
                =warn              — failures log only, execution continues
                =off               — checks become no-ops (cheap)

Use `require` at the top of a function for inputs/preconditions,
`ensure` before returning to verify postconditions, and `invariant`
mid-function for state that should always hold. The semantic split is
purely for readers; all three behave identically.
"""
from __future__ import annotations

import logging
import os
from typing import Any, NoReturn

log = logging.getLogger("d2p.invariants")


class InvariantError(AssertionError):
    """Raised when a runtime invariant fails (strict mode)."""

    def __init__(self, kind: str, msg: str, ctx: dict[str, Any]):
        self.kind = kind
        self.ctx = ctx
        ctx_str = " ".join(f"{k}={v!r}" for k, v in ctx.items())
        super().__init__(f"[{kind}] {msg}" + (f" | {ctx_str}" if ctx_str else ""))


def _mode() -> str:
    return os.environ.get("D2P_INVARIANTS", "strict").strip().lower() or "strict"


def _fail(kind: str, msg: str, ctx: dict[str, Any]) -> NoReturn | None:  # type: ignore[return]
    mode = _mode()
    if mode == "off":
        return None
    # Always log so warn-mode failures still leave a trail.
    log.error("invariant violated: %s | %s | %s", kind, msg,
              " ".join(f"{k}={v!r}" for k, v in ctx.items()))
    if mode == "warn":
        return None
    raise InvariantError(kind, msg, ctx)


def require(condition: bool, msg: str, **ctx: Any) -> None:
    """Precondition check. Use at function entry to validate inputs."""
    if _mode() == "off":
        return
    if not condition:
        _fail("require", msg, ctx)


def ensure(condition: bool, msg: str, **ctx: Any) -> None:
    """Postcondition check. Use before returning to validate outputs."""
    if _mode() == "off":
        return
    if not condition:
        _fail("ensure", msg, ctx)


def invariant(condition: bool, msg: str, **ctx: Any) -> None:
    """Internal-state check. Use mid-function for consistency assertions."""
    if _mode() == "off":
        return
    if not condition:
        _fail("invariant", msg, ctx)
