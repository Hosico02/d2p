"""LLMProvider protocol + RoleRouter + UsageAccumulator.

Every agent talks to *one* LLMProvider. The orchestrator centralizes
provider/model assignment via a RoleRouter so the user can:

  - run everything on MiniMax (current default, single model)
  - run with Claude: Haiku for executor/fix, Opus for planner/analyzer/QA
  - run with Codex: gpt-4o-mini for executor/fix, gpt-4o for thinking
  - mix: Analyzer on Claude Opus, Executors on MiniMax, etc.

Adding a new provider = one file in this package + an entry in `__init__.factory`.

Usage tracking
--------------
The RoleRouter owns one UsageAccumulator. Each provider receives a reference
and appends a record after every chat() call (tokens + cache hits + cost). The
orchestrator dumps `router.usage.summary()` into summary.json so users can
see how much each role spent.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class UsageRecord:
    role: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0


class UsageAccumulator:
    """Thread-safe ledger of every LLM call's tokens + cache + USD cost.

    Cost is only populated for providers that surface it (currently Claude
    Code CLI via `total_cost_usd`). For others we still track tokens so the
    user sees comparative volume.
    """

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    def add(self, *, role: str, model: str, input_tokens: int = 0,
            output_tokens: int = 0, cache_creation_tokens: int = 0,
            cache_read_tokens: int = 0, cost_usd: float = 0.0) -> None:
        rec = UsageRecord(
            role=role, model=model,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cache_creation_tokens=int(cache_creation_tokens or 0),
            cache_read_tokens=int(cache_read_tokens or 0),
            cost_usd=float(cost_usd or 0.0),
        )
        with self._lock:
            self._records.append(rec)

    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict[str, Any]:
        per_role: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "input": 0, "output": 0,
                     "cache_creation": 0, "cache_read": 0, "cost_usd": 0.0}
        )
        with self._lock:
            records = list(self._records)
        for r in records:
            key = f"{r.role}:{r.model}"
            d = per_role[key]
            d["calls"] += 1
            d["input"] += r.input_tokens
            d["output"] += r.output_tokens
            d["cache_creation"] += r.cache_creation_tokens
            d["cache_read"] += r.cache_read_tokens
            d["cost_usd"] += r.cost_usd
        total_cost = sum(r.cost_usd for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cache_read = sum(r.cache_read_tokens for r in records)
        total_cache_creation = sum(r.cache_creation_tokens for r in records)
        # cache-hit ratio = read / (read + creation). 1.0 = perfect cache.
        cache_total = total_cache_read + total_cache_creation
        cache_hit_ratio = (round(total_cache_read / cache_total, 3)
                           if cache_total else 0.0)
        return {
            "total_calls": len(records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_creation_tokens": total_cache_creation,
            "total_cache_read_tokens": total_cache_read,
            "cache_hit_ratio": cache_hit_ratio,
            "total_cost_usd": round(total_cost, 4),
            "per_role": {
                k: {**v, "cost_usd": round(v["cost_usd"], 4)}
                for k, v in per_role.items()
            },
        }


# ---- Provider protocol ------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    name: str        # human label e.g. "claude:opus-4-7"

    def chat(self, system: str, user: str, *,
             web_search: bool = False, json_mode: bool = False,
             temperature: float = 0.4, max_tokens: int = 4096) -> str: ...

    def chat_json(self, system: str, user: str, *,
                  web_search: bool = False, temperature: float = 0.3,
                  max_tokens: int = 4096, retries: int = 2) -> Any: ...


class RoleRouter:
    """Pre-builds one LLMProvider instance per role and serves them by role name.

    All 5 roles (analyzer / planner / executor / fix / qa) call
    `router.for_role(...)`. Smaller models (Haiku, gpt-4o-mini) get bound to
    executor + fix roles where speed and cost dominate. Heavier models (Opus,
    gpt-4o) get bound to analyzer / planner / qa where reasoning quality
    dominates.

    The router also owns a single UsageAccumulator shared by every provider it
    constructed (wiring happens in `build_router`).

    Fallback providers
    ------------------
    Optional per-role fallbacks let the orchestrator retry a failed task with
    a stronger model. Wired through D2P_ROLE_<ROLE>_FALLBACK_MODEL env, e.g.
    `D2P_ROLE_EXECUTOR_FALLBACK_MODEL=sonnet`. Usage from fallback retries is
    attributed under a `<role>-fallback` role label so cost breakdowns make
    the escalation visible.
    """

    DEFAULT_ROLES = ("analyzer", "planner", "executor", "fix", "qa")

    def __init__(self, providers: dict[str, LLMProvider],
                 usage: UsageAccumulator | None = None,
                 fallbacks: dict[str, LLMProvider] | None = None) -> None:
        if not providers:
            raise ValueError("RoleRouter needs at least one provider")
        self._providers = dict(providers)
        self._fallbacks: dict[str, LLMProvider] = dict(fallbacks or {})
        # ensure every default role resolves: missing → fall back to 'default'
        if "default" not in self._providers:
            self._providers["default"] = next(iter(providers.values()))
        self.usage = usage or UsageAccumulator()

    def for_role(self, role: str) -> LLMProvider:
        return self._providers.get(role) or self._providers["default"]

    def for_fallback(self, role: str) -> LLMProvider | None:
        """Return the role's escalation provider if configured, else None."""
        return self._fallbacks.get(role)

    def describe(self) -> dict[str, str]:
        """{role: provider.name} — useful for logging the active routing.
        Fallback roles appear as `<role>-fallback`."""
        out = {role: self._providers.get(role, self._providers["default"]).name
               for role in set(list(self._providers.keys()) + list(self.DEFAULT_ROLES))}
        for role, fp in self._fallbacks.items():
            out[f"{role}-fallback"] = fp.name
        return out
