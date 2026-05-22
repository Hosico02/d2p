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
        # Free-form counters surfaced in the summary alongside per-role
        # usage. Self-heal attempts/successes go here so the user can see
        # how often the safety net fires without scraping logs.
        self._counters: dict[str, int] = {}
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

    def increment(self, key: str, by: int = 1) -> None:
        """Bump a free-form counter (e.g. 'self_heal_attempts'). Thread-safe."""
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + by

    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

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
        # cache-hit ratio = read / (read + creation + uncached-input).
        # Includes raw input_tokens in the denominator so providers that
        # report cache_read but never cache_creation (e.g. MiniMax) don't
        # falsely show 1.0. For claude-cli (input≈0) this matches the old
        # formula within rounding.
        cache_total = total_cache_read + total_cache_creation + total_input
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
            "counters": self.counters(),
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


def chat_structured(provider: LLMProvider, system: str, user: str,
                    *, schema: dict[str, Any], temperature: float = 0.3,
                    max_tokens: int = 4096) -> Any:
    """Coerce a JSON-shaped response that conforms to `schema`.

    Calls the provider's bespoke structured-output API when available
    (faster + more reliable: the model is forced into the shape instead
    of reasoning about the format). Falls back to chat_json with the
    schema embedded in the prompt otherwise.

    `schema` is JSON Schema. Top-level should usually be an object with
    a `properties` map.
    """
    # Provider-specific fast path
    fast = getattr(provider, "chat_structured", None)
    if callable(fast):
        return fast(system, user, schema=schema,
                    temperature=temperature, max_tokens=max_tokens)
    # Generic fallback: append the schema to the user prompt as a hint.
    import json as _j
    augmented = (
        user + "\n\nReturn STRICT JSON conforming to this schema "
        "(no extra keys, no markdown fences):\n"
        + _j.dumps(schema, ensure_ascii=False, indent=2)
    )
    return provider.chat_json(system, augmented,
                              temperature=temperature, max_tokens=max_tokens)


class RoleRouter:
    """Pre-builds one LLMProvider instance per role and serves them by role name.

    All 5 roles (analyzer / planner / executor / fix / qa) call
    `router.for_role(...)`. Smaller models (Haiku, gpt-4o-mini) get bound to
    executor + fix roles where speed and cost dominate. Heavier models (Opus,
    gpt-4o) get bound to analyzer / planner / qa where reasoning quality
    dominates.

    The router also owns a single UsageAccumulator shared by every provider it
    constructed (wiring happens in `build_router`).

    Tier ladders
    ------------
    Per-role ordered ladder of providers (tier 0 = cheap/fast primary,
    tier N = strongest). The carry-over queue bumps a failing task's
    tier_idx each iter; orchestrator dispatches per task at its current
    tier via `for_role_tier`. Configure via DEFAULT_LADDERS or env
    `D2P_ROLE_<ROLE>_LADDER=a,b,c`.
    """

    DEFAULT_ROLES = ("analyzer", "planner", "executor", "fix", "qa")

    def __init__(self, providers: dict[str, LLMProvider],
                 usage: UsageAccumulator | None = None,
                 ladders: dict[str, list[LLMProvider]] | None = None) -> None:
        if not providers:
            raise ValueError("RoleRouter needs at least one provider")
        self._providers = dict(providers)
        # role -> ordered ladder of providers; tier 0 = primary, tier N = strongest.
        # Tasks that fail get re-queued for next iter at tier_idx+1 (orchestrator
        # owns the bump; dispatch picks the per-task tier).
        self._ladders: dict[str, list[LLMProvider]] = dict(ladders or {})
        # ensure every default role resolves: missing → fall back to 'default'
        if "default" not in self._providers:
            self._providers["default"] = next(iter(providers.values()))
        self.usage = usage or UsageAccumulator()

    def for_role(self, role: str) -> LLMProvider:
        return self._providers.get(role) or self._providers["default"]

    def for_role_tier(self, role: str, tier_idx: int) -> LLMProvider:
        """Return the provider at `tier_idx` of `role`'s ladder. If no ladder
        is configured for the role, falls back to for_role. tier_idx is
        clamped to [0, len(ladder)-1] so callers can pass an arbitrary
        bumped index without bounds-checking themselves."""
        ladder = self._ladders.get(role)
        if not ladder:
            return self.for_role(role)
        clamped = max(0, min(tier_idx, len(ladder) - 1))
        return ladder[clamped]

    def ladder_length(self, role: str) -> int:
        """How many tiers `role` has. 1 means no real ladder — a task at
        tier 0 that fails has nowhere to escalate to, so retries 3× at the
        top tier per the dead-task policy."""
        ladder = self._ladders.get(role)
        return len(ladder) if ladder else 1

    def describe(self) -> dict[str, str]:
        """{role: provider.name} — useful for logging the active routing.
        Ladder roles appear as `<role>-tier<N>`."""
        out = {role: self._providers.get(role, self._providers["default"]).name
               for role in set(list(self._providers.keys()) + list(self.DEFAULT_ROLES))}
        for role, ladder in self._ladders.items():
            for i, p in enumerate(ladder):
                out[f"{role}-tier{i}"] = p.name
        return out
