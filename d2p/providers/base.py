"""LLMProvider protocol + RoleRouter.

Every agent talks to *one* LLMProvider. The orchestrator centralizes
provider/model assignment via a RoleRouter so the user can:

  - run everything on MiniMax (current default, single model)
  - run with Claude: Haiku for executor/fix, Opus for planner/analyzer/QA
  - run with Codex: gpt-4o-mini for executor/fix, gpt-4o for thinking
  - mix: Analyzer on Claude Opus, Executors on MiniMax, etc.

Adding a new provider = one file in this package + an entry in `__init__.factory`.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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

    All 4 agents (analyzer / planner / executor / qa) call `router.for_role(...)`.
    Smaller models (Haiku, gpt-4o-mini) get bound to executor + fix roles
    where speed and cost dominate. Heavier models (Opus, gpt-4o) get bound
    to analyzer / planner / qa where reasoning quality dominates.
    """

    DEFAULT_ROLES = ("analyzer", "planner", "executor", "qa")

    def __init__(self, providers: dict[str, LLMProvider]) -> None:
        if not providers:
            raise ValueError("RoleRouter needs at least one provider")
        self._providers = dict(providers)
        # ensure every default role resolves: missing → fall back to 'default'
        if "default" not in self._providers:
            self._providers["default"] = next(iter(providers.values()))

    def for_role(self, role: str) -> LLMProvider:
        return self._providers.get(role) or self._providers["default"]

    def describe(self) -> dict[str, str]:
        """{role: provider.name} — useful for logging the active routing."""
        return {role: self._providers.get(role, self._providers["default"]).name
                for role in set(list(self._providers.keys()) + list(self.DEFAULT_ROLES))}
