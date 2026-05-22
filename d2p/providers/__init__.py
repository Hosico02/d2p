"""Provider factory + per-role default model maps.

Pick a provider with env `D2P_PROVIDER=minimax|claude|codex` (default minimax).
Per-role models can be overridden via env, e.g. `D2P_ROLE_EXECUTOR_MODEL=...`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .base import LLMProvider, RoleRouter, UsageAccumulator
from .claude import ClaudeProvider
from .claude_cli import ClaudeCLIProvider
from .codex import CodexProvider
from .minimax import MiniMaxProvider


# Default tier ladders per provider, per role. Tier 0 is the cheap/fast
# primary; tier N is the strongest. A task that fails at tier_idx N gets
# requeued for the next iter at tier_idx min(N+1, len(ladder)-1). At the
# top tier we retry 3× before marking the task dead.
#
# Only `executor` and `fix` get ladders — analyzer/planner/qa are one-shot
# reasoning calls where retrying at a stronger model isn't meaningful in
# the same way (a planner that produces a bad plan needs a re-plan, not
# the same prompt at a bigger model).
DEFAULT_LADDERS: dict[str, dict[str, list[str]]] = {
    "claude": {
        "executor": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
        "fix":      ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
    },
    "claude-cli": {
        "executor": ["haiku", "sonnet", "opus"],
        "fix":      ["haiku", "sonnet", "opus"],
    },
    "codex": {
        "executor": ["gpt-4o-mini", "gpt-4o"],
        "fix":      ["gpt-4o-mini", "gpt-4o"],
    },
    "minimax": {
        # Single-tier: no real escalation between MiniMax models. Top-tier
        # retry policy (3x at top before dead) takes over for failing tasks.
        "executor": ["MiniMax-M2.7-highspeed"],
        "fix":      ["MiniMax-M2.7-highspeed"],
    },
}


# Sensible per-role defaults — Hiku/mini on hot path, Opus/4o on reasoning.
DEFAULT_ROLE_MODELS: dict[str, dict[str, str]] = {
    "claude": {
        "executor":  "claude-haiku-4-5",
        "fix":       "claude-haiku-4-5",       # override to claude-sonnet-4-6 / opus when bugs resist
        "analyzer":  "claude-opus-4-7",
        "planner":   "claude-opus-4-7",
        "qa":        "claude-opus-4-7",
        "default":   "claude-haiku-4-5",
    },
    "codex": {
        "executor":  "gpt-4o-mini",
        "fix":       "gpt-4o-mini",
        "analyzer":  "gpt-4o",
        "planner":   "gpt-4o",
        "qa":        "gpt-4o",
        "default":   "gpt-4o-mini",
    },
    "minimax": {
        # MiniMax is single-model: one model for every role unless overridden.
    },
    "claude-cli": {
        # Hiku for hot path; fix uses sonnet by default (harder than feature edits)
        "executor":  "haiku",
        "fix":       "haiku",                  # override via D2P_ROLE_FIX_MODEL=sonnet|opus
        "analyzer":  "opus",
        "planner":   "opus",
        "qa":        "opus",
        "default":   "haiku",
    },
}


@dataclass
class ProviderSpec:
    """Resolved provider config used to construct the RoleRouter."""
    kind: str = "minimax"              # minimax | claude | codex
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    role_models: dict[str, str] = field(default_factory=dict)
    # Per-role tier ladder: { "executor": ["haiku","sonnet","opus"], ... }.
    # Carry-over queue uses this to bump a failing task's tier_idx each
    # iter. Defaults from DEFAULT_LADDERS[kind]; env overrides via
    # D2P_ROLE_<ROLE>_LADDER=a,b,c. Replaces the old fallback_models /
    # D2P_ROLE_<ROLE>_FALLBACK_MODEL within-iter escalation.
    role_ladders: dict[str, list[str]] = field(default_factory=dict)


def _from_env() -> ProviderSpec:
    kind = os.environ.get("D2P_PROVIDER", "minimax").lower()
    if kind not in {"minimax", "claude", "codex", "claude-cli"}:
        raise ValueError(f"unsupported D2P_PROVIDER={kind!r}")

    if kind == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        base = os.environ.get("ANTHROPIC_BASE_URL", "")
        default_model = os.environ.get("D2P_CLAUDE_MODEL", "claude-haiku-4-5")
    elif kind == "claude-cli":
        # CLI uses the user's subscription via keychain — no API key needed.
        key = "claude-cli-keychain"
        base = ""
        default_model = os.environ.get("D2P_CLAUDE_CLI_MODEL", "haiku")
    elif kind == "codex":
        key = os.environ.get("OPENAI_API_KEY", "")
        base = os.environ.get("OPENAI_BASE_URL", "")
        default_model = os.environ.get("D2P_CODEX_MODEL", "gpt-4o-mini")
    else:  # minimax
        key = os.environ.get("MINIMAX_API_KEY", "")
        base = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")
        default_model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7-highspeed")

    # role overrides from env (D2P_ROLE_EXECUTOR_MODEL=...)
    overrides: dict[str, str] = {}
    ladder_overrides: dict[str, list[str]] = {}
    for role in ("executor", "fix", "analyzer", "planner", "qa", "default"):
        env_name = f"D2P_ROLE_{role.upper()}_MODEL"
        v = os.environ.get(env_name)
        if v:
            overrides[role] = v
        ladder_env = f"D2P_ROLE_{role.upper()}_LADDER"
        ladder_v = os.environ.get(ladder_env)
        if ladder_v:
            ladder_overrides[role] = [m.strip() for m in ladder_v.split(",")
                                       if m.strip()]
    role_models = {**DEFAULT_ROLE_MODELS.get(kind, {}), **overrides}
    role_ladders = {**DEFAULT_LADDERS.get(kind, {}), **ladder_overrides}

    return ProviderSpec(
        kind=kind, api_key=key, base_url=base,
        default_model=default_model, role_models=role_models,
        role_ladders=role_ladders,
    )


def _make_provider(kind: str, *, api_key: str, base_url: str,
                   model: str, role: str,
                   working_dir: str | None = None,
                   usage: UsageAccumulator | None = None) -> LLMProvider:
    if kind == "claude":
        return ClaudeProvider(api_key=api_key, model=model,
                              base_url=base_url or None, role=role, usage=usage)
    if kind == "claude-cli":
        return ClaudeCLIProvider(model=model, role=role,
                                  working_dir=working_dir, usage=usage)
    if kind == "codex":
        return CodexProvider(api_key=api_key, model=model,
                             base_url=base_url or None, role=role, usage=usage)
    return MiniMaxProvider(api_key=api_key, model=model,
                           base_url=base_url or "https://api.minimaxi.com/anthropic",
                           role=role, usage=usage)


def build_router(spec: ProviderSpec | None = None,
                 working_dir: str | None = None) -> RoleRouter:
    """Construct one LLMProvider per role and bundle them in a RoleRouter.

    `working_dir` is the target sandbox path. Required for claude-cli so
    each CLI invocation runs in the project root.
    """
    s = spec or _from_env()
    if not s.api_key:
        env_var = {
            "minimax": "MINIMAX_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "claude-cli": "(uses keychain — run `claude login`)",
            "codex": "OPENAI_API_KEY",
        }[s.kind]
        raise RuntimeError(
            f"No API key for provider {s.kind}. Set the appropriate env: {env_var}"
        )
    usage = UsageAccumulator()
    providers: dict[str, LLMProvider] = {}
    providers["default"] = _make_provider(
        s.kind, api_key=s.api_key, base_url=s.base_url,
        model=s.role_models.get("default", s.default_model), role="default",
        working_dir=working_dir, usage=usage,
    )
    for role in ("executor", "fix", "analyzer", "planner", "qa"):
        m = s.role_models.get(role, s.default_model)
        if m == s.role_models.get("default", s.default_model):
            # Share the default provider — BUT a single provider instance has
            # a fixed `role` label, so usage attribution would all bucket as
            # 'default'. Construct a tiny per-role provider so usage is split.
            providers[role] = _make_provider(
                s.kind, api_key=s.api_key, base_url=s.base_url, model=m,
                role=role, working_dir=working_dir, usage=usage,
            )
            continue
        providers[role] = _make_provider(
            s.kind, api_key=s.api_key, base_url=s.base_url, model=m, role=role,
            working_dir=working_dir, usage=usage,
        )
    # Tier ladders: one provider per (role, tier_idx). Each tier gets its
    # own role label `<role>-tier<idx>` so per-tier usage shows in the cost
    # summary — makes it visible how often we had to escalate.
    ladders: dict[str, list[LLMProvider]] = {}
    for role, ladder_models in s.role_ladders.items():
        if role == "default" or not ladder_models:
            continue
        ladders[role] = [
            _make_provider(
                s.kind, api_key=s.api_key, base_url=s.base_url, model=m,
                role=f"{role}-tier{idx}", working_dir=working_dir, usage=usage,
            )
            for idx, m in enumerate(ladder_models)
        ]
    return RoleRouter(providers, usage=usage, ladders=ladders)


__all__ = [
    "LLMProvider", "RoleRouter", "ProviderSpec", "UsageAccumulator",
    "MiniMaxProvider", "ClaudeProvider", "ClaudeCLIProvider", "CodexProvider",
    "build_router", "DEFAULT_ROLE_MODELS", "DEFAULT_LADDERS",
]
