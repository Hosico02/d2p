"""Provider factory + per-role default model maps.

Pick a provider with env `D2P_PROVIDER=minimax|claude|codex` (default minimax).
Per-role models can be overridden via env, e.g. `D2P_ROLE_EXECUTOR_MODEL=...`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .base import LLMProvider, RoleRouter
from .claude import ClaudeProvider
from .claude_cli import ClaudeCLIProvider
from .codex import CodexProvider
from .minimax import MiniMaxProvider


# Sensible per-role defaults — Hiku/mini on hot path, Opus/4o on reasoning.
DEFAULT_ROLE_MODELS: dict[str, dict[str, str]] = {
    "claude": {
        "executor":  "claude-haiku-4-5",
        "analyzer":  "claude-opus-4-7",
        "planner":   "claude-opus-4-7",
        "qa":        "claude-opus-4-7",
        "default":   "claude-haiku-4-5",
    },
    "codex": {
        "executor":  "gpt-4o-mini",
        "analyzer":  "gpt-4o",
        "planner":   "gpt-4o",
        "qa":        "gpt-4o",
        "default":   "gpt-4o-mini",
    },
    "minimax": {
        # MiniMax is single-model: one model for every role unless overridden.
        # Empty dict = always use the provider's configured default model.
    },
    "claude-cli": {
        # Same Hiku/Opus split as Claude API, just via the CLI binary.
        "executor":  "haiku",
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
    for role in ("executor", "analyzer", "planner", "qa", "default"):
        env_name = f"D2P_ROLE_{role.upper()}_MODEL"
        v = os.environ.get(env_name)
        if v:
            overrides[role] = v
    role_models = {**DEFAULT_ROLE_MODELS.get(kind, {}), **overrides}

    return ProviderSpec(
        kind=kind, api_key=key, base_url=base,
        default_model=default_model, role_models=role_models,
    )


def _make_provider(kind: str, *, api_key: str, base_url: str,
                   model: str, role: str,
                   working_dir: str | None = None) -> LLMProvider:
    if kind == "claude":
        return ClaudeProvider(api_key=api_key, model=model,
                              base_url=base_url or None, role=role)
    if kind == "claude-cli":
        return ClaudeCLIProvider(model=model, role=role,
                                  working_dir=working_dir)
    if kind == "codex":
        return CodexProvider(api_key=api_key, model=model,
                             base_url=base_url or None, role=role)
    return MiniMaxProvider(api_key=api_key, model=model,
                           base_url=base_url or "https://api.minimaxi.com/anthropic",
                           role=role)


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
    providers: dict[str, LLMProvider] = {}
    providers["default"] = _make_provider(
        s.kind, api_key=s.api_key, base_url=s.base_url,
        model=s.role_models.get("default", s.default_model), role="default",
        working_dir=working_dir,
    )
    for role in ("executor", "analyzer", "planner", "qa"):
        m = s.role_models.get(role, s.default_model)
        if m == s.role_models.get("default", s.default_model):
            providers[role] = providers["default"]
            continue
        providers[role] = _make_provider(
            s.kind, api_key=s.api_key, base_url=s.base_url, model=m, role=role,
            working_dir=working_dir,
        )
    return RoleRouter(providers)


__all__ = [
    "LLMProvider", "RoleRouter", "ProviderSpec",
    "MiniMaxProvider", "ClaudeProvider", "ClaudeCLIProvider", "CodexProvider",
    "build_router", "DEFAULT_ROLE_MODELS",
]
