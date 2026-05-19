"""Claude Code CLI provider — spawns `claude -p` subprocesses.

Uses the user's local `claude` binary (subscription auth via keychain).
Multiple parallel CLI invocations = multiple parallel Claude Code sessions.

Per-role tool allowlists:
  executor / fix → Read Glob Grep (writes happen via SEARCH/REPLACE pipeline)
  analyzer       → Read Glob Grep WebSearch WebFetch
  others         → Read Glob Grep

Prompt structure (stable-prefix order — important for cache hits):

    === System ===
    {system}                <- LARGE + STABLE per role

    === Role ===
    {role}                  <- STABLE per provider instance

    === User ===
    {user}                  <- VARIES per call

    === Call options ===
    [temp=... json=... web=...]   <- ALWAYS LAST so the prefix above stays
                                     bytewise-stable across retries

The Claude Code SDK auto-caches stable prompt prefixes. Keeping per-call
variables in the trailing block keeps `cache_read_input_tokens` high (we
log the ratio in run summary).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

from ._json import extract_json
from .base import UsageAccumulator

log = logging.getLogger("d2p.providers.claude_cli")


# A reasonable, role-specific tool allowlist. Space-separated string is the
# format Claude CLI's --allowedTools accepts.
ROLE_DEFAULT_TOOLS: dict[str, str] = {
    "executor":  "Read Glob Grep",     # read-only; writes happen via SEARCH/REPLACE pipeline
    "fix":       "Read Glob Grep",     # same — d2p applies the patch
    "analyzer":  "Read Glob Grep WebSearch WebFetch",
    "planner":   "Read Glob Grep",
    "qa":        "Read Glob Grep",
    "default":   "Read Glob Grep",
}


class ClaudeCLIProvider:
    """Drop-in LLMProvider backed by the `claude` CLI binary."""

    def __init__(self, *, model: str = "haiku", role: str = "default",
                 working_dir: str | None = None,
                 allowed_tools: str | None = None,
                 effort: str | None = None,
                 timeout: int = 600,
                 direct_edit: bool = False,
                 binary: str | None = None,
                 usage: UsageAccumulator | None = None) -> None:
        self.role = role
        self.model = model        # 'haiku' | 'sonnet' | 'opus' | full id
        self.name = f"claude-cli:{model}@{role}"
        self.working_dir = working_dir or os.getcwd()
        self.effort = effort      # None | low | medium | high | xhigh | max
        self.timeout = timeout
        self.direct_edit = direct_edit
        self.usage = usage
        self.binary = binary or shutil.which("claude") or "claude"
        if not shutil.which(self.binary):
            raise RuntimeError(
                f"`{self.binary}` not found on PATH. Install Claude Code "
                "(https://claude.com/claude-code) and run `claude login`."
            )
        # default tools by role (executor=read-only unless direct_edit)
        if allowed_tools is not None:
            self.allowed_tools = allowed_tools
        elif direct_edit and role == "executor":
            self.allowed_tools = "Read Edit Write Glob Grep Bash"
        else:
            self.allowed_tools = ROLE_DEFAULT_TOOLS.get(role, ROLE_DEFAULT_TOOLS["default"])

    # ---- LLMProvider interface ----------------------------------------------

    def chat(self, system: str, user: str, *,
             web_search: bool = False, json_mode: bool = False,
             temperature: float = 0.4, max_tokens: int = 4096) -> str:
        # claude CLI doesn't expose temperature/max_tokens directly via -p;
        # we surface them at the *end* of the prompt as soft hints so the
        # prefix stays bytewise-stable across retries (cache-friendly).
        prompt = self._build_prompt(system, user,
                                     web_search=web_search,
                                     json_mode=json_mode,
                                     temperature=temperature,
                                     max_tokens=max_tokens)
        cmd = self._build_cmd(web_search=web_search)
        log.debug("claude-cli (%s) launch: tools=%s", self.role, self.allowed_tools)
        try:
            r = subprocess.run(
                cmd, input=prompt, cwd=self.working_dir,
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"claude CLI timed out after {self.timeout}s (role={self.role})"
            )
        if r.returncode != 0:
            raise RuntimeError(
                f"claude CLI exit {r.returncode}: {(r.stderr or r.stdout)[:500]}"
            )
        # parse JSON wrapper
        out = r.stdout.strip()
        data: dict[str, Any] = {}
        try:
            data = json.loads(out.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            # fall back: whole stdout is the text, no usage info available
            self._record_usage({})
            return out
        if data.get("is_error"):
            raise RuntimeError(
                f"claude CLI returned error: {data.get('result', '')[:300]}"
            )
        self._record_usage(data)
        return (data.get("result", "") or "").strip()

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

    # ---- internals ----------------------------------------------------------

    def _record_usage(self, data: dict[str, Any]) -> None:
        if self.usage is None:
            return
        u = data.get("usage") or {}
        # Claude CLI's `total_cost_usd` lives at the top level; some versions
        # also expose it inside `usage`. Take whichever is present.
        cost = data.get("total_cost_usd")
        if cost is None:
            cost = u.get("total_cost_usd", 0.0)
        try:
            self.usage.add(
                role=self.role, model=self.model,
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                cache_creation_tokens=u.get("cache_creation_input_tokens", 0),
                cache_read_tokens=u.get("cache_read_input_tokens", 0),
                cost_usd=cost or 0.0,
            )
        except Exception as e:
            log.debug("usage record failed (%s): %s", self.name, e)

    def _build_cmd(self, *, web_search: bool) -> list[str]:
        tools = self.allowed_tools
        if web_search and "WebSearch" not in tools:
            tools = tools + " WebSearch WebFetch"
        cmd = [
            self.binary, "-p",
            "--output-format", "json",
            "--input-format", "text",
            "--model", self.model,
            "--permission-mode", "bypassPermissions",
            "--allowedTools", tools,
            "--no-session-persistence",
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    def _build_prompt(self, system: str, user: str, *,
                      web_search: bool, json_mode: bool,
                      temperature: float, max_tokens: int) -> str:
        """Order is fixed for cache-prefix stability:
            [System] (large, stable) -> [Role] (stable) -> [User] (per call) ->
            [Call options] (per call, trailing)
        Keeping all variable hints AFTER the user block means the bytewise
        prefix up through the user portion is reproducible, and the SDK's
        prompt cache can read it instead of re-encoding the full system block.
        """
        trailing_directives: list[str] = []
        if web_search:
            trailing_directives.append(
                "You may use WebSearch / WebFetch for current info."
            )
        if json_mode:
            trailing_directives.append(
                "Return ONLY a single JSON object/array. No prose."
            )
        trailing_directives.append(
            f"[call: temp={temperature} max_tokens={max_tokens}]"
        )
        return (
            f"=== System ===\n{system}\n\n"
            f"=== Role ===\n{self.role}\n\n"
            f"=== User ===\n{user}\n\n"
            f"=== Call options ===\n" + "\n".join(trailing_directives)
        )
