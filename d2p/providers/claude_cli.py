"""Claude Code CLI provider — spawns `claude -p` subprocesses.

Uses the user's local `claude` binary (subscription auth via keychain).
Multiple parallel CLI invocations = multiple parallel Claude Code sessions.

Per-role tool allowlists:
  executor  →  Read Edit Write Glob Grep Bash   (full file-editing power)
  others    →  Read Glob Grep WebSearch WebFetch (read-only + research)

Note: we still ask the executor model to emit our ===FILE=== / ===PATCH===
delimited blocks, even though it has Edit/Write tools. This keeps the d2p
sandbox+snapshot+rollback pipeline as the single source of truth for what
actually got written. To let Claude write directly, add Edit/Write to
allowed_tools and trust its work (loses our safety nets — opt-in via
`direct_edit=True`).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

from ._json import extract_json

log = logging.getLogger("d2p.providers.claude_cli")


# A reasonable, role-specific tool allowlist. Space-separated string is the
# format Claude CLI's --allowedTools accepts.
ROLE_DEFAULT_TOOLS: dict[str, str] = {
    "executor":  "Read Glob Grep",     # read-only; writes happen via SEARCH/REPLACE pipeline
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
                 binary: str | None = None) -> None:
        self.role = role
        self.model = model        # 'haiku' | 'sonnet' | 'opus' | full id
        self.name = f"claude-cli:{model}@{role}"
        self.working_dir = working_dir or os.getcwd()
        self.effort = effort      # None | low | medium | high | xhigh | max
        self.timeout = timeout
        self.direct_edit = direct_edit
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
        # we surface them as soft hints in the prompt frontmatter.
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
        try:
            data = json.loads(out.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            # fall back: whole stdout is the text
            return out
        if data.get("is_error"):
            raise RuntimeError(
                f"claude CLI returned error: {data.get('result', '')[:300]}"
            )
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
        # CLI doesn't have a `system` slot in -p mode → prepend.
        prelude = []
        prelude.append(f"[role={self.role}  temp={temperature}  max_tokens={max_tokens}]")
        if web_search:
            prelude.append("You may use WebSearch / WebFetch for current info.")
        if json_mode:
            prelude.append("Return ONLY a single JSON object/array. No prose.")
        prelude_block = "\n".join(prelude)
        return (
            f"=== System ===\n{system}\n\n"
            f"=== Directives ===\n{prelude_block}\n\n"
            f"=== User ===\n{user}"
        )
