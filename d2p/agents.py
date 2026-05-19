"""Analyzer / Planner / Executor agents.

Design goal: no hardcoded demo-type detectors. Each agent works off the raw
project listing + a short readme, and lets the LLM decide what the project
is and what features a competitor product has. Adding a new demo type
(audio, blockchain, …) requires *no* code changes here.
"""
from __future__ import annotations

import ast
import json as _json
import logging
import re as _re
import uuid
from typing import Any, Callable, Optional, Tuple

from .fs import Sandbox
from .lang import LanguageAdapter, NullAdapter, adapter_for, detect_primary_language
from .llm import MiniMaxClient
from .models import AnalysisReport, ExecutionResult, Feature, PlanResult, Task
from .symbols import build_symbol_map

log = logging.getLogger("d2p.agents")


# ============ Analyzer ========================================================

ANALYZER_SYS = """You are the Analyzer agent in a Demo-to-Product (d2p) pipeline.
Your job:

1. Read the demo (listing + key files) and identify TWO things separately:
   - "domain": the problem area (e.g. "social deduction game", "speech-to-text")
   - "essence": the demo's CORE NATURE that must be preserved across iterations.
     The essence captures what kind of artifact this demo IS — who its real
     audience is, and what makes it distinct from typical products in the
     same domain.
   - "audience": one short phrase, e.g. "LLM agents", "developers via API",
     "research notebook users", "humans on a web UI", "CLI power-users".

   Examples:
     * werewolf demo where 6 LLM-driven players debate each other
         -> domain: "Werewolf / social deduction"
         -> essence: "an Agent-vs-Agent simulation harness where LLM players
            debate, vote and reason; humans are spectators, not players"
         -> audience: "LLM agents (humans only observe / analyze)"
     * a Whisper-based offline transcriber CLI
         -> essence: "an offline batch-processing CLI; not a real-time web app"
         -> audience: "CLI users / scripted pipelines"

2. Search the web for 3-5 MATURE COMPETITOR PRODUCTS in the same DOMAIN.

3. From competitors, extract concrete features and UI elements that would
   improve a PRODUCT BUILT FROM THIS DEMO **without changing its essence**.
   - If the demo is agent-facing, do NOT propose human-multiplayer features
     like lobby codes, voice chat, ranked ladders — those would change the
     essence into a different product.
   - DO propose agent-facing analogues: e.g. instead of "voice chat", suggest
     "structured wolf-private channel for inter-agent reasoning"; instead of
     "leaderboard", suggest "agent performance benchmark dashboard".
   - Be concrete — "login with Google" not "auth".

4. Output STRICT JSON only — no markdown, no commentary.
"""

ANALYZER_USER_TMPL = """Demo project files:
{listing}

Top-level documents (truncated):
{docs}

Return a JSON object with this exact shape:
{{
  "domain": "<one sentence>",
  "essence": "<one or two sentences — the demo's core nature that must NOT change>",
  "audience": "<one short phrase>",
  "competitors": ["<product name + 1-line desc>", ...],
  "features": [
    {{"name": "...", "category": "backend|frontend|ux|ops|docs", "description": "...", "source": "<competitor name>"}}
  ],
  "ui_elements": ["...", ...],
  "raw_notes": "<short freeform notes, max 500 chars>"
}}
Provide 8-15 features. Skip features that would change the audience or essence.
Skip anything the demo clearly already has.
"""


class Analyzer:
    def __init__(self, llm: MiniMaxClient, sandbox: Sandbox) -> None:
        self.llm = llm
        self.sandbox = sandbox

    def _gather_docs(self) -> str:
        candidates = ["README.md", "README", "readme.md", "package.json",
                      "pyproject.toml", "requirements.txt", "main.py", "app.py",
                      "index.js", "index.ts", "src/main.ts", "Cargo.toml"]
        chunks = []
        for c in candidates:
            txt = self.sandbox.read(c)
            if txt:
                chunks.append(f"=== {c} ===\n{txt[:2500]}")
        return "\n\n".join(chunks) or "(no obvious entry/doc files found)"

    def run(self) -> AnalysisReport:
        listing = "\n".join(self.sandbox.listing(max_entries=120))
        docs = self._gather_docs()
        user = ANALYZER_USER_TMPL.format(listing=listing, docs=docs)
        data = self.llm.chat_json(ANALYZER_SYS, user, web_search=True,
                                  temperature=0.3, max_tokens=6000)
        features = [Feature(**_normalize_feature(f)) for f in data.get("features", [])]
        return AnalysisReport(
            domain=data.get("domain", ""),
            essence=data.get("essence", ""),
            audience=data.get("audience", ""),
            competitors=list(data.get("competitors", [])),
            features=features,
            ui_elements=list(data.get("ui_elements", [])),
            raw_notes=data.get("raw_notes", ""),
        )


def _normalize_feature(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(f.get("name", "")).strip() or "unnamed",
        "category": str(f.get("category", "other")).strip().lower(),
        "description": str(f.get("description", "")).strip(),
        "source": str(f.get("source", "")).strip(),
    }


# ============ Planner =========================================================

PLANNER_SYS = """You are the Planner agent. You are given:
  - the analyzer's report, INCLUDING the demo's `essence` and `audience` which
    must NEVER change,
  - the existing project's file listing AND key source files in full,
  - a SYMBOL MAP listing classes/functions/routes already defined,
  - open QA bug reports (if any),
  - previous iteration results.

HARD RULES (in priority order):

1. PRESERVE ESSENCE. Every task must respect `analysis.essence` and
   `analysis.audience`. If a competitor feature would change who the demo is
   for (e.g. turning an Agent-vs-Agent harness into a human PvP web game),
   reject it or translate it into an essence-preserving analogue (e.g.
   instead of "human lobby with chat", propose "agent-vs-agent benchmark
   harness with structured channels").

2. PROJECT IS NOT EMPTY. Read the symbol map. Do not propose to re-implement
   anything already present under a different name. Extend the existing
   symbol rather than create a parallel one.

3. BUGS FIRST. If there are open QA bug reports, the highest-priority task(s)
   MUST be fixing them. Feature tasks come after.

Decide the next concrete, file-level tasks. Prefer SMALL, INDEPENDENT tasks
that can run in parallel. For large existing files, instruct the executor to
use Mode B (SEARCH/REPLACE patches), not full rewrite.

Output STRICT JSON only.
"""

PLANNER_USER_TMPL = """Analyzer report:
{analysis}

Project file listing:
{listing}

Key source files (full contents shown for the most load-bearing ones):
{key_files}

Symbol map (path -> [classes/functions/routes]):
{symbol_map}

Open bug reports (failing QA tests from previous iterations):
{open_bugs}

Previous iteration results (most recent first):
{history}

Iteration: {iteration} / {max_iter}

Return a JSON object:
{{
  "rationale": "<2-3 sentences why these tasks now>",
  "tasks": [
    {{
      "title": "...",
      "rationale": "...",
      "target_files": ["relative/path", ...],
      "instructions": "<concrete instructions for the executor>",
      "priority": 1,
      "category": "feature|bugfix|ux|docs|infra"
    }}
  ]
}}
Constraints:
- 3 to {max_tasks} tasks.
- Every task instruction must restate which aspect of `essence` it preserves.
- target_files must be paths inside the project. Use new paths for new files.
- Mention specific existing symbols you want to extend (e.g. "extend GameMaster.vote_phase").
- For files > 200 lines, instructions must say "use Mode B SEARCH/REPLACE".
- If there are open bug reports, the highest-priority task MUST be fixing one of them.
- Skip tasks already attempted-and-done in history.
"""


class Planner:
    KEY_FILE_CANDIDATES = (
        "README.md", "README", "readme.md",
        "main.py", "app.py", "server.py", "index.js", "index.ts",
        "src/main.ts", "src/index.ts", "src/App.tsx",
        "package.json", "pyproject.toml", "requirements.txt",
        "Cargo.toml",
    )

    def __init__(self, llm: MiniMaxClient, sandbox: Sandbox, *,
                 max_tasks: int = 5) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.max_tasks = max_tasks

    def _pick_key_files(self, listing: list[str]) -> list[str]:
        seen: list[str] = []
        # explicit candidates first
        for c in self.KEY_FILE_CANDIDATES:
            if c in listing and c not in seen:
                seen.append(c)
        # then the largest source files (so we don't miss e.g. game.py)
        sizes: list[tuple[int, str]] = []
        for p in listing:
            if not p.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs")):
                continue
            try:
                sz = len(self.sandbox.read(p))
            except Exception:
                continue
            sizes.append((sz, p))
        sizes.sort(reverse=True)
        for _, p in sizes[:4]:
            if p not in seen:
                seen.append(p)
        return seen[:8]

    def _build_key_files_block(self, key_files: list[str]) -> str:
        chunks = []
        for p in key_files:
            txt = self.sandbox.read(p)
            if not txt:
                continue
            chunks.append(f"=== {p} ===\n{txt[:5000]}")
        return "\n\n".join(chunks) or "(none)"

    def run(self, analysis: AnalysisReport, *, iteration: int, max_iter: int,
            history: list[dict[str, Any]],
            open_bugs: list[dict[str, Any]] | None = None) -> PlanResult:
        listing_raw = self.sandbox.listing(max_entries=200)
        listing_str = "\n".join(listing_raw)
        # strip the trailing "... (truncated)" marker before symbol/file picking
        listing = [p for p in listing_raw if not p.startswith("...")]
        key_files = self._pick_key_files(listing)
        key_files_block = self._build_key_files_block(key_files)
        symbol_map = build_symbol_map(self.sandbox.read, listing)
        user = PLANNER_USER_TMPL.format(
            analysis=_json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2),
            listing=listing_str,
            key_files=key_files_block,
            symbol_map=_json.dumps(symbol_map, ensure_ascii=False, indent=2),
            open_bugs=_json.dumps(open_bugs or [], ensure_ascii=False, indent=2),
            history=_json.dumps(history[-3:], ensure_ascii=False, indent=2) if history else "(none)",
            iteration=iteration,
            max_iter=max_iter,
            max_tasks=self.max_tasks,
        )
        data = self.llm.chat_json(PLANNER_SYS, user, temperature=0.3, max_tokens=6000)
        tasks = []
        for t in data.get("tasks", [])[: self.max_tasks]:
            tasks.append(Task(
                id=uuid.uuid4().hex[:8],
                title=str(t.get("title", "")).strip() or "untitled",
                rationale=str(t.get("rationale", "")).strip(),
                target_files=[str(x) for x in t.get("target_files", []) if x],
                instructions=str(t.get("instructions", "")).strip(),
                priority=int(t.get("priority", 5) or 5),
                category=str(t.get("category", "feature")).strip().lower(),
            ))
        return PlanResult(iteration=iteration, tasks=tasks,
                          rationale=str(data.get("rationale", "")))


# ============ Executor ========================================================

EXECUTOR_SYS = """You are the Executor agent. You receive ONE task and the
current contents of the files it targets. You output edits as delimited
plain-text blocks — NEVER JSON.

You have TWO editing modes; pick whichever is appropriate per file:

==== Mode A — full file rewrite =================================================

Use only when the file is new, or short (< ~200 lines). Emit the complete
new contents verbatim:

===FILE: relative/path.ext===
<entire new file contents>
===END===

==== Mode B — surgical patch (PREFERRED for large existing files) ==============

For files over ~200 lines, edit by SEARCH/REPLACE pairs. The SEARCH block must
match the existing file byte-for-byte (whitespace included). You can include
many SEARCH/REPLACE pairs per file.

===PATCH: relative/path.ext===
<<<SEARCH
<exact existing text, including indentation>
SEARCH>>>
<<<REPLACE
<new text>
REPLACE>>>

<<<SEARCH
<another existing snippet>
SEARCH>>>
<<<REPLACE
<replacement>
REPLACE>>>
===END===

To INSERT new code, SEARCH for the nearest unique anchor and REPLACE with
"anchor + new code". To DELETE code, REPLACE with empty body.

==== Output frame ==============================================================

STATUS: done
SUMMARY: <one short line>

<one or more ===FILE=== and/or ===PATCH=== blocks>

If you cannot or should not do the task:

STATUS: skipped
SUMMARY: <why>

Hard rules:
- Only modify files inside `target_files`, plus you MAY create at most one
  extra helper file if essential.
- Preserve the existing tech stack and style.
- Do not wrap blocks in markdown code fences.
- Never write "// ... rest unchanged" — emit Mode B patches for partial edits.
"""

EXECUTOR_USER_TMPL = """Task:
{task}

Current files (FULL contents shown; empty means the file does not exist yet):
{files_block}

Project file listing (for context):
{listing}

Pick Mode A for new/short files, Mode B for large existing files. Output now.
"""


class Executor:
    def __init__(self, llm: MiniMaxClient, sandbox: Sandbox,
                 adapter: Optional[LanguageAdapter] = None) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.adapter = adapter or adapter_for(detect_primary_language(sandbox))

    def run(self, task: Task, *,
            post_check: Optional[Callable[[], Tuple[bool, str]]] = None,
            ) -> ExecutionResult:
        files_block_parts = []
        for rel in task.target_files[:6]:
            current = self.sandbox.read(rel)
            files_block_parts.append(
                f"=== {rel} ===\n{current if current else '(file does not exist yet)'}"
            )
        files_block = "\n\n".join(files_block_parts) or "(no target files specified)"
        listing = "\n".join(self.sandbox.listing(max_entries=80))

        user = EXECUTOR_USER_TMPL.format(
            task=_json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
            files_block=files_block,
            listing=listing,
        )
        try:
            raw = self.llm.chat(EXECUTOR_SYS, user, temperature=0.2,
                                max_tokens=16000)
        except Exception as e:
            return ExecutionResult(task_id=task.id, status="failed",
                                   summary="", error=f"executor LLM error: {e}")

        parsed = parse_executor_output(raw)
        status = parsed["status"]
        summary = parsed["summary"]
        if status == "skipped":
            return ExecutionResult(task_id=task.id, status="skipped",
                                   summary=summary or "model skipped")

        changed: list[str] = []
        rejected: list[str] = []
        forbidden = {f.strip() for f in (task.forbidden_files or []) if f.strip()}

        # Apply patch ops first so they see original file content.
        for rel, ops in parsed["patches"]:
            if rel in forbidden:
                rejected.append(f"{rel}: forbidden (test file, read-only)")
                continue
            existing = self.sandbox.read(rel)
            if not existing:
                rejected.append(f"{rel}: patch target does not exist")
                continue
            new_content, miss = _apply_search_replace(existing, ops)
            if miss:
                retry_content, retry_miss = self._retry_patch(
                    rel=rel, file_content=existing, misses=[miss], task=task,
                )
                if retry_miss:
                    rejected.append(
                        f"{rel}: SEARCH not found after retry: {miss[:80]!r}"
                    )
                    continue
                new_content = retry_content
            # Apply destructive-shrink guard to PATCH outputs too — many small
            # SEARCH/REPLACE ops can collectively delete most of a file even
            # though each op looks innocuous (cf. player.py 108→28 incident).
            reject = _guard_destructive_write(rel, existing, new_content)
            if reject:
                rejected.append(f"{rel}: {reject}")
                continue
            try:
                self.sandbox.write(rel, new_content)
            except Exception as e:
                return ExecutionResult(task_id=task.id, status="failed",
                                       summary=summary,
                                       error=f"write {rel}: {e}",
                                       files_changed=changed)
            syntax_err = _post_write_syntax_check(self.sandbox, rel, self.adapter)
            if syntax_err:
                healed = self._self_heal(rel, syntax_err, task)
                if healed:
                    changed.append(rel)
                else:
                    self.sandbox.write(rel, existing)
                    rejected.append(f"{rel}: {syntax_err} (self-heal also failed)")
            else:
                changed.append(rel)

        # Then full-file rewrites.
        for rel, content in parsed["files"]:
            if rel in forbidden:
                rejected.append(f"{rel}: forbidden (test file, read-only)")
                continue
            existing = self.sandbox.read(rel)
            reject = _guard_destructive_write(rel, existing, content)
            if reject:
                rejected.append(f"{rel}: {reject}")
                continue
            try:
                written = self.sandbox.write(rel, content)
            except Exception as e:
                return ExecutionResult(task_id=task.id, status="failed",
                                       summary=summary,
                                       error=f"write {rel}: {e}",
                                       files_changed=changed)
            syntax_err = _post_write_syntax_check(self.sandbox, written)
            if syntax_err:
                healed = self._self_heal(written, syntax_err, task)
                if healed:
                    changed.append(written)
                else:
                    if existing:
                        self.sandbox.write(rel, existing)
                    else:
                        self.sandbox.delete(rel)
                    rejected.append(f"{rel}: {syntax_err} (self-heal also failed)")
            else:
                changed.append(written)

        if rejected and not changed:
            return ExecutionResult(task_id=task.id, status="failed",
                                   summary=summary,
                                   error="; ".join(rejected))
        if not changed:
            return ExecutionResult(task_id=task.id, status="skipped",
                                   summary=summary or "no files produced")
        result = ExecutionResult(task_id=task.id, status="done",
                                 summary=summary, files_changed=changed)
        if rejected:
            result.error = "partial; rejected: " + "; ".join(rejected)

        # Test-driven post-check loop (used by QA fix tasks). Up to N attempts;
        # each retry gets the extracted failing-assertion summary so the model
        # narrows in on what the test actually wants. Replaces the single-retry
        # design that only fixed bugs ~10% of the time across 10 iters.
        if post_check is not None and result.status == "done":
            for attempt in range(1, self.MAX_FIX_ATTEMPTS + 1):
                ok, output = post_check()
                if ok:
                    return result
                if attempt >= self.MAX_FIX_ATTEMPTS:
                    _apply_post_check_to_result(
                        result, post_check_ok=False, post_check_output=output)
                    break
                pinpoint = _extract_assertion_summary(output)
                retry_task = Task(
                    id=f"{task.id}-retry{attempt}",
                    title=task.title, rationale=task.rationale,
                    target_files=task.target_files,
                    instructions=task.instructions + (
                        f"\n\n=== RETRY ATTEMPT {attempt}/{self.MAX_FIX_ATTEMPTS - 1} ===\n"
                        f"The test STILL fails. Most-actionable error line:\n  {pinpoint}\n\n"
                        f"Full test-output tail:\n{(output or '')[-1200:]}\n\n"
                        f"Pinpoint which function/branch in your previous edit "
                        f"does NOT satisfy this assertion, and rewrite ONLY "
                        f"that piece. Smaller, more precise diff."
                    ),
                    priority=task.priority, category=task.category,
                    forbidden_files=task.forbidden_files,
                )
                retry_result = self.run(retry_task)
                if retry_result.status != "done":
                    # the retry write itself failed (search-miss / syntax /
                    # destructive-shrink). No point continuing — surface
                    # the original failure.
                    _apply_post_check_to_result(
                        result, post_check_ok=False, post_check_output=output)
                    break
                result.files_changed = list(dict.fromkeys(
                    result.files_changed + retry_result.files_changed))
                result.summary = (result.summary + f" | retry{attempt}: " +
                                  retry_result.summary)[:240]
        return result

    # number of times the fix Executor tries to make a failing test green.
    # Each attempt is one Executor.run() + post_check call. Latency cost is
    # linear in this; 3 is a good sweet spot from the data.
    MAX_FIX_ATTEMPTS = 3

    # ---- patch retry on SEARCH miss -----------------------------------------

    def _retry_patch(self, *, rel: str, file_content: str,
                     misses: list[str], task: Task) -> tuple[str, str]:
        """One-shot retry when SEARCH text wasn't found. Returns (new_content, miss).
        miss is empty on success.
        """
        user = _format_patch_retry_user(rel, file_content, misses, task.title)
        try:
            raw = self.llm.chat(PATCH_RETRY_SYS, user, temperature=0.1,
                                max_tokens=8000)
        except Exception:
            return file_content, misses[0] if misses else "retry-llm-failed"
        parsed = parse_executor_output(raw)
        for path, ops in parsed["patches"]:
            if path != rel:
                continue
            new_content, miss = _apply_search_replace(file_content, ops)
            return new_content, miss
        return file_content, misses[0] if misses else "retry-no-patch"

    # ---- self-heal -----------------------------------------------------------

    SELF_HEAL_SYS = (
        "You are the Self-Heal sub-agent. Your previous write produced a "
        "syntactically-invalid file. Fix ONLY the syntax error. Output the "
        "complete new file contents in a single ===FILE=== block — no patches, "
        "no commentary."
    )

    def _self_heal(self, rel: str, syntax_err: str, task: Task) -> bool:
        """Ask the model to fix a syntax error in the file we just wrote.
        Returns True if the heal succeeded (file now parses); False otherwise.
        """
        broken = self.sandbox.read(rel)
        if not broken or not rel.endswith(".py"):
            return False
        numbered = _with_line_numbers(broken)
        user = (
            f"File: {rel}\n"
            f"Syntax error: {syntax_err}\n"
            f"Original task: {task.title}\n\n"
            f"Current (broken) contents with line numbers:\n{numbered}\n\n"
            f"Re-emit the entire file with the syntax fixed. Use:\n"
            f"===FILE: {rel}===\n<contents>\n===END===\n"
        )
        try:
            raw = self.llm.chat(self.SELF_HEAL_SYS, user, temperature=0.1,
                                max_tokens=16000)
        except Exception as e:
            log.warning("self-heal LLM error on %s: %s", rel, e)
            return False
        parsed = parse_executor_output(raw)
        for path, content in parsed["files"]:
            if path != rel:
                continue
            # don't undo work — guard against destructive shrink during heal
            if _guard_destructive_write(rel, broken, content):
                continue
            self.sandbox.write(rel, content)
            if not _post_write_syntax_check(self.sandbox, rel, self.adapter):
                log.info("self-heal succeeded for %s", rel)
                return True
            # heal made it worse — restore broken state so outer rollback works
            self.sandbox.write(rel, broken)
            return False
        return False


PATCH_RETRY_SYS = (
    "Your previous patch SEARCH text was not found in the file. You will be "
    "shown the actual file (with line numbers) and the failing SEARCH texts. "
    "Emit a single fresh ===PATCH=== block whose SEARCH texts are copied "
    "byte-for-byte from the file. Do not invent code that is not present."
)


def _format_patch_retry_user(rel: str, file_content: str,
                             misses: list[str], original_task_title: str) -> str:
    misses_block = "\n\n".join(f"<<<MISS>>>\n{m}\n<<<END>>>" for m in misses)
    numbered = _with_line_numbers(file_content)
    return (
        f"Original task: {original_task_title}\n"
        f"File: {rel}\n\n"
        f"Misses (none of these were found verbatim in {rel}):\n{misses_block}\n\n"
        f"Actual current content of {rel}:\n{numbered}\n\n"
        f"Emit ONE ===PATCH: {rel}=== block with corrected SEARCH/REPLACE pairs."
    )


def _with_line_numbers(text: str, max_chars: int = 40000) -> str:
    out_lines = []
    total = 0
    for i, line in enumerate(text.splitlines(), 1):
        rendered = f"{i:4d}| {line}"
        total += len(rendered) + 1
        if total > max_chars:
            out_lines.append("... (truncated)")
            break
        out_lines.append(rendered)
    return "\n".join(out_lines)


def _guard_destructive_write(rel: str, existing: str, new: str) -> str:
    if not existing:
        return ""
    old_lines = existing.count("\n") + 1
    new_lines = new.count("\n") + 1
    if old_lines >= 80 and new_lines < max(40, old_lines * 0.4):
        return (
            f"destructive write blocked: {rel} would shrink from "
            f"{old_lines} -> {new_lines} lines"
        )
    return ""


_ASSERT_LINE_RE = _re.compile(
    r"^(?:[A-Z]\w*(?:Error|Exception):.+|FAIL(?:ED)?:\s.+|assert.+|"
    r"AssertionError.+|FAILED \(.+\))$",
    _re.MULTILINE,
)


def _extract_assertion_summary(test_output: str, max_chars: int = 240) -> str:
    """Pull the single most actionable line from a noisy test runner output.

    Models do dramatically better with one sharp 'AssertionError: 1 != 2 at
    line 42' line than with 1500 chars of traceback noise.
    """
    if not test_output:
        return "(no test output)"
    # try the structured-error regex first
    candidates = _ASSERT_LINE_RE.findall(test_output)
    # prefer the LAST match (deepest in traceback / final summary)
    if candidates:
        return candidates[-1].strip()[:max_chars]
    # fallback — last non-blank line
    lines = [l for l in test_output.splitlines() if l.strip()]
    return (lines[-1] if lines else "(empty)")[:max_chars]


def _apply_post_check_to_result(res: ExecutionResult, *,
                                post_check_ok: bool,
                                post_check_output: str) -> ExecutionResult:
    """Demote `done` -> `failed` and append the test output when post_check fails."""
    if post_check_ok:
        return res
    tail = (post_check_output or "")[-1200:]
    addendum = f"post-check failed: {tail}"
    res.error = (res.error + " | " + addendum) if res.error else addendum
    if res.status == "done":
        res.status = "failed"
    return res


def _post_write_syntax_check(sandbox: Sandbox, rel: str,
                              adapter: Optional[LanguageAdapter] = None) -> str:
    """Delegates to the language adapter; legacy callers (tests) pass no
    adapter and we still do a Python-only ast.parse as the safety net."""
    if adapter is not None:
        return adapter.syntax_check(sandbox, rel)
    # legacy fast-path for unit tests that exercise this helper directly
    if not rel.endswith(".py"):
        return ""
    try:
        ast.parse(sandbox.read(rel))
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"
    return ""


def _apply_search_replace(content: str, ops: list[tuple[str, str]]) -> tuple[str, str]:
    """Apply SEARCH/REPLACE ops left-to-right. Returns (new_content, miss_snippet).
    miss_snippet is empty on success, otherwise it is the failing SEARCH text.

    Two-pass matching:
      1. exact byte-for-byte (preferred — fully unambiguous)
      2. whitespace-tolerant line-anchored fallback:
         - trailing whitespace stripped per line
         - leading whitespace allowed to differ uniformly (re-indented blocks)
         - blank lines collapsed
         Falls back ONLY if the match is unique; ambiguous matches are refused.
    """
    current = content
    for search, replace in ops:
        if not search:
            continue
        if search in current:
            current = current.replace(search, replace, 1)
            continue
        # fuzzy line-anchored fallback
        match = _fuzzy_locate(current, search)
        if match is None:
            return current, search
        start, end, _ = match
        # preserve the original block's leading indent shift so REPLACE lines up
        original_block = current[start:end]
        adjusted_replace = _reindent_to(original_block, search, replace)
        current = current[:start] + adjusted_replace + current[end:]
    return current, ""


def _fuzzy_locate(haystack: str, needle: str) -> tuple[int, int, int] | None:
    """Find `needle` in `haystack` tolerating leading whitespace differences.
    Returns (start, end, indent_shift) or None. Returns None on ambiguity.
    """
    needle_lines = [ln.rstrip() for ln in needle.splitlines()]
    # drop leading/trailing blank lines from needle
    while needle_lines and not needle_lines[0].strip():
        needle_lines.pop(0)
    while needle_lines and not needle_lines[-1].strip():
        needle_lines.pop()
    if not needle_lines:
        return None

    # compute relative indents within needle (first non-blank line = 0)
    base_indent = len(needle_lines[0]) - len(needle_lines[0].lstrip())
    needle_rel = [(len(ln) - len(ln.lstrip()) - base_indent, ln.lstrip())
                  for ln in needle_lines]

    hay_lines = haystack.splitlines(keepends=True)
    line_offsets = [0]
    for ln in hay_lines:
        line_offsets.append(line_offsets[-1] + len(ln))

    matches: list[tuple[int, int, int]] = []
    n = len(needle_rel)
    for i in range(len(hay_lines) - n + 1):
        first = hay_lines[i].rstrip("\n").rstrip()
        if not first.lstrip() or first.lstrip() != needle_rel[0][1]:
            continue
        hay_base = len(hay_lines[i]) - len(hay_lines[i].lstrip())
        ok = True
        for j in range(1, n):
            cand = hay_lines[i + j].rstrip("\n").rstrip()
            cand_rel_indent = (len(cand) - len(cand.lstrip())) - hay_base
            want_indent, want_text = needle_rel[j]
            if cand.lstrip() != want_text or cand_rel_indent != want_indent:
                ok = False
                break
        if ok:
            start = line_offsets[i]
            end = line_offsets[i + n]
            matches.append((start, end, hay_base - base_indent))
            if len(matches) > 1:
                return None  # ambiguous — refuse
    return matches[0] if len(matches) == 1 else None


def _reindent_to(original_block: str, search: str, replace: str) -> str:
    """When the haystack block was at a different indent than SEARCH, shift
    REPLACE by the same delta so it slots in cleanly."""
    orig_first = original_block.splitlines()[0] if original_block else ""
    search_first = next((ln for ln in search.splitlines() if ln.strip()), "")
    delta = ((len(orig_first) - len(orig_first.lstrip()))
             - (len(search_first) - len(search_first.lstrip())))
    if delta == 0:
        return replace
    sign = " " * delta if delta > 0 else None
    out_lines = []
    for ln in replace.splitlines(keepends=True):
        if sign:
            out_lines.append(sign + ln if ln.strip() else ln)
        else:
            # delta < 0: dedent up to |delta| spaces if present
            shave = -delta
            stripped = ln[shave:] if ln[:shave].isspace() else ln
            out_lines.append(stripped)
    return "".join(out_lines)


# ============ Output parser ===================================================

_FILE_BLOCK_RE = _re.compile(
    r"===FILE:\s*(?P<path>[^=\n]+?)\s*===\s*\n(?P<body>.*?)(?:\n===END===|\Z)",
    _re.DOTALL,
)
_PATCH_BLOCK_RE = _re.compile(
    r"===PATCH:\s*(?P<path>[^=\n]+?)\s*===\s*\n(?P<body>.*?)(?:\n===END===|\Z)",
    _re.DOTALL,
)
_SR_PAIR_RE = _re.compile(
    r"<<<\s*SEARCH\s*\n(?P<search>.*?)\n\s*SEARCH>>>\s*\n"
    r"<<<\s*REPLACE\s*\n?(?P<replace>.*?)\n?\s*REPLACE>>>",
    _re.DOTALL,
)


def parse_executor_output(text: str) -> dict[str, Any]:
    text = text or ""
    status_m = _re.search(r"^\s*STATUS:\s*(\w+)", text, _re.MULTILINE)
    summary_m = _re.search(r"^\s*SUMMARY:\s*(.+)$", text, _re.MULTILINE)
    status = (status_m.group(1).strip().lower() if status_m else "done")
    summary = (summary_m.group(1).strip() if summary_m else "")

    files: list[tuple[str, str]] = []
    for m in _FILE_BLOCK_RE.finditer(text):
        path = m.group("path").strip().strip("`'\"")
        body = m.group("body")
        if body.startswith("\n"):
            body = body[1:]
        body = _strip_outer_fence(body)
        if path:
            files.append((path, body))

    patches: list[tuple[str, list[tuple[str, str]]]] = []
    for m in _PATCH_BLOCK_RE.finditer(text):
        path = m.group("path").strip().strip("`'\"")
        body = m.group("body")
        ops = [(sm.group("search"), sm.group("replace"))
               for sm in _SR_PAIR_RE.finditer(body)]
        if path and ops:
            patches.append((path, ops))

    if status not in {"done", "skipped", "failed"}:
        status = "done" if (files or patches) else "skipped"
    if status == "done" and not files and not patches:
        status = "skipped"
    return {"status": status, "summary": summary,
            "files": files, "patches": patches}


def _strip_outer_fence(body: str) -> str:
    stripped = body.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped[3:-3]
        nl = inner.find("\n")
        if nl != -1 and inner[:nl].strip().isalnum():
            inner = inner[nl + 1 :]
        return inner.rstrip() + "\n"
    return body
