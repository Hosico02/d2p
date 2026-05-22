"""Executor agent — receives ONE Task, emits ===FILE=== / ===PATCH=== blocks.

Editing modes:
- Mode A: full file rewrite (for new or short files).
- Mode B: SEARCH/REPLACE patches (preferred for large existing files).

Pipeline:
- parse executor output into (files, patches),
- apply patches (with fuzzy-locate fallback, then one LLM-driven retry on
  SEARCH miss),
- run destructive-shrink guard on every write,
- post-write syntax check; on syntax error invoke self-heal (one shot)
  before giving up,
- optional post_check callback (used by QA fix tasks) that runs the bug's
  failing test up to MAX_FIX_ATTEMPTS times with extracted assertion
  pinpointing.

All search/replace + parser helpers live in this module; they're
re-exported from `d2p.agents` for the unit tests.
"""
from __future__ import annotations

import ast
import json as _json
import logging
import re as _re
from typing import Any, Callable, Optional, Tuple

from dataclasses import dataclass, field

from ..fs import Sandbox
from ..lang import LanguageAdapter, adapter_for, detect_primary_language
from ..providers.base import LLMProvider
from ..models import ExecutionResult, Task


@dataclass
class PreparedExecution:
    """Carries the LLM-call output across the prepare → commit boundary.

    The two-phase split lets the orchestrator do the slow LLM call OUTSIDE
    the per-file lock while keeping the actual writes serialised. Two
    tasks targeting the same file can now both spend their LLM time in
    parallel and only contend on the (~ms) write phase.

    The `source_snapshot` records what each target file looked like AT
    PREPARE TIME. commit() compares against the file's current state to
    detect concurrent modifications (the cost of moving LLM out of the
    lock). PATCH-mode handles concurrency naturally via SEARCH/REPLACE
    against current content; FILE-mode refuses to clobber a changed file.
    """
    task: Task
    parsed: dict[str, Any]              # parse_executor_output(...) result
    status: str                          # "done" / "skipped" / "failed"
    summary: str
    source_snapshot: dict[str, str] = field(default_factory=dict)
    llm_error: Optional[str] = None     # set if the LLM call itself failed

log = logging.getLogger("d2p.agents.executor")


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
    # number of times the fix Executor tries to make a failing test green.
    # Each attempt is one Executor.run() + post_check call. Latency cost is
    # linear in this; 3 is a good sweet spot from the data.
    MAX_FIX_ATTEMPTS = 3

    SELF_HEAL_SYS = (
        "You are the Self-Heal sub-agent. Your previous write produced a "
        "syntactically-invalid file. Fix ONLY the syntax error. Output the "
        "complete new file contents in a single ===FILE=== block — no patches, "
        "no commentary."
    )

    def __init__(self, llm: LLMProvider, sandbox: Sandbox,
                 adapter: Optional[LanguageAdapter] = None,
                 usage: Any = None,
                 heal_llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.adapter = adapter or adapter_for(detect_primary_language(sandbox))
        # Optional UsageAccumulator handle so _self_heal can bump
        # 'self_heal_attempts' / 'self_heal_succeeded' counters that show
        # up in summary.json. If None (legacy callers / tests), counters
        # are a no-op.
        self.usage = usage
        # Optional stronger model for the 2nd self-heal attempt. The
        # orchestrator passes router.for_role_tier(role, tier_idx + 1)
        # when a next tier exists; None when this task is already at the
        # top tier (in which case the 2nd attempt reuses self.llm).
        self.heal_llm = heal_llm

    # ---- prepare/commit split -----------------------------------------------
    # The orchestrator calls `prepare` OUTSIDE the per-file lock (slow LLM
    # call) and `commit` INSIDE the lock (cheap writes). For one-shot
    # callers / unit tests, `run` is the combined wrapper.

    def prepare(self, task: Task) -> "PreparedExecution":
        """Phase 1: read target files, call the LLM, parse the output.
        No sandbox writes. Safe to call concurrently without locks because
        sandbox.read is read-only.
        """
        files_block_parts = []
        source_snapshot: dict[str, str] = {}
        for rel in task.target_files[:6]:
            current = self.sandbox.read(rel)
            source_snapshot[rel] = current
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
            return PreparedExecution(
                task=task, parsed={}, status="failed", summary="",
                source_snapshot=source_snapshot,
                llm_error=f"executor LLM error: {e}",
            )

        parsed = parse_executor_output(raw)
        return PreparedExecution(
            task=task, parsed=parsed,
            status=parsed["status"], summary=parsed["summary"],
            source_snapshot=source_snapshot,
        )

    def commit(self, prepared: "PreparedExecution", *,
               post_check: Optional[Callable[[], Tuple[bool, str]]] = None,
               max_fix_attempts: Optional[int] = None,
               ) -> ExecutionResult:
        """Phase 2: apply writes, run syntax check, self-heal, post_check.
        Caller MUST hold the per-file lock(s) for `prepared.task.target_files`.

        Concurrent-modification handling: if a FILE-mode target's current
        content differs from the snapshot captured at prepare time, we
        refuse the write (it would clobber another task's parallel edit).
        PATCH-mode targets are safe because SEARCH/REPLACE runs against
        the CURRENT file content — either the anchors are still present
        (apply cleanly) or they're not (we get a miss and the retry path
        handles it)."""
        task = prepared.task
        if prepared.llm_error:
            return ExecutionResult(task_id=task.id, status="failed",
                                   summary="", error=prepared.llm_error)

        status = prepared.status
        summary = prepared.summary
        parsed = prepared.parsed
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
                retry_ok, retry_content, retry_err = self._retry_destructive(
                    rel=rel, existing=existing, attempted=new_content, task=task,
                )
                if retry_ok:
                    log.info("destructive-write recovered via small-patch "
                             "retry on %s", rel)
                    new_content = retry_content
                else:
                    rejected.append(f"{rel}: {reject} (retry: {retry_err})")
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
            # Concurrent-modification check: FILE-mode would clobber any
            # write a parallel task made between our prepare() and commit().
            # If the file changed since we read it during prepare, refuse —
            # the orchestrator will surface "concurrent modification" and
            # the user/Planner can re-run.
            snapshot_at_prepare = prepared.source_snapshot.get(rel)
            if (snapshot_at_prepare is not None
                    and existing != snapshot_at_prepare):
                rejected.append(
                    f"{rel}: concurrent modification — file changed between "
                    f"LLM prepare and write (FILE-mode refused; "
                    f"consider re-running with PATCH mode)"
                )
                continue
            reject = _guard_destructive_write(rel, existing, content)
            if reject:
                retry_ok, retry_content, retry_err = self._retry_destructive(
                    rel=rel, existing=existing, attempted=content, task=task,
                )
                if retry_ok:
                    log.info("destructive-write (FILE-mode) recovered via "
                             "small-patch retry on %s", rel)
                    content = retry_content
                else:
                    rejected.append(f"{rel}: {reject} (retry: {retry_err})")
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
            cap = max_fix_attempts if max_fix_attempts is not None \
                else self.MAX_FIX_ATTEMPTS
            cap = max(1, cap)
            # Accumulate every attempt's summary so each retry sees the full
            # trail of "what we tried that didn't work". Without this the
            # model often re-tries the same edit it just made.
            attempt_log: list[str] = [f"attempt 0 (initial): {result.summary[:160]}"]
            for attempt in range(1, cap + 1):
                ok, output = post_check()
                if ok:
                    return result
                if attempt >= cap:
                    _apply_post_check_to_result(
                        result, post_check_ok=False, post_check_output=output)
                    break
                pinpoint = _extract_assertion_summary(output)
                # Snapshot current file contents so the retry sees the ACTUAL
                # post-edit state of the SUT — kills anchor hallucination and
                # lets the model reason about what its own previous patch did.
                current_state = _format_current_files(
                    self.sandbox, task.target_files, max_chars_per_file=4000)
                retry_task = Task(
                    id=f"{task.id}-retry{attempt}",
                    title=task.title, rationale=task.rationale,
                    target_files=task.target_files,
                    instructions=task.instructions + (
                        f"\n\n=== RETRY ATTEMPT {attempt}/{cap - 1} ===\n"
                        f"The test STILL fails. Most-actionable error line:\n  {pinpoint}\n\n"
                        f"Prior attempts on this task:\n" +
                        "\n".join(f"  - {line}" for line in attempt_log) + "\n\n"
                        f"Current state of target files (after your previous edits):\n"
                        f"{current_state}\n\n"
                        f"Full test-output tail:\n{(output or '')[-1500:]}\n\n"
                        f"Look at the test assertion + your file state above. "
                        f"Pinpoint which function/branch does NOT satisfy "
                        f"the assertion, and rewrite ONLY that piece. "
                        f"Smaller, more precise diff."
                    ),
                    priority=task.priority, category=task.category,
                    forbidden_files=task.forbidden_files,
                )
                retry_result = self.run(retry_task)
                attempt_log.append(
                    f"attempt {attempt} ({retry_result.status}): "
                    f"{(retry_result.summary or retry_result.error or '?')[:160]}"
                )
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

    def run(self, task: Task, *,
            post_check: Optional[Callable[[], Tuple[bool, str]]] = None,
            ) -> ExecutionResult:
        """Backward-compat wrapper: prepare + commit in one call. Useful for
        callers that don't need lock-free parallelism (tests, one-off
        invocations). The orchestrator calls prepare/commit separately to
        run the LLM portion outside the per-file lock."""
        prepared = self.prepare(task)
        return self.commit(prepared, post_check=post_check)

    # ---- patch retry on SEARCH miss -----------------------------------------

    def _retry_destructive(self, *, rel: str, existing: str,
                           attempted: str, task: Task,
                           ) -> tuple[bool, str, str]:
        """Recovery for a blocked destructive rewrite.

        Tells the model explicitly that its previous output would have
        shrunk the file from X to Y lines, shows the real file with line
        numbers, and demands a small SEARCH/REPLACE patch. Returns
        (ok, new_content, error_msg). On failure new_content is unchanged.
        """
        old_lines = existing.count("\n") + 1
        new_lines = attempted.count("\n") + 1
        user = (
            f"File: {rel}\n"
            f"Task: {task.title}\n\n"
            f"Your previous output would have shrunk this file from "
            f"{old_lines} to {new_lines} lines. That is destructive and "
            f"was BLOCKED. You must preserve the existing structure.\n\n"
            f"Actual current file with line numbers:\n"
            f"{_with_line_numbers(existing)}\n\n"
            f"Original task instructions:\n{task.instructions[:2000]}\n\n"
            f"Emit ONE small ===PATCH=== block that makes the smallest "
            f"possible change to satisfy the task. SEARCH text must match "
            f"the file above byte-for-byte. Do NOT emit a full ===FILE=== "
            f"rewrite — small patch only."
        )
        try:
            raw = self.llm.chat(DESTRUCTIVE_RETRY_SYS, user,
                                temperature=0.1, max_tokens=8000)
        except Exception as e:
            return False, existing, f"llm-error: {_re.sub(chr(10), ' ', str(e))[:120]}"
        parsed = parse_executor_output(raw)
        for path, ops in parsed["patches"]:
            if path != rel:
                continue
            new_content, miss = _apply_search_replace(existing, ops)
            if miss:
                return False, existing, f"search-miss: {miss[:80]!r}"
            if _guard_destructive_write(rel, existing, new_content):
                return False, existing, "retry-still-destructive"
            return True, new_content, ""
        return False, existing, "no-patch-emitted"

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

    def _self_heal(self, rel: str, syntax_err: str, task: Task) -> bool:
        """Up to two attempts at fixing a syntax error in `rel`:
          1. Same-tier (self.llm). Cheap, often enough.
          2. Next-tier (self.heal_llm) when configured — typically the
             orchestrator passes tier_idx+1's provider, so a haiku-induced
             error gets a sonnet rescue, a sonnet-induced error gets opus.

        Returns True iff the file parses cleanly after one of the attempts.
        Failure-error from attempt 1 is fed back into attempt 2 so the
        stronger model knows what specifically didn't work.
        """
        broken = self.sandbox.read(rel)
        if not broken or not rel.endswith(".py"):
            return False
        # Attempt 1: same-tier.
        if self._self_heal_one(rel, syntax_err, task, llm=self.llm,
                               prior_failure=None):
            return True
        # Re-read the file — attempt 1 may have written something broken
        # and then restored, OR left broken state if it never made it that
        # far. Either way `broken` could be stale.
        broken_after_1 = self.sandbox.read(rel)
        cur_err_after_1 = _post_write_syntax_check(
            self.sandbox, rel, self.adapter)
        if not cur_err_after_1:
            # somehow already healed (shouldn't happen — attempt 1 would
            # have returned True). Safety: report success.
            return True
        # Attempt 2: escalate to heal_llm if available, else retry same tier
        # with the attempt-1 failure error as additional context.
        next_llm = self.heal_llm or self.llm
        return self._self_heal_one(rel, cur_err_after_1, task, llm=next_llm,
                                   prior_failure=syntax_err,
                                   prior_state=broken_after_1)

    def _self_heal_one(self, rel: str, syntax_err: str, task: Task, *,
                       llm: LLMProvider,
                       prior_failure: Optional[str] = None,
                       prior_state: Optional[str] = None) -> bool:
        """Single self-heal attempt with `llm`. Returns True iff the file
        parses cleanly after the write.
        """
        if self.usage is not None:
            self.usage.increment("self_heal_attempts")
        broken = prior_state if prior_state is not None else self.sandbox.read(rel)
        if not broken:
            return False
        numbered = _with_line_numbers(broken)
        prior_block = ""
        if prior_failure:
            prior_block = (
                f"NOTE: A prior heal attempt also failed with: {prior_failure}\n"
                f"Address that as well — do not reproduce the same broken "
                f"indentation/structure.\n\n"
            )
        user = (
            f"File: {rel}\n"
            f"Syntax error: {syntax_err}\n"
            f"Original task: {task.title}\n\n"
            f"{prior_block}"
            f"Current (broken) contents with line numbers:\n{numbered}\n\n"
            f"Re-emit the entire file with the syntax fixed. Use:\n"
            f"===FILE: {rel}===\n<contents>\n===END===\n"
        )
        try:
            raw = llm.chat(self.SELF_HEAL_SYS, user, temperature=0.1,
                           max_tokens=16000)
        except Exception as e:
            log.warning("self-heal LLM error on %s (using %s): %s",
                        rel, getattr(llm, "name", "?"), e)
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
                log.info("self-heal succeeded for %s (via %s)",
                         rel, getattr(llm, "name", "?"))
                if self.usage is not None:
                    self.usage.increment("self_heal_succeeded")
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


DESTRUCTIVE_RETRY_SYS = (
    "Your previous output would have rewritten an entire file with a much "
    "smaller version — that's destructive and was BLOCKED by the sandbox. "
    "You will be shown the actual current file. Emit ONE ===PATCH=== block "
    "with the SMALLEST possible SEARCH/REPLACE edit that satisfies the "
    "task. SEARCH must match the file byte-for-byte. NEVER emit a full "
    "===FILE=== rewrite for this retry."
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


def _format_current_files(sandbox: "Sandbox", paths: list[str], *,
                          max_chars_per_file: int = 4000) -> str:
    """Snapshot target files for the retry prompt. Caps each at
    max_chars_per_file so a 2k-line template doesn't blow the context.
    Larger files get head + tail rather than head-only — the tail often
    contains the function the LLM just edited."""
    parts: list[str] = []
    for rel in paths:
        try:
            content = sandbox.read(rel)
        except Exception as e:
            parts.append(f"--- {rel} ---\n(read failed: {e})")
            continue
        if not content:
            parts.append(f"--- {rel} ---\n(empty)")
            continue
        if len(content) <= max_chars_per_file:
            parts.append(f"--- {rel} ({len(content)} chars) ---\n{content}")
            continue
        head_n = max_chars_per_file // 2
        tail_n = max_chars_per_file - head_n
        parts.append(
            f"--- {rel} ({len(content)} chars, truncated) ---\n"
            f"{content[:head_n]}\n... [middle omitted] ...\n"
            f"{content[-tail_n:]}"
        )
    return "\n\n".join(parts) if parts else "(no target files)"


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
