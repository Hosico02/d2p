"""Planner agent — given the Analyzer report + current repo state + open
bugs, emits the next concrete file-level Tasks.

The Planner sees a compressed history of prior iterations (stripped of
raw test stdout/stderr by `_compress_history`) so its prompt doesn't
balloon over a long run. It also takes a `feature_cap` so it directly
targets the orchestrator's desired task count instead of producing 5
and getting post-hoc trimmed to 1 under bug debt.
"""
from __future__ import annotations

import json as _json
import logging
import uuid
from typing import Any

from ..fs import Sandbox
from ..providers.base import LLMProvider, chat_structured
from ..models import AnalysisReport, PlanResult, Task
from ..symbols import build_symbol_map


# JSON Schema for the Planner's output. When the provider supports
# structured output (claude API tool-use, OpenAI response_format), the
# model is forced into this shape — no format-thinking overhead, no
# JSON-parse failures. claude-cli and minimax fall back to embedding the
# schema in the prompt.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["rationale", "tasks"],
    "properties": {
        "rationale": {
            "type": "string",
            "description": "2-3 sentences on why these tasks now.",
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "instructions", "target_files",
                             "priority", "category"],
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "target_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "instructions": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 9},
                    "category": {
                        "type": "string",
                        "enum": ["feature", "bugfix", "ux", "docs", "infra"],
                    },
                },
            },
        },
    },
}

log = logging.getLogger("d2p.agents.planner")


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
- {min_tasks} to {max_tasks} tasks.
- Every task instruction must restate which aspect of `essence` it preserves.
- target_files must be paths inside the project. Use new paths for new files.
- Mention specific existing symbols you want to extend (e.g. "extend GameMaster.vote_phase").
- For files > 200 lines, instructions must say "use Mode B SEARCH/REPLACE".
- If there are open bug reports, the highest-priority task MUST be fixing one of them.
- Skip tasks already attempted-and-done in history.
"""


def _compress_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip a raw iteration history down to the fields the Planner actually
    needs to reason about ("don't repeat what just ran, do follow up on what
    failed"). Drops big payloads: full ExecutionResult.summary text, per-test
    qa.test_runs stdout/stderr, retired metadata. Keeps a one-line marker
    per task/fix and short bug titles.
    """
    out: list[dict[str, Any]] = []
    for entry in history:
        compact_results = []
        for r in entry.get("results", []) or []:
            compact_results.append({
                "task_id": r.get("task_id"),
                "status": r.get("status"),
                "files": r.get("files_changed", [])[:4],
            })
        compact_fixes = []
        for r in entry.get("qa_fix_results", []) or []:
            compact_fixes.append({
                "task_id": r.get("task_id"),
                "status": r.get("status"),
            })
        qa = entry.get("qa") or {}
        compact_qa = {
            "new_bug_titles": [b.get("title", "")[:80]
                               for b in (qa.get("new_bugs") or [])],
            "fixed_bug_titles": [b.get("title", "")[:80]
                                 for b in (qa.get("fixed_bugs") or [])],
            "open_bug_titles": [b.get("title", "")[:80]
                                for b in (qa.get("open_bugs") or [])],
        } if qa else {}
        out.append({
            "iteration": entry.get("iteration"),
            "results": compact_results,
            "qa_fixes": compact_fixes,
            "qa": compact_qa,
        })
    return out


class Planner:
    KEY_FILE_CANDIDATES = (
        "README.md", "README", "readme.md",
        "main.py", "app.py", "server.py", "index.js", "index.ts",
        "src/main.ts", "src/index.ts", "src/App.tsx",
        "package.json", "pyproject.toml", "requirements.txt",
        "Cargo.toml",
    )

    def __init__(self, llm: LLMProvider, sandbox: Sandbox, *,
                 max_tasks: int = 5) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.max_tasks = max_tasks

    # Tighter key-files block: 5 files × 3000 chars (was 8 × 5000). The
    # original was ~40 KB of prompt input per Planner call; the model
    # pays cache-creation tokens on it every time the codebase shifts.
    # The 15 KB target keeps the most load-bearing context and trims
    # the long tail that rarely changes the plan.
    KEY_FILES_MAX = 5
    KEY_FILE_CHARS = 3000

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
        for _, p in sizes[:3]:
            if p not in seen:
                seen.append(p)
        return seen[: self.KEY_FILES_MAX]

    def _build_key_files_block(self, key_files: list[str]) -> str:
        chunks = []
        for p in key_files:
            txt = self.sandbox.read(p)
            if not txt:
                continue
            chunks.append(f"=== {p} ===\n{txt[: self.KEY_FILE_CHARS]}")
        return "\n\n".join(chunks) or "(none)"

    def run(self, analysis: AnalysisReport, *, iteration: int, max_iter: int,
            history: list[dict[str, Any]],
            open_bugs: list[dict[str, Any]] | None = None,
            feature_cap: int | None = None) -> PlanResult:
        """Build the next plan. If `feature_cap` is set, the Planner is told
        to emit at most that many tasks — avoids the previous "Planner
        produces 5, orchestrator post-hoc trims to 1 under bug debt"
        pattern that wasted 4× the reasoning budget."""
        cap = feature_cap if feature_cap is not None else self.max_tasks
        listing_raw = self.sandbox.listing(max_entries=200)
        listing_str = "\n".join(listing_raw)
        # strip the trailing "... (truncated)" marker before symbol/file picking
        listing = [p for p in listing_raw if not p.startswith("...")]
        key_files = self._pick_key_files(listing)
        key_files_block = self._build_key_files_block(key_files)
        symbol_map = build_symbol_map(self.sandbox.read, listing)
        # Floor on requested task count: keep at least 3 so the Planner has
        # room to surface multiple angles; cap on the upper end so it
        # doesn't go crazy. The lower bound also stops the prompt template
        # from rendering "3 to 1 tasks" (nonsense) when bug debt forces
        # cap=1.
        plan_lo = min(3, cap)
        # Compress history before serialising it into the prompt. Raw
        # history entries embed full task dicts + qa.test_runs (full
        # subprocess stdout/stderr per test, ~kB each). After a few iters
        # this rapidly inflates the Planner prompt with mostly-irrelevant
        # ballast that the model also pays cache-creation cost on. Keep
        # just the signal: iteration, what tasks ran, their status, what
        # bugs got found/fixed.
        compact_history = _compress_history(history[-3:]) if history else []
        user = PLANNER_USER_TMPL.format(
            analysis=_json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2),
            listing=listing_str,
            key_files=key_files_block,
            symbol_map=_json.dumps(symbol_map, ensure_ascii=False, indent=2),
            open_bugs=_json.dumps(open_bugs or [], ensure_ascii=False, indent=2),
            history=_json.dumps(compact_history, ensure_ascii=False, indent=2) if compact_history else "(none)",
            iteration=iteration,
            max_iter=max_iter,
            min_tasks=plan_lo,
            max_tasks=cap,
        )
        # Use chat_structured when the provider supports it (Anthropic API,
        # OpenAI). Falls back to chat_json with the schema appended to the
        # prompt otherwise (claude-cli, minimax) — same correctness, no
        # speed bonus.
        data = chat_structured(self.llm, PLANNER_SYS, user,
                               schema=PLAN_SCHEMA,
                               temperature=0.3, max_tokens=4000)
        tasks = []
        for t in data.get("tasks", [])[:cap]:
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
