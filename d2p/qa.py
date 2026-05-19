"""QA agent — generates failing tests that capture bugs, runs them, persists
the test corpus under <target>/tests/d2p_qa/.

A failing test is the bug report. Tests must be:
- inherits from unittest.TestCase
- contains at least one def test_…
- > 300 chars total (catches truncated LLM output)
- stdlib-only OR uses libs the project already imports

The fix Task is dispatched with `forbidden_files=[test_path]` so the Executor
cannot weaken/delete the test that documents the bug.

Philosophy
----------
Test failure IS the bug report. A test that fails is a bug; a test that passes
is a regression check the project will never lose. Across iterations, the
corpus grows: each newly discovered bug becomes a permanent guardrail. Across
projects, the *checklist* of bug categories transfers (see qa_checklist.json).

The QA agent has three responsibilities:

1. RUN existing accumulated tests (regression sweep).
2. PROBE the current project for new bugs by simulating user-style scenarios
   (informed by the checklist + the analyzer's view of the domain).
3. EMIT failing test files into tests/d2p_qa/ and bug-fix Task objects that
   the orchestrator hands to executors.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fs import Sandbox
from .lang import LanguageAdapter, adapter_for, detect_primary_language
from .llm import MiniMaxClient
from .models import Task

log = logging.getLogger("d2p.qa")

CORPUS_DIR = "tests/d2p_qa"          # lives inside the target project
INIT_FILE = "tests/d2p_qa/__init__.py"


@dataclass
class BugReport:
    id: str
    title: str
    test_path: str                   # relative to target root
    category: str                    # one of qa_checklist ids, or 'custom'
    summary: str
    suspected_files: list[str] = field(default_factory=list)
    last_failure: str = ""           # truncated stderr/stdout from the failing run
    status: str = "open"             # open | fixed | flaky

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "test_path": self.test_path,
            "category": self.category, "summary": self.summary,
            "suspected_files": self.suspected_files,
            "last_failure": self.last_failure, "status": self.status,
        }


@dataclass
class QAReport:
    new_bugs: list[BugReport]
    open_bugs: list[BugReport]       # bugs known from prior runs that still fail
    fixed_bugs: list[BugReport]      # previously failing tests that now pass
    test_runs: dict[str, Any]        # raw test runner output by path

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_bugs": [b.to_dict() for b in self.new_bugs],
            "open_bugs": [b.to_dict() for b in self.open_bugs],
            "fixed_bugs": [b.to_dict() for b in self.fixed_bugs],
            "test_runs": self.test_runs,
        }


# ============ QA agent ========================================================

QA_PROBE_SYS = """You are the QA agent. You simulate a real user interacting
with the project and look for bugs. You do NOT write fluffy "test that the
thing exists" tests — every test you write should be a HYPOTHESIS about a
likely bug, expressed as a runnable Python test that FAILS if the bug exists.

You will receive: the project layout, key source files, the analyzer's domain
report, and a bug-class checklist (categories like input_validation,
concurrency, …). Use the checklist to ensure you cover several categories.

OUTPUT FORMAT — delimited plain text, NEVER JSON:

For each test you want to add, emit one block:

===TEST: tests/d2p_qa/test_<slug>.py===
META: {"title": "...", "category": "<checklist id>", "suspected_files": ["..."]}
<entire test file contents — plain Python, stdlib `unittest`>
===END===

Hard rules for the tests themselves:
- Use only the stdlib. NO pytest, NO third-party fixtures, NO `@pytest.fixture`,
  NO conftest.py, NO bare `def test_*` functions at module level.
- EVERY test class MUST inherit from `unittest.TestCase`. EVERY test method
  MUST live inside such a class. This is non-negotiable.
- Each test file must be independently runnable: `python -m unittest tests.d2p_qa.test_X`.
- Import the project's modules by relative `sys.path.insert` to the project
  root, so the test works regardless of how it's invoked.
- For Flask/FastAPI apps, prefer the framework's test_client (no network).
- For CLI tools, use `subprocess.run` with a short timeout.
- Tests should be FAST: target < 5 seconds total per file.
- A test must FAIL today (so it captures a real bug). If you cannot construct
  a failing scenario, do NOT emit the test — skip it.

REQUIRED skeleton (copy and adapt — do NOT deviate):

```
\"\"\"<one-line bug hypothesis>\"\"\"
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# project imports
from app import app  # adapt to project


class TestSomething(unittest.TestCase):
    def setUp(self):
        # construct fixtures here
        pass

    def test_descriptive_name(self):
        # exercise the suspected-buggy path
        # assert what *should* be true; if the bug exists the assertion fails
        self.assertEqual(...)


if __name__ == \"__main__\":
    unittest.main()
```
"""

QA_PROBE_USER_TMPL = """Project root: {root}
Domain (from Analyzer): {domain}
Essence (must be respected — do NOT write tests that assume the demo is for a
different audience than this): {essence}
Audience: {audience}

Primary language: {language}
Use this language's idiomatic test framework. The required skeleton for THIS
project (copy and adapt, do not deviate from the conventions shown):

{test_template}

Place new tests under: {corpus_dir}/

Bug-class checklist (cover at least 3 different categories):
{checklist}

Project file listing:
{listing}

Key source files:
{key_files}

Symbol map:
{symbol_map}

Already-existing QA tests (do NOT duplicate these):
{existing_tests}

Emit 2-4 new test blocks. Each must be likely to FAIL on the current code.
"""


class QAAgent:
    def __init__(self, llm: MiniMaxClient, sandbox: Sandbox,
                 checklist_path: Path | None = None,
                 adapter: LanguageAdapter | None = None) -> None:
        self.llm = llm
        self.sandbox = sandbox
        if checklist_path is None:
            checklist_path = Path(__file__).resolve().parent / "qa_checklist.json"
        self.checklist = json.loads(checklist_path.read_text())
        self.adapter = adapter or adapter_for(detect_primary_language(sandbox))
        # Keep CORPUS_DIR working for legacy hardcoded references while letting
        # adapters override it per-language (e.g. JS uses the same dir).
        self.corpus_dir = self.adapter.test_corpus_dir
        self._test_python = self._pick_test_python()
        log.info("QA language=%s corpus=%s runner-python=%s",
                 self.adapter.name, self.corpus_dir, self._test_python)

    def _pick_test_python(self) -> str:
        """Pick a python interpreter that can import the project's dependencies.

        Strategy: collect top-level non-stdlib imports from the project's entry
        files; try each candidate interpreter (system python3 first, then
        d2p's sys.executable); use the first one that imports them all.
        """
        candidates = []
        sys_python = shutil.which("python3") or shutil.which("python")
        if sys_python and sys_python != sys.executable:
            candidates.append(sys_python)
        candidates.append(sys.executable)

        # cheap import probe: take top-of-file imports from entry candidates
        probe_mods = self._probe_modules()
        if not probe_mods:
            return candidates[0]

        for py in candidates:
            try:
                cmd = [py, "-c", "; ".join(f"import {m}" for m in probe_mods)]
                r = subprocess.run(cmd, capture_output=True, timeout=10)
                if r.returncode == 0:
                    return py
            except Exception:
                continue
        return candidates[0]

    def _probe_modules(self) -> list[str]:
        entries = ["app.py", "main.py", "server.py", "index.py", "src/main.py"]
        third_party: list[str] = []
        stdlib_hints = {
            "os", "sys", "json", "re", "time", "threading", "queue", "uuid",
            "subprocess", "pathlib", "typing", "dataclasses", "collections",
            "logging", "functools", "itertools", "math", "random", "tempfile",
            "argparse", "ast", "enum", "io",
        }
        seen = set()
        rx = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
                        re.MULTILINE)
        for entry in entries:
            txt = self.sandbox.read(entry)
            if not txt:
                continue
            for m in rx.finditer(txt):
                mod = (m.group(1) or m.group(2)).split(".")[0]
                if mod in stdlib_hints or mod in seen:
                    continue
                seen.add(mod)
                third_party.append(mod)
                if len(third_party) >= 5:
                    return third_party
        return third_party

    # ---- public --------------------------------------------------------------

    def run(self, *, analysis_summary: str, essence: str, audience: str,
            key_files_block: str,
            symbol_map: dict[str, list[str]]) -> tuple[QAReport, list[Task]]:
        self._ensure_corpus_dir()
        existing = self._list_existing_tests()

        # 1. regression sweep on existing tests
        prior_runs: dict[str, Any] = {}
        prior_failed: list[BugReport] = []
        prior_fixed: list[BugReport] = []
        prior_meta = self._load_meta()
        for rel in existing:
            outcome = self._run_test_file(rel)
            prior_runs[rel] = outcome
            meta = prior_meta.get(rel, {})
            if outcome["status"] != "passed":
                br = BugReport(
                    id=meta.get("id", uuid.uuid4().hex[:8]),
                    title=meta.get("title", rel),
                    test_path=rel,
                    category=meta.get("category", "custom"),
                    summary=meta.get("title", ""),
                    suspected_files=meta.get("suspected_files", []),
                    last_failure=outcome["output"][-1500:],
                    status="open",
                )
                prior_failed.append(br)
            else:
                if meta.get("status") == "open":
                    prior_fixed.append(BugReport(
                        id=meta.get("id", uuid.uuid4().hex[:8]),
                        title=meta.get("title", rel),
                        test_path=rel,
                        category=meta.get("category", "custom"),
                        summary=meta.get("title", ""),
                        suspected_files=meta.get("suspected_files", []),
                        status="fixed",
                    ))

        # 2. probe for new bugs (LLM-driven) — no throttle. The real fix to
        # bug-debt is improving fix success rate, not hiding bugs.
        new_bugs = self._probe_for_new_bugs(
            analysis_summary=analysis_summary,
            essence=essence,
            audience=audience,
            key_files_block=key_files_block,
            symbol_map=symbol_map,
            existing_tests=existing,
        )

        # 3. each new bug runs its test now to confirm it fails (otherwise drop it)
        confirmed_new: list[BugReport] = []
        for br in new_bugs:
            outcome = self._run_test_file(br.test_path)
            prior_runs[br.test_path] = outcome
            if outcome["status"] != "passed":
                br.last_failure = outcome["output"][-1500:]
                confirmed_new.append(br)
            else:
                # the model misjudged — the "bug" isn't real. Delete the test to
                # avoid polluting the corpus with green-on-arrival noise.
                self.sandbox.delete(br.test_path)

        # 4. persist meta
        all_failing = prior_failed + confirmed_new
        self._save_meta(all_failing, prior_fixed)

        report = QAReport(
            new_bugs=confirmed_new,
            open_bugs=prior_failed,
            fixed_bugs=prior_fixed,
            test_runs=prior_runs,
        )
        fix_tasks = [self._bug_to_task(b) for b in all_failing]

        # Auto-restore tasks for any "cannot import name X from Y" we saw —
        # these are the most common cause of fix failure in practice and the
        # fix is mechanical (re-add the symbol). Run them at priority 0 so
        # they execute before regular bug-fix tasks.
        seen_pairs: set[tuple[str, str]] = set()
        restore_tasks: list[Task] = []
        for bug in all_failing:
            for sym, mod in detect_missing_symbol_failures(bug.last_failure):
                if (sym, mod) in seen_pairs:
                    continue
                seen_pairs.add((sym, mod))
                module_path = f"{mod.replace('.', '/')}.py"
                if not self.sandbox.read(module_path):
                    continue  # module file doesn't even exist — skip
                restore_tasks.append(Task(
                    id=f"restore-{uuid.uuid4().hex[:6]}",
                    title=f"[RESTORE] Add {sym} back to {module_path}",
                    rationale=f"Missing symbol detected in test failures: "
                              f"cannot import name '{sym}' from '{mod}'",
                    target_files=[module_path],
                    instructions=(
                        f"The symbol `{sym}` is missing from {module_path} "
                        f"but is referenced elsewhere in the project "
                        f"(at minimum, the QA tests need it). Add `{sym}` back "
                        f"to {module_path} with a reasonable, self-consistent "
                        f"implementation that matches how it is used.\n"
                        f"Use Mode B (SEARCH/REPLACE) if {module_path} is "
                        f"non-trivial — do NOT rewrite the whole file."
                    ),
                    priority=0,
                    category="bugfix",
                ))
        return report, restore_tasks + fix_tasks

    # ---- internals -----------------------------------------------------------

    def _ensure_corpus_dir(self) -> None:
        # __init__.py only matters for Python's unittest discovery
        if self.adapter.name == "python":
            init = f"{self.corpus_dir}/__init__.py"
            if not self.sandbox.read(init):
                self.sandbox.write(init, "")
            if not self.sandbox.read("tests/__init__.py"):
                self.sandbox.write("tests/__init__.py", "")

    def _list_existing_tests(self) -> list[str]:
        return [p for p in self.sandbox.listing(max_entries=400)
                if p.startswith(self.corpus_dir + "/")
                and p.endswith(".py")
                and not p.endswith("__init__.py")]

    def _load_meta(self) -> dict[str, dict[str, Any]]:
        raw = self.sandbox.read(f"{self.corpus_dir}/_meta.json")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _save_meta(self, failing: list[BugReport],
                   fixed: list[BugReport]) -> None:
        prior = self._load_meta()
        for b in failing:
            prior[b.test_path] = {**b.to_dict(), "status": "open"}
        for b in fixed:
            prior[b.test_path] = {**b.to_dict(), "status": "fixed"}
        self.sandbox.write(f"{self.corpus_dir}/_meta.json",
                           json.dumps(prior, ensure_ascii=False, indent=2))

    def flip_meta_status(self, test_path: str, status: str) -> None:
        """Atomically flip a single bug's status in _meta.json. Called by the
        orchestrator the instant a fix's post_check turns green, so 'fixed'
        statistics reflect reality this iteration — not the next."""
        prior = self._load_meta()
        if test_path not in prior:
            return
        prior[test_path]["status"] = status
        self.sandbox.write(f"{self.corpus_dir}/_meta.json",
                           json.dumps(prior, ensure_ascii=False, indent=2))

    def _run_test_file(self, rel_path: str) -> dict[str, Any]:
        """Run one test file via the adapter's runner. Python falls back to
        unittest with the picked interpreter; JS uses `node --test`; unknown
        languages return a no-op 'unsupported' verdict."""
        cmd = self.adapter.test_runner_cmd(rel_path, sandbox=self.sandbox)
        if not cmd:
            # Python adapter overrides may be inactive (e.g. NullAdapter);
            # legacy fall-through using unittest + picked python.
            if rel_path.endswith(".py"):
                module = rel_path.replace("/", ".").rsplit(".py", 1)[0]
                cmd = [self._test_python, "-m", "unittest", module, "-v"]
            else:
                return {"status": "unsupported", "returncode": 0,
                        "output": f"no runner for {rel_path}"}
        elif rel_path.endswith(".py"):
            # Adapter returned the python module command — substitute the
            # picked interpreter (which has the project's deps installed).
            cmd = [self._test_python] + cmd[1:]
        env = {**os.environ}
        try:
            r = subprocess.run(
                cmd, cwd=str(self.sandbox.root),
                capture_output=True, text=True, timeout=30, env=env,
            )
            output = (r.stdout + "\n" + r.stderr).strip()
            status = "passed" if r.returncode == 0 else "failed"
            return {"status": status, "returncode": r.returncode, "output": output}
        except subprocess.TimeoutExpired as e:
            return {"status": "timeout", "returncode": -1,
                    "output": f"timeout after 30s\n{e.stdout or ''}\n{e.stderr or ''}"}
        except Exception as e:
            return {"status": "error", "returncode": -1, "output": str(e)}

    def _probe_for_new_bugs(self, *, analysis_summary: str,
                            essence: str, audience: str,
                            key_files_block: str,
                            symbol_map: dict[str, list[str]],
                            existing_tests: list[str]) -> list[BugReport]:
        listing = "\n".join(self.sandbox.listing(max_entries=160))
        checklist = json.dumps(self.checklist, ensure_ascii=False, indent=2)
        existing_block = "\n".join(existing_tests) or "(none)"
        user = QA_PROBE_USER_TMPL.format(
            root=str(self.sandbox.root),
            domain=analysis_summary,
            essence=essence or "(unspecified)",
            audience=audience or "(unspecified)",
            language=self.adapter.name,
            test_template=self.adapter.test_template() or "(no template — language not fully supported)",
            corpus_dir=self.corpus_dir,
            checklist=checklist,
            listing=listing,
            key_files=key_files_block,
            symbol_map=json.dumps(symbol_map, ensure_ascii=False, indent=2),
            existing_tests=existing_block,
        )
        try:
            raw = self.llm.chat(QA_PROBE_SYS, user, temperature=0.4,
                                max_tokens=12000)
        except Exception as e:
            log.warning("QA probe LLM error: %s", e)
            return []

        parsed = parse_qa_output(raw)
        bugs: list[BugReport] = []
        for path, meta, content in parsed:
            if not path.startswith(self.corpus_dir + "/"):
                path = f"{self.corpus_dir}/{Path(path).name}"
            quality_reason = _validate_test_quality(content, self.adapter.name)
            if quality_reason:
                log.warning("QA test %s rejected: %s", path, quality_reason)
                continue
            try:
                self.sandbox.write(path, content)
            except Exception as e:
                log.warning("QA write %s failed: %s", path, e)
                continue
            bugs.append(BugReport(
                id=uuid.uuid4().hex[:8],
                title=str(meta.get("title", path)),
                test_path=path,
                category=str(meta.get("category", "custom")),
                summary=str(meta.get("title", "")),
                suspected_files=list(meta.get("suspected_files", []) or []),
            ))
        return bugs

    def _bug_to_task(self, bug: BugReport) -> Task:
        test_src = self.sandbox.read(bug.test_path) or "<test file not found>"
        instructions = (
            f"A QA test is failing. Make it PASS by fixing the underlying bug.\n"
            f"Test path (READ-ONLY — do not modify): {bug.test_path}\n"
            f"Run with: python -m unittest "
            f"{bug.test_path.replace('/', '.').rsplit('.py', 1)[0]}\n\n"
            f"=== Test source (this is exactly what your fix must satisfy) ===\n"
            f"{test_src}\n"
            f"=== Failure output ===\n{bug.last_failure[:1500]}\n\n"
            f"You may need to modify: {bug.suspected_files or 'use your judgement'}\n"
            f"Aim for the SMALLEST possible diff. Do not refactor unrelated code.\n"
            f"Hard rule: do NOT weaken or delete the test. Fix the code under test."
        )
        return Task(
            id=f"qa-{bug.id}",
            title=f"[QA] Fix: {bug.title}"[:120],
            rationale=f"QA test {bug.test_path} fails (category={bug.category})",
            target_files=bug.suspected_files or [],
            instructions=instructions,
            priority=1,
            category="bugfix",
            forbidden_files=[bug.test_path],
        )


_MISSING_SYM_RE = re.compile(
    r"ImportError:\s*cannot import name ['\"]([A-Za-z_][\w]*)['\"]\s+from\s+['\"]([\w.]+)['\"]"
)


def detect_missing_symbol_failures(text: str) -> list[tuple[str, str]]:
    """Return list of (symbol, module) pairs found in a test failure output."""
    if not text:
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _MISSING_SYM_RE.finditer(text):
        pair = (m.group(1), m.group(2))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def _validate_test_quality(content: str, language: str = "python") -> str:
    """Return non-empty reason if the test file is unfit to enter the corpus.

    Per-language checks: Python wants `unittest.TestCase` + `def test_*`;
    JS/TS want `node:test` import (or `test(`) and at least one assertion call.
    """
    if len(content) < 300:
        return f"too short ({len(content)} bytes — likely truncated)"
    if language == "python":
        if "unittest.TestCase" not in content and "TestCase" not in content:
            return "no TestCase subclass — unittest can't discover tests"
        if not re.search(r"def\s+test_\w+\s*\(", content):
            return "no `def test_*` method"
    elif language in ("javascript", "typescript"):
        if "node:test" not in content and "test(" not in content:
            return "no node:test usage"
        if "assert" not in content:
            return "no assertion calls"
    # very crude truncation check — last non-empty line should look complete
    last = content.rstrip().splitlines()[-1] if content.strip() else ""
    if last.endswith((",", "(", "[", "{", "+", "-", "=")):
        return f"trailing-syntax suggests truncation: {last[-40:]!r}"
    return ""


# ============ Parser ==========================================================

_TEST_BLOCK_RE = re.compile(
    r"===TEST:\s*(?P<path>[^=\n]+?)\s*===\s*\n"
    r"(?:META:\s*(?P<meta>\{.*?\})\s*\n)?"
    r"(?P<body>.*?)(?:\n===END===|\Z)",
    re.DOTALL,
)


def parse_qa_output(text: str) -> list[tuple[str, dict[str, Any], str]]:
    out = []
    for m in _TEST_BLOCK_RE.finditer(text or ""):
        path = m.group("path").strip().strip("`'\"")
        meta_raw = m.group("meta") or "{}"
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = {}
        body = m.group("body")
        if body.startswith("\n"):
            body = body[1:]
        body = body.rstrip() + "\n"
        if path and body.strip():
            out.append((path, meta, body))
    return out
