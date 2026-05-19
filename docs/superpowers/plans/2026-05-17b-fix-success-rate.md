# Fix-Success-Rate Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` syntax.

**Goal:** Push d2p's per-iteration fix-success-rate (`fixed/(fixed+open)`) from ~50% toward ~80% by closing three concrete gaps observed in the 3-iter live run.

**Architecture:** Three independent improvements stacked on top of T1-T4:
1. Health probe now includes the QA test corpus itself, catching symbol deletions whose only external references are in `tests/d2p_qa/*`.
2. Fix-task instructions now include the FULL test file contents (currently only failure stderr is shown to the model).
3. Fix Executor re-runs the test immediately after writing and, on failure, does one in-iter retry with the live failure output — no waiting for the next iteration.

**Tech Stack:** unchanged.

---

## File Structure

- **Modify** `d2p/health.py` — `ProjectHealth.default_modules()` includes `tests/d2p_qa/<stem>` form for any test file in the corpus.
- **Modify** `d2p/qa.py` — `_bug_to_task` embeds the test file's contents into `instructions`.
- **Modify** `d2p/agents.py` — `Executor.run()` gains an optional `post_check` callback; for bugfix-category tasks the orchestrator wires it to "run this test path; return (ok, output)". The Executor consults it after success and, on failure, makes one more LLM round.
- **Modify** `d2p/orchestrator.py` — wire the `post_check` for QA-fix batch.
- **Modify** `tests/test_health.py` and `tests/test_units.py` — new tests.

---

## Task 5: Health probe covers QA test corpus

**Files:**
- Modify: `d2p/health.py:48-58` (`default_modules`)
- Test: `tests/test_health.py` (new test method)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_health.py`:

```python
    def test_default_modules_includes_qa_corpus(self) -> None:
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests" / "__init__.py").write_text("")
        (self.root / "tests" / "d2p_qa").mkdir()
        (self.root / "tests" / "d2p_qa" / "__init__.py").write_text("")
        (self.root / "tests" / "d2p_qa" / "test_bug_x.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n"
        )
        h = ProjectHealth(self.sb)
        mods = h.default_modules()
        self.assertIn("tests.d2p_qa.test_bug_x", mods)
        # And it imports cleanly:
        result = h.probe(mods)
        self.assertEqual(result["tests.d2p_qa.test_bug_x"], "ok")
```

- [ ] **Step 2: Run to confirm fail**

`.venv/bin/python -m unittest tests.test_health -v` → FAIL (KeyError or assertion).

- [ ] **Step 3: Implement**

In `d2p/health.py`, replace `default_modules`:

```python
    def default_modules(self) -> list[str]:
        out: list[str] = []
        # top-level *.py
        for p in sorted(self.sandbox.listing(max_entries=400)):
            if not p.endswith(".py") or "/" in p:
                continue
            stem = p[:-3]
            if stem in {"setup", "conftest"}:
                continue
            out.append(stem)
        # QA corpus tests under tests/d2p_qa/
        for p in sorted(self.sandbox.listing(max_entries=400)):
            if not p.startswith("tests/d2p_qa/") or not p.endswith(".py"):
                continue
            if p.endswith("__init__.py") or p.endswith("conftest.py"):
                continue
            out.append(p[:-3].replace("/", "."))
        return out
```

- [ ] **Step 4: Run full suite**

`.venv/bin/python -m unittest discover -s tests -p "test_*.py"` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add d2p/health.py tests/test_health.py
git commit -m "feat(d2p): health probe also imports tests/d2p_qa corpus"
```

---

## Task 6: Embed test source in fix-task instructions

**Files:**
- Modify: `d2p/qa.py:_bug_to_task`
- Test: `tests/test_units.py` (new `TestFixTaskHasTestSource`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_units.py`:

```python
class TestFixTaskHasTestSource(unittest.TestCase):
    def test_instructions_include_test_file_contents(self) -> None:
        import tempfile
        from pathlib import Path
        from d2p.fs import Sandbox
        from d2p.qa import QAAgent, BugReport

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "tests" / "d2p_qa").mkdir()
            test_body = ("import unittest\n"
                         "class TestFoo(unittest.TestCase):\n"
                         "    def test_a(self):\n"
                         "        self.assertEqual(1, 2)  # canary marker\n")
            (root / "tests" / "d2p_qa" / "test_bug.py").write_text(test_body)
            sb = Sandbox(root)
            # construct an agent without invoking LLM
            qa = QAAgent.__new__(QAAgent)
            qa.sandbox = sb
            bug = BugReport(id="abc", title="t", test_path="tests/d2p_qa/test_bug.py",
                            category="custom", summary="t", suspected_files=["app.py"],
                            last_failure="AssertionError: 1 != 2")
            task = qa._bug_to_task(bug)
            self.assertIn("canary marker", task.instructions)
            self.assertIn("tests/d2p_qa/test_bug.py", task.instructions)
```

- [ ] **Step 2: Run to confirm fail**

`.venv/bin/python -m unittest tests.test_units.TestFixTaskHasTestSource -v` → FAIL (canary not in instructions).

- [ ] **Step 3: Implement**

In `d2p/qa.py`, replace `_bug_to_task`:

```python
    def _bug_to_task(self, bug: BugReport) -> Task:
        test_src = self.sandbox.read(bug.test_path) or "<test file not found>"
        instructions = (
            f"A QA test is failing. Make the test PASS by fixing the underlying bug.\n"
            f"Test path (READ-ONLY — do not modify): {bug.test_path}\n"
            f"Run it locally with: python -m unittest "
            f"{bug.test_path.replace('/', '.').rsplit('.py', 1)[0]}\n\n"
            f"=== Test source (this is what your fix must satisfy) ===\n{test_src}\n"
            f"=== Failure output ===\n{bug.last_failure[:1500]}\n\n"
            f"You may need to modify any of: {bug.suspected_files or 'use your judgement'}\n"
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
```

- [ ] **Step 4: Run full suite**

`.venv/bin/python -m unittest discover -s tests -p "test_*.py"` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add d2p/qa.py tests/test_units.py
git commit -m "feat(d2p): fix tasks include full test source"
```

---

## Task 7: In-iter fix retry

**Files:**
- Modify: `d2p/agents.py` (`Executor.run` accepts `post_check`)
- Modify: `d2p/orchestrator.py` (wire post_check for fix tasks)
- Test: `tests/test_units.py` (new `TestExecutorPostCheck`)

- [ ] **Step 1: Write the failing test (pure unit, no LLM)**

Append to `tests/test_units.py`:

```python
class TestExecutorPostCheck(unittest.TestCase):
    """The Executor must accept a post_check callable, invoke it on success,
    and surface its verdict in ExecutionResult.error if it returns (False, msg)."""

    def test_post_check_failure_marks_partial(self) -> None:
        from d2p.agents import Executor
        from d2p.models import Task, ExecutionResult

        # construct without LLM
        ex = Executor.__new__(Executor)
        ex.llm = None  # type: ignore
        # patch _retry_fix to be inert; we only test the bookkeeping
        ex.sandbox = None  # type: ignore
        # the smoke-style verification: post_check_result_status returns a string
        from d2p.agents import _apply_post_check_to_result
        res = ExecutionResult(task_id="t1", status="done",
                              summary="did it", files_changed=["app.py"])
        out = _apply_post_check_to_result(res, post_check_ok=False,
                                          post_check_output="AssertionError: nope")
        self.assertEqual(out.status, "failed")
        self.assertIn("post-check failed", out.error)
        self.assertIn("AssertionError", out.error)

    def test_post_check_success_keeps_done(self) -> None:
        from d2p.agents import _apply_post_check_to_result
        from d2p.models import ExecutionResult
        res = ExecutionResult(task_id="t2", status="done",
                              summary="did it", files_changed=["app.py"])
        out = _apply_post_check_to_result(res, post_check_ok=True,
                                          post_check_output="")
        self.assertEqual(out.status, "done")
        self.assertEqual(out.error, "")
```

- [ ] **Step 2: Run to confirm fail**

`.venv/bin/python -m unittest tests.test_units.TestExecutorPostCheck -v` → FAIL with ImportError.

- [ ] **Step 3: Implement helper + plumbing**

In `d2p/agents.py`, add near `_post_write_syntax_check`:

```python
def _apply_post_check_to_result(res: "ExecutionResult", *,
                                post_check_ok: bool,
                                post_check_output: str) -> "ExecutionResult":
    """Mutate the ExecutionResult based on a post-check verdict.

    If the post-check fails, demote `done` -> `failed` and tag the error.
    Caller is responsible for the actual restore of files (typically via
    the orchestrator's snapshot/rollback path).
    """
    if post_check_ok:
        return res
    tail = (post_check_output or "")[-1200:]
    addendum = f"post-check failed: {tail}"
    res.error = (res.error + " | " + addendum) if res.error else addendum
    if res.status == "done":
        res.status = "failed"
    return res
```

In `Executor.run`, add an optional `post_check` parameter. Signature change:

```python
    def run(self, task: Task, *,
            post_check: "Callable[[], tuple[bool, str]] | None" = None,
            ) -> ExecutionResult:
```

After all writes succeed (`if rejected and not changed: ...` block already exists; right before the final return), insert:

```python
        if post_check is not None and result.status == "done":
            ok, output = post_check()
            if not ok:
                # one in-iter retry: tell the model the test still fails
                retry_task = Task(
                    id=task.id + "-retry",
                    title=task.title, rationale=task.rationale,
                    target_files=task.target_files,
                    instructions=task.instructions + (
                        "\n\n=== Previous attempt left the test still failing. "
                        "Output: ===\n" + output[-1500:] +
                        "\n\nProduce a SMALLER, more targeted change."
                    ),
                    priority=task.priority, category=task.category,
                    forbidden_files=task.forbidden_files,
                )
                retry_result = self.run(retry_task)  # no nested post_check
                if retry_result.status == "done":
                    ok2, out2 = post_check()
                    if ok2:
                        # merge file changes; mark done
                        merged = list(dict.fromkeys(result.files_changed + retry_result.files_changed))
                        result.files_changed = merged
                        result.summary = (result.summary + " | retry: " + retry_result.summary)[:200]
                        return result
                    output = out2
                _apply_post_check_to_result(result,
                                            post_check_ok=False,
                                            post_check_output=output)
            else:
                pass
        return result
```

(Be mindful: the existing function returns `result` in two places — early `failed`/`skipped` paths and the final success path. Only the final-success-path `return result` should be wrapped by post_check.)

In `d2p/orchestrator.py`, when running QA-fix tasks, wire a `post_check` closure:

```python
    def _qa_fix_post_check(self, bug_test_path: str):
        def _check():
            r = self.qa._run_test_file(bug_test_path)
            return (r["status"] == "passed", r.get("output", ""))
        return _check
```

And in `_run_tasks_parallel`, allow optional `post_check_for_task: Callable[[Task], Callable] | None`. Pass it to Executor.run. For feature tasks: None. For fix tasks: `lambda t: self._qa_fix_post_check(t.metadata_test_path)` — but Task has no such field; simplest:

Match task id prefix `qa-` → look up `bug.test_path` in a side dict the orchestrator builds when emitting fix tasks. Add this dict at construction time in the QA-fix block.

In orchestrator.run(), where we call `self._run_tasks_parallel(fix_tasks)`, change to:

```python
                    fix_test_paths = {t.id: b.test_path
                                      for t, b in zip(fix_tasks,
                                                      (qa_report.new_bugs +
                                                       qa_report.open_bugs))}
                    qa_fix_results = self._run_tasks_parallel(
                        fix_tasks,
                        post_check_for=lambda t: self._qa_fix_post_check(
                            fix_test_paths[t.id]),
                    )
```

(Update `_run_tasks_parallel` signature to accept a kwarg `post_check_for: Callable[[Task], Callable] | None = None`, defaulting to None, and pass `post_check=post_check_for(task) if post_check_for else None` to `Executor.run`.)

- [ ] **Step 4: Run full suite**

`.venv/bin/python -m unittest discover -s tests -p "test_*.py"` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add d2p/agents.py d2p/orchestrator.py tests/test_units.py
git commit -m "feat(d2p): fix Executor re-runs the test and retries once on failure"
```

---

## Self-Review

1. **Spec coverage:** Three improvements all mapped to tasks. T5 closes `build_system_prompt`-style symbol deletion. T6 gives the model what it currently lacks (the test source). T7 closes the iter-boundary waste loop.
2. **No placeholders:** every step has complete code.
3. **Types:** `post_check: Callable[[], tuple[bool, str]]` consistent across `Executor.run`, `_run_tasks_parallel`, and `_qa_fix_post_check`. `default_modules() -> list[str]` unchanged shape.
