"""Closed-loop driver: Analyzer -> Planner -> Executors -> QA -> Fix-Executors -> repeat."""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .agents import Analyzer, Executor, Planner
from .config import Config
from .fs import Sandbox
from .health import ProjectHealth
from .models import ExecutionResult, PlanResult, Task
from .providers import RoleRouter, build_router
from .qa import QAAgent
from .symbols import build_symbol_map

log = logging.getLogger("d2p.orchestrator")


def _rollback_if_health_regressed(sandbox: Sandbox, probe: ProjectHealth,
                                  *, baseline: dict[str, str],
                                  snapshot: dict[str, Any]) -> bool:
    """If any module that WAS healthy in `baseline` is now broken, restore the
    snapshot and return True. Otherwise return False (writes are kept).

    This catches the case where ast.parse passes but a symbol got deleted,
    breaking other modules that import it.
    """
    if not baseline:
        return False
    current = probe.probe(list(baseline.keys()))
    regressed = [m for m, status in current.items()
                 if baseline.get(m) == "ok" and status != "ok"]
    if not regressed:
        return False
    log.warning("HEALTH REGRESSION in %s — rolling back. errors: %s",
                regressed, {m: current[m][:80] for m in regressed})
    sandbox.restore(snapshot)
    return True


def _discover_pre_existing_tests(sandbox: Sandbox) -> list[str]:
    """Find the demo's own test files (excluding d2p_qa corpus)."""
    out = []
    for p in sandbox.listing(max_entries=400):
        if not p.endswith(".py") or not p.startswith("tests/"):
            continue
        if p.startswith("tests/d2p_qa/") or p.endswith("__init__.py"):
            continue
        out.append(p)
    return out


def _rollback_if_baseline_test_regressed(sandbox: Sandbox, qa,
                                         *, baseline: dict[str, bool],
                                         snapshot: dict[str, Any]) -> bool:
    """For each demo-author test that PASSED at baseline, verify it still
    passes. If any flipped to fail/error, rollback. This catches RUNTIME
    regressions (e.g. dataclass `field()` misuse) that import-probe misses.
    """
    if not baseline:
        return False
    for path, was_passing in baseline.items():
        if not was_passing:
            continue
        r = qa._run_test_file(path)
        if r["status"] != "passed":
            log.warning("BASELINE TEST REGRESSION: %s — rolling back", path)
            sandbox.restore(snapshot)
            return True
    return False


class Orchestrator:
    def __init__(self, target_dir: str, *, cfg: Config | None = None,
                 parallel: int | None = None,
                 max_iterations: int | None = None,
                 enable_qa: bool = True,
                 router: RoleRouter | None = None) -> None:
        self.cfg = cfg or Config()
        if parallel is not None:
            self.cfg.parallel_executors = parallel
        if max_iterations is not None:
            self.cfg.max_iterations = max_iterations
        self.enable_qa = enable_qa
        self.sandbox = Sandbox(target_dir)
        # Per-role LLM router — Haiku/mini for hot path, Opus/4o for reasoning.
        # Defaults pulled from D2P_PROVIDER / role env overrides.
        # working_dir is needed for claude-cli (subprocess cwd).
        self.router = router or build_router(working_dir=str(self.sandbox.root))
        log.info("LLM routing: %s", self.router.describe())
        # Each agent gets the provider tuned for its role.
        self.analyzer = Analyzer(self.router.for_role("analyzer"), self.sandbox)
        self.planner = Planner(self.router.for_role("planner"), self.sandbox)
        # Executor instances are created per-batch inside _run_tasks_parallel
        # so they pick up the latest "executor" provider binding.
        self.qa = (QAAgent(self.router.for_role("qa"), self.sandbox)
                   if enable_qa else None)
        self.health = ProjectHealth(self.sandbox)
        self.run_dir = self.sandbox.root / ".d2p" / time.strftime("run-%Y%m%d-%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public

    def run(self) -> dict[str, Any]:
        log.info("d2p starting on %s", self.sandbox.root)
        analysis = self.analyzer.run()
        self._dump("analysis.json", analysis.to_dict())
        log.info("Analyzer: essence=%r audience=%r features=%d",
                 analysis.essence[:80], analysis.audience, len(analysis.features))

        history: list[dict[str, Any]] = []
        open_bugs: list[dict[str, Any]] = []

        bug_debt_threshold = 12        # P2: raised from 6 → 12
        always_min_features = 1        # P2: even under debt, always allow ≥1
        for it in range(1, self.cfg.max_iterations + 1):
            log.info("=== iteration %d / %d ===", it, self.cfg.max_iterations)

            # P2 SOFT THROTTLE: when backlog is high, RUN FEWER feature tasks
            # rather than dropping them entirely. Always allow at least
            # `always_min_features` so the product keeps moving — fully
            # skipping features for many iters created the "10-iter run
            # produces less than 3-iter run" pathology.
            debt = len(open_bugs) if open_bugs else 0
            if debt >= bug_debt_threshold:
                feature_cap = always_min_features
                log.info("Bug backlog %d >= %d — throttling features to %d this iter",
                         debt, bug_debt_threshold, feature_cap)
            else:
                feature_cap = self.planner.max_tasks   # default 5

            plan = self.planner.run(
                analysis, iteration=it,
                max_iter=self.cfg.max_iterations,
                history=history, open_bugs=open_bugs,
            )
            self._dump(f"plan_iter{it}.json", plan.to_dict())
            log.info("Planner produced %d tasks (cap=%d)", len(plan.tasks), feature_cap)
            # P2: trim to cap, prioritising small/low-risk tasks first
            plan.tasks = sorted(plan.tasks,
                                key=lambda t: (t.priority, len(t.target_files)))[:feature_cap]
            if not plan.tasks:
                log.info("Planner emitted no tasks — converged.")
                break
            results = self._run_tasks_parallel(plan.tasks)

            self._dump(f"exec_iter{it}.json", [r.to_dict() for r in results])
            done = sum(1 for r in results if r.status == "done")
            log.info("Iteration %d: %d/%d feature tasks done", it, done, len(results))

            qa_report = None
            qa_fix_results: list[ExecutionResult] = []
            regressions: list[dict[str, Any]] = []
            if self.qa is not None:
                qa_report, fix_tasks = self._run_qa(analysis)
                self._dump(f"qa_iter{it}.json", qa_report.to_dict())
                log.info("QA: new_bugs=%d open_bugs=%d fixed=%d",
                         len(qa_report.new_bugs), len(qa_report.open_bugs),
                         len(qa_report.fixed_bugs))
                if fix_tasks:
                    # 1) snapshot the entire test-result baseline + all files
                    # the fix tasks might touch, BEFORE running fixes.
                    fix_target_files = sorted({
                        f for t in fix_tasks for f in t.target_files
                    })
                    pre_snapshot = self.sandbox.snapshot(fix_target_files)
                    pre_pass_tests = self._test_baseline()
                    log.info("Regression baseline: %d tests currently passing",
                             sum(1 for v in pre_pass_tests.values() if v))

                    # 2) per-task snapshot so we can selectively rollback
                    task_snapshots: dict[str, dict[str, Any]] = {}
                    for t in fix_tasks:
                        task_snapshots[t.id] = self.sandbox.snapshot(t.target_files)

                    # 3) map task -> bug.test_path so post_check knows what to run
                    bug_test_paths: dict[str, str] = {}
                    for t, b in zip(fix_tasks,
                                    qa_report.new_bugs + qa_report.open_bugs):
                        bug_test_paths[t.id] = b.test_path

                    def _pc_for(t):
                        path = bug_test_paths.get(t.id)
                        return self._qa_fix_post_check(path) if path else None

                    qa_fix_results = self._run_tasks_parallel(
                        fix_tasks, post_check_for=_pc_for)
                    self._dump(f"qa_fix_iter{it}.json",
                               [r.to_dict() for r in qa_fix_results])

                    # 3) re-run the WHOLE corpus to detect regressions
                    all_tests = self._all_corpus_tests()
                    post_runs = {p: self.qa._run_test_file(p) for p in all_tests}
                    regressions = [
                        {"test": p,
                         "output": post_runs[p]["output"][-800:]}
                        for p in all_tests
                        if pre_pass_tests.get(p) is True
                        and post_runs[p]["status"] != "passed"
                    ]
                    if regressions:
                        log.warning("REGRESSION: %d previously-passing tests broke; rolling back fixes",
                                    len(regressions))
                        # roll back ALL fixes from this round (conservative).
                        for tid, snap in task_snapshots.items():
                            self.sandbox.restore(snap)
                        for r in qa_fix_results:
                            r.error = (r.error + " | rolled back due to regression") if r.error else "rolled back due to regression"
                            r.status = "failed"
                        # re-run tests after rollback so the rerun file shows truth
                        post_runs = {p: self.qa._run_test_file(p) for p in all_tests}

                    self._dump(f"qa_rerun_iter{it}.json", post_runs)
                    self._dump(f"qa_regressions_iter{it}.json", regressions)
                    still_open = [b for b in (qa_report.new_bugs + qa_report.open_bugs)
                                  if post_runs.get(b.test_path, {}).get("status") != "passed"]
                    log.info("After fix+regression-sweep: %d bugs still open",
                             len(still_open))
                    open_bugs = [b.to_dict() for b in still_open]
                else:
                    open_bugs = []

            history.append({
                "iteration": it,
                "plan_rationale": plan.rationale,
                "results": [r.to_dict() for r in results],
                "qa": qa_report.to_dict() if qa_report else None,
                "qa_fix_results": [r.to_dict() for r in qa_fix_results],
            })

            if done == 0 and not qa_fix_results:
                log.info("No tasks succeeded this iteration — stopping.")
                break

        summary = {
            "analysis": analysis.to_dict(),
            "iterations": history,
            "open_bugs": open_bugs,
            "run_dir": str(self.run_dir),
        }
        self._dump("summary.json", summary)
        log.info("d2p run complete. Artifacts in %s", self.run_dir)
        return summary

    # ---------------------------------------------------------------- internal

    def _run_qa(self, analysis) -> tuple[Any, list[Task]]:
        listing = [p for p in self.sandbox.listing(max_entries=200)
                   if not p.startswith("...")]
        # build the same context the Planner gets, so QA targets the right files
        key_files = self.planner._pick_key_files(listing)
        key_files_block = self.planner._build_key_files_block(key_files)
        symbol_map = build_symbol_map(self.sandbox.read, listing)
        return self.qa.run(
            analysis_summary=analysis.domain,
            essence=analysis.essence,
            audience=analysis.audience,
            key_files_block=key_files_block,
            symbol_map=symbol_map,
        )

    def _rerun_qa_tests(self, bugs) -> dict[str, dict[str, Any]]:
        out = {}
        for b in bugs:
            out[b.test_path] = self.qa._run_test_file(b.test_path)
        return out

    def _all_corpus_tests(self) -> list[str]:
        from .qa import CORPUS_DIR
        return [p for p in self.sandbox.listing(max_entries=400)
                if p.startswith(CORPUS_DIR + "/")
                and p.endswith(".py")
                and not p.endswith("__init__.py")
                and not p.endswith("conftest.py")]

    def _test_baseline(self) -> dict[str, bool]:
        """For each test file, True if it currently passes."""
        out: dict[str, bool] = {}
        for p in self._all_corpus_tests():
            out[p] = (self.qa._run_test_file(p)["status"] == "passed")
        return out

    def _qa_fix_post_check(self, bug_test_path: str):
        def _check() -> tuple[bool, str]:
            r = self.qa._run_test_file(bug_test_path)
            ok = (r["status"] == "passed")
            # P3: flip _meta.json status the moment the test goes green, so
            # statistics reflect reality this iteration instead of waiting
            # for the next QA sweep to discover it.
            if ok:
                try:
                    self.qa.flip_meta_status(bug_test_path, "fixed")
                except Exception as e:
                    log.warning("status flip failed for %s: %s", bug_test_path, e)
            return (ok, r.get("output", ""))
        return _check

    def _run_tasks_parallel(self, tasks: list[Task], *,
                            post_check_for=None) -> list[ExecutionResult]:
        """Run feature/fix tasks in parallel.

        Performance design (the 2026-05-18 rewrite):
        - Health baseline + demo-test baseline probed ONCE here, cached, shared
          by all tasks. Previously they were re-probed inside every task's lock.
        - Only the executor.run() WRITE phase stays under the per-file lock.
          Post-task health-probe + rollback now run OUTSIDE the lock so they
          don't serialise parallel tasks targeting different files.
        - Default parallel_executors bumped to 4.
        """
        max_workers = max(1, min(self.cfg.parallel_executors, len(tasks)))
        executor_llm = self.router.for_role("executor")
        executors = [Executor(executor_llm, self.sandbox)
                     for _ in range(max_workers)]
        locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        results: list[ExecutionResult] = []
        results_lock = threading.Lock()
        ordered = sorted(tasks, key=lambda t: (t.priority, -len(t.target_files)))

        # B) Cache iter-level baselines once (was: per-task inside lock).
        cached_health_baseline = self.health.probe(self.health.default_modules())
        cached_test_baseline: dict[str, bool] = {}
        if self.qa is not None:
            for tp in _discover_pre_existing_tests(self.sandbox):
                cached_test_baseline[tp] = (
                    self.qa._run_test_file(tp)["status"] == "passed"
                )

        def _run(idx: int, task: Task) -> None:
            keys = sorted(set(task.target_files)) or [f"__notarget__:{task.id}"]
            acquired: list[threading.Lock] = []
            pc = post_check_for(task) if post_check_for else None
            snapshot: dict[str, Any] = {}
            try:
                for k in keys:
                    lk = locks[k]
                    lk.acquire()
                    acquired.append(lk)
                # A) Snapshot is the only thing that MUST be inside the write
                # lock — it has to capture the pre-write state atomically with
                # the executor's writes. Health probes happen after release.
                snapshot = self.sandbox.snapshot(task.target_files)
                try:
                    res = executors[idx % max_workers].run(task, post_check=pc)
                except Exception as e:
                    res = ExecutionResult(task_id=task.id, status="failed",
                                          summary="", error=str(e))
            finally:
                for lk in acquired:
                    lk.release()

            # Post-task probes — OUTSIDE the lock (multiple tasks' subprocesses
            # can run concurrently; this is the main speed win).
            if res.status == "done":
                rolled = _rollback_if_health_regressed(
                    self.sandbox, self.health,
                    baseline=cached_health_baseline, snapshot=snapshot,
                )
                if not rolled and cached_test_baseline and self.qa is not None:
                    rolled = _rollback_if_baseline_test_regressed(
                        self.sandbox, self.qa,
                        baseline=cached_test_baseline, snapshot=snapshot,
                    )
                if rolled:
                    res.status = "failed"
                    res.error = ((res.error + " | ") if res.error else "") + \
                                "regression detected — rolled back"
                    res.files_changed = []

            log.info("task %s [%s] -> %s (%d files)",
                     task.id, task.title[:60], res.status, len(res.files_changed))
            with results_lock:
                results.append(res)

        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_run, i, t) for i, t in enumerate(ordered)]
            for f in cf.as_completed(futs):
                f.result()
        return results

    def _dump(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
