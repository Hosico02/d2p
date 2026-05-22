"""Closed-loop driver: Analyzer -> Planner -> Executors -> QA -> Fix-Executors -> repeat."""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from .agents import Analyzer, Executor, Planner
from .config import Config
from .fs import Sandbox
from .health import ProjectHealth
from .models import ExecutionResult, PlanResult, Task
from .providers import RoleRouter, build_router
from .qa import QAAgent
from .report import write_report
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


# Error-message substrings that indicate a *structural* failure (executor
# couldn't even attempt the work — wrong task framing, forbidden target,
# empty plan, etc.). For these, retrying with a stronger model wastes
# tokens because the constraint isn't model-quality.
_STRUCTURAL_FAILURE_MARKERS = (
    "forbidden (test file",   # tried to write a QA-protected test
    "forbidden (",            # any forbidden_files violation
    "no target files",        # empty plan output
    "no ===FILE===",          # parser found zero blocks
    "no ===PATCH===",
    "path escapes sandbox",   # tried to write outside sandbox
)


def _should_escalate(error: str) -> bool:
    """Decide whether a task failure is worth retrying with the fallback
    model. Structural failures (forbidden file, wrong framing, sandbox
    escape) won't fix themselves with a stronger model — skip the retry.
    Everything else (regression rolled back, SEARCH miss, post-check fail,
    syntax error, LLM hiccup) is worth one more shot.
    """
    if not error:
        return True
    e = error.lower()
    return not any(marker in e for marker in _STRUCTURAL_FAILURE_MARKERS)


def _flatten(msg: str, max_len: int = 200) -> str:
    """Single-line, length-capped form of an error string. Without flattening,
    a multi-line Traceback inside the log line shreds grep + monitor output
    because each line of the traceback starts a new log entry."""
    if not msg:
        return ""
    s = msg.replace("\r", " ").replace("\n", " | ")
    return s[:max_len]


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
                 use_analyzer_cache: bool = True,
                 resume_from: str | Path | None = None,
                 router: RoleRouter | None = None) -> None:
        self.cfg = cfg or Config()
        if parallel is not None:
            self.cfg.parallel_executors = parallel
        if max_iterations is not None:
            self.cfg.max_iterations = max_iterations
        self.enable_qa = enable_qa
        self.use_analyzer_cache = use_analyzer_cache
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
        # Resume from a prior run_dir vs start a fresh one. On resume, we
        # reuse the existing dir (so per-iter JSON dumps accumulate) and
        # mark `resume_from_iter` so run() knows to rebuild history from
        # the per-iter files on disk. On fresh start, mkdir a new
        # timestamped dir as before.
        if resume_from is not None:
            self.run_dir = Path(resume_from).resolve()
            if not self.run_dir.is_dir():
                raise FileNotFoundError(f"resume_from not a dir: {self.run_dir}")
            log.info("Resuming from %s", self.run_dir)
        else:
            self.run_dir = self.sandbox.root / ".d2p" / time.strftime("run-%Y%m%d-%H%M%S")
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._resume = resume_from is not None
        # Background thread pool for prefetching next-iter Analyzer.run()
        # while the current iter's executor/qa/fix work runs. Saves the
        # ~50-70s Analyzer wait when --reanalyze-every is set.
        self._bg_pool = cf.ThreadPoolExecutor(max_workers=1,
                                              thread_name_prefix="d2p-bg")
        self._next_analysis_future: cf.Future | None = None
        # Race-mode pool. Long-lived so that when the winning side
        # commits early, we can abandon (not wait on) the slow side's
        # prepare(). Sized for the worst case: parallel tasks × 2 sides.
        # Pool keeps running threads even after we stop caring about
        # their results; on orchestrator shutdown the daemon nature of
        # subprocess timeouts caps the leak.
        self._race_pool = cf.ThreadPoolExecutor(
            max_workers=max(2, self.cfg.parallel_executors * 2),
            thread_name_prefix="d2p-race",
        )

    # ------------------------------------------------------------------ public

    def run(self) -> dict[str, Any]:
        run_started = time.monotonic()
        log.info("d2p starting on %s", self.sandbox.root)
        # Persistent analyzer cache keyed by codebase fingerprint. Lives
        # alongside the target's .d2p dir but OUTSIDE this run's dir, so
        # subsequent runs against the same demo skip the (slow + costly)
        # web-search-fueled re-analysis when nothing changed.
        cache_path = self.sandbox.root / ".d2p" / "analysis_cache.json"
        _t0 = time.monotonic()
        analysis, hit = self.analyzer.run_cached(
            cache_path, use_cache=self.use_analyzer_cache,
        )
        analyzer_elapsed = round(time.monotonic() - _t0, 1)
        self._dump("analysis.json", analysis.to_dict())
        self._dump("competitors.json",
                   [c.to_dict() for c in analysis.competitors_detail])
        self._dump("capabilities.json", analysis.demo_capabilities)
        self._dump("gap_matrix.json", [f.to_dict() for f in analysis.features])
        log.info(
            "Analyzer (%s): essence=%r audience=%r "
            "competitors=%d capabilities=%d features=%d (gap_high=%d, partial=%d)",
            "cache HIT" if hit else "fresh",
            analysis.essence[:80], analysis.audience,
            len(analysis.competitors_detail),
            len(analysis.demo_capabilities),
            len(analysis.features),
            sum(1 for f in analysis.features if f.gap_severity == "high"),
            sum(1 for f in analysis.features if f.in_demo == "partial"),
        )

        history: list[dict[str, Any]] = []
        open_bugs: list[dict[str, Any]] = []
        # On resume, rebuild history from the per-iter JSON dumps already
        # on disk. We pick the lowest iter number that lacks a complete
        # exec/qa set and continue from there. Iters with full artifacts
        # are loaded into `history` verbatim so the Planner sees the
        # same prior context it would have seen.
        resume_from_iter = 1
        if self._resume:
            history, resume_from_iter, open_bugs = self._reload_history()
            log.info("Resume: rebuilt %d completed iters, continuing from iter %d, "
                     "%d open bugs",
                     len(history), resume_from_iter, len(open_bugs))
        # Cumulative cost at the start of each iter — diff with the next
        # measurement gives the per-iter cost delta (cleaner than guessing
        # from per-call records).
        last_cost = self.router.usage.summary()["total_cost_usd"]

        bug_debt_threshold = 12        # P2: raised from 6 → 12
        always_min_features = 1        # P2: even under debt, always allow ≥1
        for it in range(resume_from_iter, self.cfg.max_iterations + 1):
            iter_started = time.monotonic()
            log.info("=== iteration %d / %d ===", it, self.cfg.max_iterations)

            # Periodic re-analysis: refresh feature list mid-run, but preserve
            # essence/audience invariants (changing them mid-run would defeat
            # the whole purpose of having them).
            #
            # The Analyzer.run() for THIS iter was kicked off as a background
            # future at the END of the previous iter (see `_schedule_next_analysis`).
            # By the time we get here, it's usually already done — we just
            # collect the result instead of blocking another ~50-70s.
            if (self.cfg.reanalyze_every and it > 1
                    and (it - 1) % self.cfg.reanalyze_every == 0):
                log.info("Picking up prefetched Analyzer (reanalyze_every=%d)",
                         self.cfg.reanalyze_every)
                try:
                    if self._next_analysis_future is not None:
                        fresh = self._next_analysis_future.result()
                        self._next_analysis_future = None
                    else:
                        # First trigger or prefetch missed — fall back to sync
                        fresh = self.analyzer.run()
                    fresh.essence = analysis.essence
                    fresh.audience = analysis.audience
                    analysis = fresh
                    self._dump(f"analysis_iter{it}.json", analysis.to_dict())
                    log.info("Re-analyzed: features now %d",
                             len(analysis.features))
                except Exception as e:
                    log.warning("re-analysis failed, keeping prior: %s", e)

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

            # Per-stage timings — surfaced in iter md and summary.json so
            # users can see where each iter spent its wall-clock budget.
            stage_t: dict[str, float] = {}
            _t = time.monotonic()
            plan = self.planner.run(
                analysis, iteration=it,
                max_iter=self.cfg.max_iterations,
                history=history, open_bugs=open_bugs,
                feature_cap=feature_cap,
            )
            stage_t["planner_s"] = round(time.monotonic() - _t, 1)
            self._dump(f"plan_iter{it}.json", plan.to_dict())
            log.info("Planner produced %d tasks (cap=%d) in %.1fs",
                     len(plan.tasks), feature_cap, stage_t["planner_s"])
            # P2: defence in depth — Planner already targets <= feature_cap,
            # but trim+resort here in case it overshot. Small/low-risk first.
            plan.tasks = sorted(plan.tasks,
                                key=lambda t: (t.priority, len(t.target_files)))[:feature_cap]
            if not plan.tasks:
                log.info("Planner emitted no tasks — converged.")
                break
            _t = time.monotonic()
            results = self._run_tasks_parallel(plan.tasks)
            stage_t["executor_s"] = round(time.monotonic() - _t, 1)

            self._dump(f"exec_iter{it}.json", [r.to_dict() for r in results])
            done = sum(1 for r in results if r.status == "done")
            log.info("Iteration %d: %d/%d feature tasks done in %.1fs",
                     it, done, len(results), stage_t["executor_s"])

            qa_report = None
            qa_fix_results: list[ExecutionResult] = []
            regressions: list[dict[str, Any]] = []
            retired_this_iter: list[str] = []
            if self.qa is not None:
                _t = time.monotonic()
                qa_report, fix_tasks = self._run_qa(analysis, iteration=it)
                stage_t["qa_s"] = round(time.monotonic() - _t, 1)
                self._dump(f"qa_iter{it}.json", qa_report.to_dict())
                log.info("QA: new_bugs=%d open_bugs=%d fixed=%d retired=%d",
                         len(qa_report.new_bugs), len(qa_report.open_bugs),
                         len(qa_report.fixed_bugs),
                         len(qa_report.retired_bugs))
                if fix_tasks:
                    # Optional per-iter fix cap. Each fix can trigger an
                    # escalation, which is the most expensive single call in
                    # the system; without a cap one iter can dispatch 6 fixes
                    # × $0.10/escalation = serious budget. Cap qa-* tasks
                    # only — restore-symbol tasks (id="restore-*") are cheap
                    # mechanical edits that other tests depend on, so they
                    # always run.
                    cap = self.cfg.max_concurrent_fixes
                    if cap:
                        restore_tasks = [t for t in fix_tasks
                                         if t.id.startswith("restore-")]
                        bug_fix_tasks = [t for t in fix_tasks
                                         if t.id.startswith("qa-")]
                        if len(bug_fix_tasks) > cap:
                            meta = self.qa._load_meta()
                            def _attempts_of(t):
                                # O(1) lookup via the test_path embedded in
                                # the task's forbidden_files (set by
                                # QAAgent._bug_to_task). No string endswith
                                # heuristics, no O(N×M) scan.
                                if not t.forbidden_files:
                                    return 0
                                return int(meta.get(t.forbidden_files[0], {})
                                               .get("attempts", 0) or 0)
                            bug_fix_tasks.sort(
                                key=lambda t: (_attempts_of(t), t.priority))
                            log.info("Fix cap: keeping %d/%d bug-fix tasks "
                                     "this iter (by lowest attempts; %d "
                                     "restore tasks exempt)",
                                     cap, len(bug_fix_tasks), len(restore_tasks))
                            bug_fix_tasks = bug_fix_tasks[:cap]
                        fix_tasks = restore_tasks + bug_fix_tasks
                    # 1) snapshot the entire test-result baseline + all files
                    # the fix tasks might touch, BEFORE running fixes.
                    fix_target_files = sorted({
                        f for t in fix_tasks for f in t.target_files
                    })
                    pre_snapshot = self.sandbox.snapshot(fix_target_files)
                    # Reuse the corpus test outcomes QA just produced —
                    # qa_report.test_runs is the same data _test_baseline()
                    # would re-compute, but free (we already paid for those
                    # subprocess runs during the QA sweep).
                    pre_pass_tests = {
                        p: (outcome.get("status") == "passed")
                        for p, outcome in (qa_report.test_runs or {}).items()
                    }
                    log.info("Regression baseline: %d tests currently passing "
                             "(reused from QA sweep)",
                             sum(1 for v in pre_pass_tests.values() if v))

                    # 2) per-task snapshot so we can selectively rollback
                    task_snapshots: dict[str, dict[str, Any]] = {}
                    for t in fix_tasks:
                        task_snapshots[t.id] = self.sandbox.snapshot(t.target_files)

                    # 3) map task -> bug.test_path so post_check knows what
                    # to run. Pull the test_path directly from each task's
                    # forbidden_files (set by QAAgent._bug_to_task) — robust
                    # against fix-cap re-ordering, restore_tasks interleaving,
                    # and any future task reshuffling. Restore tasks have
                    # empty forbidden_files so they're correctly absent here.
                    bug_test_paths: dict[str, str] = {}
                    for t in fix_tasks:
                        if t.id.startswith("qa-") and t.forbidden_files:
                            bug_test_paths[t.id] = t.forbidden_files[0]

                    def _pc_for(t):
                        path = bug_test_paths.get(t.id)
                        return self._qa_fix_post_check(path) if path else None

                    _t = time.monotonic()
                    qa_fix_results = self._run_tasks_parallel(
                        fix_tasks, post_check_for=_pc_for, role="fix")
                    stage_t["fix_s"] = round(time.monotonic() - _t, 1)
                    self._dump(f"qa_fix_iter{it}.json",
                               [r.to_dict() for r in qa_fix_results])

                    # 3) re-run the WHOLE corpus to detect regressions
                    _t = time.monotonic()
                    all_tests = self._all_corpus_tests()
                    post_runs = {p: self.qa._run_test_file(p) for p in all_tests}
                    stage_t["regression_sweep_s"] = round(time.monotonic() - _t, 1)
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

                    # Retire bugs that have racked up too many failed attempts.
                    # The test stays in the corpus (so it still flags if it
                    # accidentally turns green), but no more fix tasks are
                    # generated for it — frees the next iters to work on
                    # features and bugs that haven't burned the budget yet.
                    #
                    # IMPORTANT: only bump `attempts` for bugs that were
                    # actually DISPATCHED as fix tasks this iter. Bugs
                    # deferred by max_concurrent_fixes (or absent for any
                    # other reason) shouldn't count — otherwise the wontfix
                    # threshold trips on bugs we never actually tried to fix.
                    dispatched_test_paths = set(bug_test_paths.values())
                    threshold = self.cfg.qa_wontfix_after_attempts
                    survivors: list = []
                    for b in still_open:
                        if b.test_path in dispatched_test_paths:
                            n = self.qa.bump_attempts(b.test_path)
                        else:
                            # deferred: keep prior attempts unchanged. Read
                            # the current value so the threshold check below
                            # still works if attempts are already past it
                            # from earlier iters.
                            n = int(self.qa._load_meta()
                                       .get(b.test_path, {})
                                       .get("attempts", 0) or 0)
                        if threshold and n >= threshold:
                            self.qa.mark_wontfix(b.test_path)
                            retired_this_iter.append(b.test_path)
                        else:
                            survivors.append(b)
                    if retired_this_iter:
                        log.info("Retired %d bug(s) as wontfix (attempts >= %d): %s",
                                 len(retired_this_iter), threshold,
                                 retired_this_iter)
                    log.info("After fix+regression-sweep: %d bugs still open, %d retired",
                             len(survivors), len(retired_this_iter))
                    open_bugs = [b.to_dict() for b in survivors]
                else:
                    open_bugs = []

            done_fixes = sum(1 for r in qa_fix_results if r.status == "done")
            iter_elapsed_s = round(time.monotonic() - iter_started, 1)
            cur_cost = self.router.usage.summary()["total_cost_usd"]
            iter_cost_delta = round(cur_cost - last_cost, 4)
            last_cost = cur_cost
            log.info("Iter %d done in %.1fs (cost delta=$%.4f, cum=$%.4f)",
                     it, iter_elapsed_s, iter_cost_delta, cur_cost)
            history.append({
                "iteration": it,
                "elapsed_s": iter_elapsed_s,
                "stage_timings": stage_t,
                "cost_delta_usd": iter_cost_delta,
                "cumulative_cost_usd": cur_cost,
                "plan_rationale": plan.rationale,
                "results": [r.to_dict() for r in results],
                "qa": qa_report.to_dict() if qa_report else None,
                "qa_fix_results": [r.to_dict() for r in qa_fix_results],
                "retired_this_iter": retired_this_iter,
            })

            # End-of-iter change digest — what tasks ran, what files moved,
            # what bugs got found/fixed/retired. Useful for the user to scan
            # a long run without spelunking through JSON dumps.
            self._emit_iter_changes_md(
                it, plan=plan, results=results, qa_report=qa_report,
                qa_fix_results=qa_fix_results,
                retired_this_iter=retired_this_iter,
                still_open_count=len(open_bugs),
                elapsed_s=iter_elapsed_s,
                cost_delta_usd=iter_cost_delta,
                stage_timings=stage_t,
            )

            # Convergence: stop iterating when NEITHER side made forward
            # progress. Previous check only looked at "no fix tasks ran",
            # which kept burning iters when fixes ran but all failed.
            if done == 0 and done_fixes == 0:
                log.info("Converged: iter %d had 0 feature wins and 0 fix wins.",
                         it)
                break

            # If the NEXT iter will trigger a re-analysis, kick off
            # Analyzer.run() now in the background so it overlaps with
            # the next iter's planner/executor/qa work. The result is
            # collected at the top of the next iter.
            next_it = it + 1
            if (self.cfg.reanalyze_every and next_it > 1
                    and (next_it - 1) % self.cfg.reanalyze_every == 0
                    and self._next_analysis_future is None):
                log.info("Prefetching Analyzer for iter %d in background",
                         next_it)
                self._next_analysis_future = self._bg_pool.submit(
                    self.analyzer.run)

        total_elapsed_s = round(time.monotonic() - run_started, 1)
        summary = {
            "analysis": analysis.to_dict(),
            "iterations": history,
            "open_bugs": open_bugs,
            "run_dir": str(self.run_dir),
            "elapsed_s": total_elapsed_s,
            "analyzer_elapsed_s": analyzer_elapsed,
            "analyzer_cache_hit": hit,
            "usage": self.router.usage.summary(),
        }
        self._dump("summary.json", summary)
        # Render a self-contained HTML report alongside the JSON dump.
        # Single-file, no external CSS/JS, openable from disk — convenient
        # for sharing without zipping the whole run_dir.
        try:
            write_report(summary, self.run_dir / "report.html")
        except Exception as e:
            log.warning("HTML report write failed: %s", e)
        log.info("d2p run complete in %.1fs. Artifacts in %s",
                 total_elapsed_s, self.run_dir)
        u = cast(dict[str, Any], summary["usage"])
        log.info("Usage: %d calls, $%.4f total, cache_hit=%s",
                 u["total_calls"], u["total_cost_usd"], u["cache_hit_ratio"])
        # Drain any outstanding bg prefetch — its result is now useless,
        # but the thread should exit cleanly before we return.
        self._bg_pool.shutdown(wait=False, cancel_futures=True)
        return summary

    # ---------------------------------------------------------------- internal

    def _run_qa(self, analysis, *, iteration: int) -> tuple[Any, list[Task]]:
        assert self.qa is not None, "_run_qa called with QA disabled"
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
            iteration=iteration,
        )

    def _emit_iter_changes_md(self, it: int, *,
                              plan, results: list[ExecutionResult],
                              qa_report, qa_fix_results: list[ExecutionResult],
                              retired_this_iter: list[str],
                              still_open_count: int = 0,
                              elapsed_s: float = 0.0,
                              cost_delta_usd: float = 0.0,
                              stage_timings: dict[str, float] | None = None) -> None:
        """Write iter{N}_changes.md — a human-readable digest of what moved
        this iteration. Far easier to skim than the JSON dumps when reviewing
        a multi-iter run."""
        lines: list[str] = []
        lines.append(f"# Iteration {it} — changes")
        lines.append("")
        lines.append(f"Elapsed: {elapsed_s:.1f}s  •  Cost delta: ${cost_delta_usd:.4f}")
        if stage_timings:
            # Render as planner=Xs, executor=Ys, qa=Zs, fix=Ws — instantly
            # shows where the iter's time went without scrolling JSON.
            parts = [f"{k.removesuffix('_s')}={v:.1f}s"
                     for k, v in stage_timings.items()]
            lines.append("Stage timings: " + ", ".join(parts))
        lines.append("")
        lines.append(f"Rationale: {plan.rationale or '(none)'}")
        lines.append("")

        # Feature tasks
        done = [r for r in results if r.status == "done"]
        failed = [r for r in results if r.status != "done"]
        lines.append(f"## Feature tasks ({len(done)}/{len(results)} succeeded)")
        title_by_id = {t.id: t.title for t in plan.tasks}
        if results:
            for r in results:
                mark = "OK" if r.status == "done" else "X "
                title = title_by_id.get(r.task_id, r.task_id)
                files = ", ".join(r.files_changed) or "(no files)"
                line = f"- [{mark}] {title} — files: {files}"
                if r.status != "done" and r.error:
                    line += f"\n      error: {r.error[:160]}"
                lines.append(line)
        else:
            lines.append("- (none)")
        lines.append("")

        # QA fixes
        if qa_report is not None:
            done_list = [r for r in qa_fix_results if r.status == "done"]
            lines.append(
                f"## QA fixes ({len(done_list)}/{len(qa_fix_results)} succeeded)"
            )
            if qa_fix_results:
                for r in qa_fix_results:
                    mark = "OK" if r.status == "done" else "X "
                    files = ", ".join(r.files_changed) or "(no files)"
                    line = f"- [{mark}] {r.task_id} — files: {files}"
                    if r.status != "done" and r.error:
                        line += f"\n      error: {r.error[:160]}"
                    lines.append(line)
            else:
                lines.append("- (no fix tasks dispatched)")
            lines.append("")

            # Bug flow summary. Each label here means exactly one thing —
            # earlier versions of this section conflated "open at QA entry"
            # with "open after fix sweep", giving counts that didn't match
            # reality. Now the labels make the lifecycle explicit:
            #   carried in   = bugs failing entering this iter (from prior runs)
            #   new          = bugs first discovered this iter
            #   incidentally fixed = bugs that turned green via feature work,
            #                  i.e. the QA regression sweep found them passing
            #                  before any fix task ran for them
            #   fix tasks    = qa-* tasks dispatched this iter (capped+escalated)
            #   retired      = bugs flipped to wontfix this iter
            #   still open going forward = the count carried into next iter
            fix_done = sum(1 for r in qa_fix_results if r.status == "done")
            fix_failed = sum(1 for r in qa_fix_results if r.status != "done")
            lines.append("## Bugs")
            lines.append(f"- carried in (open from prior iters): "
                         f"{len(qa_report.open_bugs)}")
            lines.append(f"- new this iter: {len(qa_report.new_bugs)}")
            lines.append(f"- incidentally fixed (passed before fix sweep): "
                         f"{len(qa_report.fixed_bugs)}")
            lines.append(f"- fix tasks: {fix_done} ok, {fix_failed} failed")
            lines.append(f"- retired (wontfix) this iter: {len(retired_this_iter)}")
            if retired_this_iter:
                for tp in retired_this_iter:
                    lines.append(f"    - {tp}")
            lines.append(f"- **still open going forward: {still_open_count}**")
            lines.append("")
            if qa_report.new_bugs:
                lines.append("### New bugs")
                for b in qa_report.new_bugs:
                    lines.append(f"- {b.test_path}: {b.title}")
                lines.append("")
            if qa_report.fixed_bugs:
                lines.append("### Fixed bugs")
                for b in qa_report.fixed_bugs:
                    lines.append(f"- {b.test_path}: {b.title}")
                lines.append("")

        # File-level digest (collapse files_changed across all results)
        all_files: dict[str, list[str]] = {}
        for r in results + qa_fix_results:
            if r.status != "done":
                continue
            for f in r.files_changed:
                all_files.setdefault(f, []).append(r.task_id)
        if all_files:
            lines.append("## Files touched")
            for f, tids in sorted(all_files.items()):
                lines.append(f"- `{f}` ({len(tids)} task: {', '.join(tids)})")
            lines.append("")

        # Per-iter usage delta — useful for spotting cost spikes
        try:
            usage = self.router.usage.summary()
            lines.append("## Cumulative usage")
            lines.append(f"- total calls: {usage['total_calls']}")
            lines.append(f"- total cost: ${usage['total_cost_usd']:.4f}")
            lines.append(f"- cache hit ratio: {usage['cache_hit_ratio']}")
            lines.append("")
        except Exception:
            pass

        (self.run_dir / f"iter{it}_changes.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _rerun_qa_tests(self, bugs) -> dict[str, dict[str, Any]]:
        assert self.qa is not None
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
        assert self.qa is not None
        out: dict[str, bool] = {}
        for p in self._all_corpus_tests():
            out[p] = (self.qa._run_test_file(p)["status"] == "passed")
        return out

    def _qa_fix_post_check(self, bug_test_path: str):
        qa = self.qa
        assert qa is not None
        def _check() -> tuple[bool, str]:
            r = qa._run_test_file(bug_test_path)
            ok = (r["status"] == "passed")
            # P3: flip _meta.json status the moment the test goes green, so
            # statistics reflect reality this iteration instead of waiting
            # for the next QA sweep to discover it.
            if ok:
                try:
                    qa.flip_meta_status(bug_test_path, "fixed")
                except Exception as e:
                    log.warning("status flip failed for %s: %s", bug_test_path, e)
            return (ok, r.get("output", ""))
        return _check

    def _run_tasks_parallel(self, tasks: list[Task], *,
                            post_check_for=None,
                            role: str = "executor") -> list[ExecutionResult]:
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
        executor_llm = self.router.for_role(role)
        log.info("Task batch (role=%s) using %s", role, executor_llm.name)
        # Per-task fresh Executor instances. The internal max_fix_attempts loop
        # plus the optional escalation-to-fallback path own the task end-to-end;
        # only on `done` (or final escalation failure) does the instance get
        # released. No cross-task reuse — guarantees clean state per task.
        fallback_llm = self.router.for_fallback(role)
        if fallback_llm is not None:
            log.info("Escalation available (role=%s): fallback=%s",
                     role, fallback_llm.name)
        # Race mode: kick off primary + fallback prepare() in parallel so
        # the escalation LLM call doesn't add to wall time. Active when
        # this `role` is in cfg.race_roles AND a fallback is configured.
        # Inside commit(), we cap max_fix_attempts=1 so we don't compound
        # race × MAX_FIX_ATTEMPTS=3 (the original opt-in --fix-race bug).
        race_on = role in self.cfg.race_roles and fallback_llm is not None
        if race_on:
            log.info("Race mode ENABLED (role=%s): primary + fallback "
                     "prepare() in parallel; max_fix_attempts=1", role)
        elif role in self.cfg.race_roles:
            log.info("Race mode requested for role=%s but no fallback "
                     "configured — skipping race", role)
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

        def _attempt(task: Task, executor: Executor,
                     pc) -> tuple[ExecutionResult, dict[str, Any]]:
            """One attempt with the LLM call OUTSIDE the per-file lock.

            Phase 1 (unlocked): executor.prepare() reads target files and
                runs the LLM. This is the slow part — moving it out of
                the lock is the whole point of the prepare/commit split.
                Two tasks targeting the same file now spend their LLM
                budget in parallel.

            Phase 2 (locked): executor.commit() writes / patches / does
                syntax check + self-heal + post_check. Cheap, contended
                on the actual file. FILE-mode writes refuse to clobber
                if the file changed since prepare() read it; PATCH-mode
                handles concurrency naturally via SEARCH/REPLACE.

            Phase 3 (unlocked): health + baseline-test probes / rollback.
            """
            # --- Phase 1: prepare (LLM call) — no lock held ---
            try:
                prepared = executor.prepare(task)
            except Exception as e:
                res = ExecutionResult(task_id=task.id, status="failed",
                                      summary="", error=f"prepare: {e}")
                return res, {}

            # --- Phase 2: commit (writes) under per-file lock ---
            keys = sorted(set(task.target_files)) or [f"__notarget__:{task.id}"]
            acquired: list[threading.Lock] = []
            snapshot: dict[str, Any] = {}
            try:
                for k in keys:
                    lk = locks[k]
                    lk.acquire()
                    acquired.append(lk)
                snapshot = self.sandbox.snapshot(task.target_files)
                try:
                    res = executor.commit(prepared, post_check=pc)
                except Exception as e:
                    res = ExecutionResult(task_id=task.id, status="failed",
                                          summary="", error=f"commit: {e}")
            finally:
                for lk in acquired:
                    lk.release()

            # --- Phase 3: probes / rollback (unlocked) ---
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
            return res, snapshot

        def _commit_phase(task: Task, executor: Executor, prepared,
                          pc, max_fix_attempts: int | None = None,
                          ) -> tuple[ExecutionResult, dict[str, Any]]:
            """Locked write phase + post-task probes. Used by both the
            normal path and the race path.

            `max_fix_attempts=1` is the race-mode cap that prevents
            race × Executor-retry compounding.
            """
            keys = sorted(set(task.target_files)) or [f"__notarget__:{task.id}"]
            acquired: list[threading.Lock] = []
            snapshot: dict[str, Any] = {}
            try:
                for k in keys:
                    lk = locks[k]
                    lk.acquire()
                    acquired.append(lk)
                snapshot = self.sandbox.snapshot(task.target_files)
                try:
                    res = executor.commit(prepared, post_check=pc,
                                          max_fix_attempts=max_fix_attempts)
                except Exception as e:
                    res = ExecutionResult(task_id=task.id, status="failed",
                                          summary="", error=f"commit: {e}")
            finally:
                for lk in acquired:
                    lk.release()
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
            return res, snapshot

        def _run(idx: int, task: Task) -> None:
            pc = post_check_for(task) if post_check_for else None

            if race_on:
                # Race v2: submit primary + fallback prepare() to a shared
                # long-lived pool, then iterate as_completed. As soon as
                # ONE side finishes successfully, attempt its commit; if
                # the commit succeeds we ABANDON the slower side (its
                # thread keeps running until the LLM call returns, but
                # we don't wait on it). If commit fails, fall through to
                # the next as_completed result.
                #
                # The race itself IS the retry, so commit() is capped at
                # max_fix_attempts=1 to avoid race × per-executor retry
                # compounding.
                assert fallback_llm is not None  # guarded by race_on
                fb_executor = Executor(fallback_llm, self.sandbox,
                                       usage=self.router.usage)
                primary_exec = Executor(executor_llm, self.sandbox,
                                        usage=self.router.usage)
                f_prim = self._race_pool.submit(primary_exec.prepare, task)
                f_fb = self._race_pool.submit(fb_executor.prepare, task)
                sides = {f_prim: ("primary", primary_exec),
                         f_fb: ("fallback", fb_executor)}
                res = None
                errors: list[str] = []
                for fut in cf.as_completed([f_prim, f_fb]):
                    side, ex = sides[fut]
                    try:
                        prep = fut.result()
                    except Exception as e:
                        errors.append(f"{side}.prepare: {_flatten(str(e), 120)}")
                        continue
                    if prep.status == "failed":
                        errors.append(f"{side}.prepare: {_flatten(prep.llm_error or 'no output', 120)}")
                        continue
                    cand, _ = _commit_phase(task, ex, prep, pc,
                                            max_fix_attempts=1)
                    if cand.status == "done":
                        cand.summary = (cand.summary or "") + f" [raced:{side}-won]"
                        res = cand
                        log.info("task %s race won by %s (other side abandoned)",
                                 task.id, side)
                        break
                    errors.append(f"{side}.commit: {_flatten(cand.error or '?', 120)}")
                if res is None:
                    res = ExecutionResult(
                        task_id=task.id, status="failed", summary="",
                        error="race: both sides failed — " + " ; ".join(errors),
                    )
            else:
                # Normal path: primary only; sequential escalation on failure.
                # Fresh Executor bound to this task for its full lifecycle
                # (prepare + commit + internal max_fix_attempts retries).
                primary_exec = Executor(executor_llm, self.sandbox,
                                        usage=self.router.usage)
                res, _snapshot = _attempt(task, primary_exec, pc)

                if res.status != "done" and fallback_llm is not None:
                    if _should_escalate(res.error):
                        log.info("task %s failed (%s) — escalating to %s",
                                 task.id, _flatten(res.error, 120),
                                 fallback_llm.name)
                        fb_executor = Executor(fallback_llm, self.sandbox,
                                               usage=self.router.usage)
                        fb_res, _ = _attempt(task, fb_executor, pc)
                        if fb_res.status == "done":
                            fb_res.summary = (fb_res.summary or "") + " [escalated]"
                            res = fb_res
                        else:
                            # keep the original error too — debug visibility for users
                            # who want to know which model failed which way
                            res.error = ((res.error or "") +
                                         " | escalation also failed: " +
                                         _flatten(fb_res.error, 120))
                    else:
                        log.info("task %s failed structurally (%s) — skipping escalation",
                                 task.id, _flatten(res.error, 120))

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

    def _reload_history(self) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
        """Rebuild orchestrator history from per-iter dumps on disk.

        Returns (history, resume_from_iter, open_bugs).

        An iter is considered "complete" iff at minimum its plan + exec
        JSON files both exist. Iters that have a plan but no exec (or
        an exec but no qa_rerun when QA is enabled) get redone — partial
        work is discarded so the rerun captures the full per-iter
        artifact set.
        """
        def _read(name: str) -> Any:
            p = self.run_dir / name
            if not p.is_file():
                return None
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None

        history: list[dict[str, Any]] = []
        open_bugs: list[dict[str, Any]] = []
        i = 1
        while True:
            plan = _read(f"plan_iter{i}.json")
            results = _read(f"exec_iter{i}.json")
            if not plan or results is None:
                # iter i is incomplete — restart from here
                break
            qa = _read(f"qa_iter{i}.json")
            qa_fixes = _read(f"qa_fix_iter{i}.json") or []
            qa_rerun = _read(f"qa_rerun_iter{i}.json") or {}
            history.append({
                "iteration": i,
                "plan_rationale": plan.get("rationale", ""),
                "results": results,
                "qa": qa,
                "qa_fix_results": qa_fixes,
                "retired_this_iter": [],
                # timings unknown post-mortem — best effort
                "elapsed_s": 0.0,
                "cost_delta_usd": 0.0,
                "cumulative_cost_usd": 0.0,
                "stage_timings": {},
            })
            # Re-derive open_bugs from the QA rerun for the most recent
            # iter — that's the state the next iter would have seen.
            if qa and qa_rerun:
                open_bugs = []
                bugs = (qa.get("new_bugs") or []) + (qa.get("open_bugs") or [])
                for b in bugs:
                    tp = b.get("test_path")
                    if not tp:
                        continue
                    if qa_rerun.get(tp, {}).get("status") != "passed":
                        open_bugs.append(b)
            i += 1
        return history, i, open_bugs
