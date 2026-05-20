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
        self.run_dir = self.sandbox.root / ".d2p" / time.strftime("run-%Y%m%d-%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
        log.info("Analyzer (%s): essence=%r audience=%r features=%d",
                 "cache HIT" if hit else "fresh",
                 analysis.essence[:80], analysis.audience, len(analysis.features))

        history: list[dict[str, Any]] = []
        open_bugs: list[dict[str, Any]] = []
        # Cumulative cost at the start of each iter — diff with the next
        # measurement gives the per-iter cost delta (cleaner than guessing
        # from per-call records).
        last_cost = self.router.usage.summary()["total_cost_usd"]

        bug_debt_threshold = 12        # P2: raised from 6 → 12
        always_min_features = 1        # P2: even under debt, always allow ≥1
        for it in range(1, self.cfg.max_iterations + 1):
            iter_started = time.monotonic()
            log.info("=== iteration %d / %d ===", it, self.cfg.max_iterations)

            # Periodic re-analysis: refresh feature list mid-run, but preserve
            # essence/audience invariants (changing them mid-run would defeat
            # the whole purpose of having them).
            if (self.cfg.reanalyze_every and it > 1
                    and (it - 1) % self.cfg.reanalyze_every == 0):
                log.info("Re-running Analyzer (reanalyze_every=%d)",
                         self.cfg.reanalyze_every)
                try:
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
        log.info("d2p run complete in %.1fs. Artifacts in %s",
                 total_elapsed_s, self.run_dir)
        log.info("Usage: %d calls, $%.4f total, cache_hit=%s",
                 summary["usage"]["total_calls"],
                 summary["usage"]["total_cost_usd"],
                 summary["usage"]["cache_hit_ratio"])
        return summary

    # ---------------------------------------------------------------- internal

    def _run_qa(self, analysis, *, iteration: int) -> tuple[Any, list[Task]]:
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
            fix_done = [r for r in qa_fix_results if r.status == "done"]
            lines.append(
                f"## QA fixes ({len(fix_done)}/{len(qa_fix_results)} succeeded)"
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
        executors = [Executor(executor_llm, self.sandbox,
                              usage=self.router.usage)
                     for _ in range(max_workers)]
        fallback_llm = self.router.for_fallback(role)
        if fallback_llm is not None:
            log.info("Escalation available (role=%s): fallback=%s",
                     role, fallback_llm.name)
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

        def _run(idx: int, task: Task) -> None:
            pc = post_check_for(task) if post_check_for else None
            res, _snapshot = _attempt(task, executors[idx % max_workers], pc)

            # Escalation: if the primary executor failed AND a fallback model
            # is wired for this role AND the failure is the kind a stronger
            # model could plausibly fix, retry once.
            # The sandbox is already rolled back to the pre-task state, so the
            # second attempt starts from the same baseline as the first.
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
