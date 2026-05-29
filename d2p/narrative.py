"""Compact, human-narrative per-iteration summaries for the Hub Overview UI.

Pure functions over already-collected iteration data — no LLM, no I/O — so
they're unit-testable and cheap on the hot path. The detailed local digest
(orchestrator._emit_iter_changes_md) keeps its richer table form; this module
produces the compact prose the Hub renders. Both read the same data objects,
so there is one source of data truth.

Failure markers are intentional and load-bearing: the Hub's hasTrouble()
helper looks for "失败" (executor) and "未解决" (qa) to highlight a problem
iteration. Keep those substrings out of the no-trouble paths.
"""
from __future__ import annotations

from typing import Any, Optional

_ERR_CLIP = 160


def _clip(s: str, n: int = _ERR_CLIP) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def analyzer_line(analysis: Any, *, reanalyzed: bool) -> str:
    domain = (getattr(analysis, "domain", "") or "未知领域").strip()
    essence = (getattr(analysis, "essence", "") or "").strip()
    n_feat = len(getattr(analysis, "features", []) or [])
    n_comp = len(getattr(analysis, "competitors", []) or [])
    bits = [f"理解为 {domain}"]
    if essence:
        bits.append(f"本质 {essence}")
    bits.append(f"对标 {n_comp} 个竞品、{n_feat} 项能力")
    prefix = "(本轮重新分析) " if reanalyzed else ""
    return prefix + "；".join(bits)


def planner_line(plan: Any) -> str:
    tasks = list(getattr(plan, "tasks", []) or [])
    if not tasks:
        return "本轮无新特性任务"
    titles = "、".join(t.title for t in tasks[:3])
    suffix = "…" if len(tasks) > 3 else ""
    line = f"排了 {len(tasks)} 个任务：{titles}{suffix}"
    rationale = _clip(getattr(plan, "rationale", "") or "")
    if rationale:
        line += f"。理由：{rationale}"
    return line


def executor_line(results: Any, title_by_id: dict[str, str]) -> str:
    rows = list(results or [])
    if not rows:
        return "本轮未执行特性任务"
    done = [r for r in rows if r.status == "done"]
    failed = [r for r in rows if r.status != "done"]
    parts = [f"完成 {len(done)}/{len(rows)} 个特性任务"]
    if done:
        files = sorted({f for r in done for f in r.files_changed})
        if files:
            shown = "、".join(files[:5])
            more = f" 等 {len(files)} 个文件" if len(files) > 5 else ""
            parts.append(f"改动文件：{shown}{more}")
    for r in failed:
        title = title_by_id.get(r.task_id, r.task_id)
        msg = f"（{_clip(r.error)}）" if r.error else ""
        parts.append(f"失败：{title}{msg}")
    return "；".join(parts)


def qa_line(qa_report: Optional[Any], qa_fix_results: Any, *,
            still_open: int) -> str:
    if qa_report is None:
        return "本轮未跑 QA"
    fixes = list(qa_fix_results or [])
    fix_done = sum(1 for r in fixes if r.status == "done")
    fix_failed = sum(1 for r in fixes if r.status != "done")
    parts = [
        f"新增 {len(qa_report.new_bugs)} 个 bug",
        f"顺带修好 {len(qa_report.fixed_bugs)} 个",
    ]
    if fixes:
        parts.append(f"fix 任务 {fix_done} 成 {fix_failed} 败")
    parts.append(f"仍有 {still_open} 个未解决" if still_open else "全部清零")
    line = "；".join(parts)
    if qa_report.new_bugs:
        names = "、".join(f"{b.title}（{b.test_path}）"
                         for b in qa_report.new_bugs[:3])
        line += f"。新 bug：{names}" + ("…" if len(qa_report.new_bugs) > 3 else "")
    return line


def build_iter_narrative(*, analysis: Any, plan: Any, results: Any,
                         qa_report: Optional[Any], qa_fix_results: Any,
                         still_open_count: int,
                         reanalyzed: bool = False) -> dict[str, str]:
    title_by_id = {t.id: t.title for t in getattr(plan, "tasks", []) or []}
    return {
        "analyzer_summary": analyzer_line(analysis, reanalyzed=reanalyzed),
        "planner_summary": planner_line(plan),
        "executor_summary": executor_line(results, title_by_id),
        "qa_summary": qa_line(qa_report, qa_fix_results,
                              still_open=still_open_count),
    }
