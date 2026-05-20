"""Render a self-contained HTML run report from summary.json.

The report consolidates the per-iter markdown digests plus the cumulative
usage block into a single browseable page — useful for sharing run
artifacts without zipping the whole run_dir. No external CSS/JS deps;
everything is inline so the file works opened directly from disk.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1080px; margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a;
       line-height: 1.5; }
h1, h2, h3 { color: #0b0b0b; margin-top: 2rem; }
h1 { border-bottom: 2px solid #ddd; padding-bottom: 0.4rem; }
.metric { display: inline-block; margin: 0 1.5rem 0.5rem 0; }
.metric .label { font-size: 0.85rem; color: #666; }
.metric .value { font-size: 1.4rem; font-weight: 600; }
table { border-collapse: collapse; margin: 0.8rem 0; }
th, td { padding: 0.4rem 0.8rem; text-align: left; border-bottom: 1px solid #eee; }
th { background: #f7f7f7; font-weight: 600; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 3px;
         font-size: 0.85rem; font-weight: 500; }
.b-done { background: #d8f3dc; color: #1b5e20; }
.b-fail { background: #ffd6d6; color: #8b0000; }
.b-skip { background: #eee; color: #555; }
.b-warn { background: #fff3cd; color: #856404; }
.task { margin: 0.3rem 0; padding: 0.2rem 0; }
code { background: #f4f4f4; padding: 0.1rem 0.35rem; border-radius: 3px;
       font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.9em; }
.muted { color: #777; font-size: 0.9em; }
.rule { border: 0; border-top: 1px solid #eee; margin: 2rem 0; }
.stage-bar { background: #eee; border-radius: 4px; height: 0.7rem;
             margin: 0.3rem 0 0.7rem 0; overflow: hidden; display: flex; }
.stage-bar > div { height: 100%; }
.stage-planner   { background: #748ffc; }
.stage-executor  { background: #69db7c; }
.stage-qa        { background: #ffd43b; }
.stage-fix       { background: #ff8787; }
.stage-regression_sweep { background: #b197fc; }
.stage-other     { background: #ced4da; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _badge(status: str) -> str:
    cls = {
        "done": "b-done", "fixed": "b-done", "passed": "b-done",
        "failed": "b-fail", "error": "b-fail",
        "skipped": "b-skip", "wontfix": "b-warn", "open": "b-warn",
    }.get(status, "b-skip")
    return f'<span class="badge {cls}">{_esc(status)}</span>'


def _stage_bar(stage_timings: dict[str, float]) -> str:
    if not stage_timings:
        return ""
    total = sum(stage_timings.values()) or 1.0
    parts = []
    for name, sec in stage_timings.items():
        pct = 100.0 * sec / total
        # strip trailing "_s" if present
        short = name.removesuffix("_s")
        cls = f"stage-{short}" if short in {
            "planner", "executor", "qa", "fix", "regression_sweep"
        } else "stage-other"
        parts.append(
            f'<div class="{cls}" style="width:{pct:.1f}%" '
            f'title="{_esc(short)}: {sec:.1f}s ({pct:.1f}%)"></div>'
        )
    return f'<div class="stage-bar">{"".join(parts)}</div>'


def _render_iter(it: dict[str, Any], idx: int) -> str:
    stage_timings = it.get("stage_timings", {}) or {}
    elapsed = it.get("elapsed_s", 0)
    cost = it.get("cost_delta_usd", 0)
    cum_cost = it.get("cumulative_cost_usd", 0)
    parts = [
        f"<h2>Iteration {idx + 1}</h2>",
        f'<div class="muted">Elapsed: {elapsed:.1f}s  •  Cost: ${cost:.4f}  '
        f'•  Cumulative: ${cum_cost:.4f}</div>',
        _stage_bar(stage_timings),
    ]
    if stage_timings:
        bits = [f"{k.removesuffix('_s')}={v:.1f}s"
                for k, v in stage_timings.items()]
        parts.append(f'<div class="muted">Stage timings: {", ".join(bits)}</div>')

    rationale = it.get("plan_rationale") or ""
    if rationale:
        parts.append(f"<p><strong>Rationale.</strong> {_esc(rationale)}</p>")

    # Feature tasks
    results = it.get("results", []) or []
    if results:
        done = sum(1 for r in results if r.get("status") == "done")
        parts.append(f"<h3>Feature tasks ({done}/{len(results)})</h3>")
        parts.append("<table><thead><tr><th>Task</th><th>Status</th>"
                     "<th>Files</th></tr></thead><tbody>")
        for r in results:
            files = ", ".join(_esc(f) for f in (r.get("files_changed") or []))
            err = r.get("error") or ""
            err_block = (f'<br><span class="muted">{_esc(err[:200])}</span>'
                         if err and r.get("status") != "done" else "")
            parts.append(
                f"<tr><td><code>{_esc(r.get('task_id',''))}</code>"
                f"{err_block}</td>"
                f"<td>{_badge(r.get('status', '?'))}</td>"
                f"<td>{files or '<span class=muted>(none)</span>'}</td></tr>"
            )
        parts.append("</tbody></table>")

    # QA + fixes
    qa = it.get("qa") or {}
    qa_fixes = it.get("qa_fix_results", []) or []
    if qa or qa_fixes:
        parts.append("<h3>QA fixes</h3>")
        if qa_fixes:
            done = sum(1 for r in qa_fixes if r.get("status") == "done")
            parts.append(f'<div class="muted">{done}/{len(qa_fixes)} succeeded</div>')
            parts.append("<table><thead><tr><th>Task</th><th>Status</th>"
                         "<th>Files</th></tr></thead><tbody>")
            for r in qa_fixes:
                files = ", ".join(_esc(f) for f in (r.get("files_changed") or []))
                err = r.get("error") or ""
                err_block = (f'<br><span class="muted">{_esc(err[:200])}</span>'
                             if err and r.get("status") != "done" else "")
                parts.append(
                    f"<tr><td><code>{_esc(r.get('task_id',''))}</code>"
                    f"{err_block}</td>"
                    f"<td>{_badge(r.get('status', '?'))}</td>"
                    f"<td>{files or '<span class=muted>(none)</span>'}</td></tr>"
                )
            parts.append("</tbody></table>")
        if qa:
            parts.append("<h3>Bug flow</h3><ul>")
            parts.append(f"<li>Carried in: {len(qa.get('open_bugs', []))}</li>")
            parts.append(f"<li>New: {len(qa.get('new_bugs', []))}</li>")
            parts.append(f"<li>Incidentally fixed: {len(qa.get('fixed_bugs', []))}</li>")
            parts.append(f"<li>Retired: {len(it.get('retired_this_iter', []))}</li>")
            parts.append("</ul>")
            if qa.get("new_bugs"):
                parts.append("<h4>New bugs</h4><ul>")
                for b in qa["new_bugs"]:
                    parts.append(
                        f"<li><code>{_esc(b.get('test_path',''))}</code>: "
                        f"{_esc(b.get('title',''))}</li>"
                    )
                parts.append("</ul>")

    parts.append('<hr class="rule">')
    return "\n".join(parts)


def render_html(summary: dict[str, Any]) -> str:
    """Render the summary dict to a complete self-contained HTML document."""
    analysis = summary.get("analysis", {}) or {}
    iters = summary.get("iterations", []) or []
    usage = summary.get("usage", {}) or {}
    open_bugs = summary.get("open_bugs", []) or []
    counters = usage.get("counters", {}) or {}

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html><head>")
    parts.append('<meta charset="utf-8">')
    parts.append("<title>d2p run report</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")

    # Header
    parts.append("<h1>d2p run report</h1>")
    parts.append(
        '<div class="metric"><div class="label">Domain</div>'
        f'<div class="value">{_esc(analysis.get("domain", "?"))}</div></div>'
    )
    parts.append(
        '<div class="metric"><div class="label">Audience</div>'
        f'<div class="value">{_esc(analysis.get("audience", "?"))}</div></div>'
    )
    parts.append(
        '<div class="metric"><div class="label">Wall-clock</div>'
        f'<div class="value">{summary.get("elapsed_s", 0):.1f}s</div></div>'
    )
    parts.append(
        '<div class="metric"><div class="label">Cost</div>'
        f'<div class="value">${usage.get("total_cost_usd", 0):.4f}</div></div>'
    )
    parts.append(
        '<div class="metric"><div class="label">Calls</div>'
        f'<div class="value">{usage.get("total_calls", 0)}</div></div>'
    )
    parts.append(
        '<div class="metric"><div class="label">Cache hit</div>'
        f'<div class="value">{usage.get("cache_hit_ratio", 0)}</div></div>'
    )

    # Essence — the most important invariant
    if analysis.get("essence"):
        parts.append(
            f'<p><strong>Essence (immutable).</strong> '
            f'{_esc(analysis["essence"])}</p>'
        )

    parts.append('<hr class="rule">')

    # Iterations
    for i, it in enumerate(iters):
        parts.append(_render_iter(it, i))

    # Open bugs + usage table
    if open_bugs:
        parts.append(f"<h2>Open bugs ({len(open_bugs)})</h2><ul>")
        for b in open_bugs:
            attempts = b.get("attempts", 0)
            parts.append(
                f"<li><code>{_esc(b.get('test_path',''))}</code>: "
                f"{_esc(b.get('title',''))} "
                f'<span class="muted">(attempts={attempts}, '
                f'first_seen=iter{b.get("first_seen_iter", "?")})</span></li>'
            )
        parts.append("</ul>")

    parts.append("<h2>Usage breakdown</h2>")
    parts.append(
        "<table><thead><tr><th>Role</th><th>Calls</th><th>Input</th>"
        "<th>Output</th><th>Cache read</th><th>Cache creation</th>"
        "<th>Cost</th></tr></thead><tbody>"
    )
    for role_key, v in (usage.get("per_role", {}) or {}).items():
        parts.append(
            f"<tr><td><code>{_esc(role_key)}</code></td>"
            f"<td>{v.get('calls', 0)}</td>"
            f"<td>{v.get('input', 0):,}</td>"
            f"<td>{v.get('output', 0):,}</td>"
            f"<td>{v.get('cache_read', 0):,}</td>"
            f"<td>{v.get('cache_creation', 0):,}</td>"
            f"<td>${v.get('cost_usd', 0):.4f}</td></tr>"
        )
    parts.append("</tbody></table>")

    if counters:
        parts.append("<h3>Counters</h3><ul>")
        for k, v in counters.items():
            parts.append(f"<li><code>{_esc(k)}</code>: {v}</li>")
        parts.append("</ul>")

    parts.append('<p class="muted">Generated by d2p. '
                 f'Run dir: <code>{_esc(summary.get("run_dir", ""))}</code></p>')
    parts.append("</body></html>")
    return "\n".join(parts)


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(render_html(summary), encoding="utf-8")
