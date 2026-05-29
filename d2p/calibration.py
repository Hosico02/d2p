"""Calibration harness for the d2p Verifier.

See docs/superpowers/specs/2026-05-26-d2p-calibration-harness-design.md
in the demo2project repo for the design rationale.

Pure-logic functions (classify_outcome, compute_metrics, match_categories,
load_baseline_meta) are testable without LLM calls. run_baseline() and
main() integrate with the live Verifier."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from d2p.agents.pre_evidence import collect as collect_pre_evidence
from d2p.agents.verifier import VerifyClaim, PreEvidence
from d2p.fs import Sandbox


# Outcome enum values
OUTCOME_CATCH = "catch"
OUTCOME_MISS = "miss"
OUTCOME_CLEAN_PASS = "clean_pass"
OUTCOME_FALSE_ALARM = "false_alarm"
OUTCOME_ERROR = "error"

CATCH_THRESHOLD = 0.8
FP_THRESHOLD = 0.2

# Stable, committed snapshot of the most recent calibration. Read at run
# time by the orchestrator to surface verifier confidence to summary.json
# and the Hub. Path is relative to the Forge repo root (parent of the d2p
# package) so it resolves regardless of the target demo's cwd.
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "docs" / "calibration" / "latest.json"


@dataclass
class Metrics:
    catch_rate: float            # catch / (catch + miss); 0.0 if denominator 0
    fp_rate: float               # false_alarm / (clean_pass + false_alarm); 0.0 if denominator 0
    pass_on_broken: int          # # of broken baselines with verdict == "pass"
    criteria_met: bool           # catch >= 0.8 AND fp <= 0.2 AND pass_on_broken == 0
    total_baselines: int
    errors: int                  # # of rows with outcome == "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_outcome(*, kind: str, verdict: str,
                     expected_verdict_in: list[str]) -> str:
    """Determine outcome bucket for a baseline row.
    Returns one of OUTCOME_* constants. Unknown kind -> OUTCOME_ERROR
    so callers see misconfigured expected.json files clearly."""
    in_expected = verdict in expected_verdict_in
    if kind == "broken":
        return OUTCOME_CATCH if in_expected else OUTCOME_MISS
    if kind == "productized":
        return OUTCOME_CLEAN_PASS if in_expected else OUTCOME_FALSE_ALARM
    return OUTCOME_ERROR


def match_categories(*, actual: list[str],
                     expected_substrings: list[str]) -> Optional[bool]:
    """Soft substring match: any actual category contains any expected
    substring (case-insensitive). Empty expected_substrings -> None
    (caller treats as "skip", not as failure)."""
    if not expected_substrings:
        return None
    actual_lc = [c.lower() for c in actual]
    expected_lc = [s.lower() for s in expected_substrings]
    return any(any(sub in cat for sub in expected_lc) for cat in actual_lc)


def compute_metrics(rows: list[dict]) -> Metrics:
    """Aggregate outcome rows into Metrics. Rows with outcome OUTCOME_ERROR
    are counted in total/errors but excluded from rate denominators."""
    catch = sum(1 for r in rows if r["outcome"] == OUTCOME_CATCH)
    miss = sum(1 for r in rows if r["outcome"] == OUTCOME_MISS)
    clean = sum(1 for r in rows if r["outcome"] == OUTCOME_CLEAN_PASS)
    false_alarm = sum(1 for r in rows if r["outcome"] == OUTCOME_FALSE_ALARM)
    errors = sum(1 for r in rows if r["outcome"] == OUTCOME_ERROR)

    broken_denom = catch + miss
    prod_denom = clean + false_alarm
    catch_rate = catch / broken_denom if broken_denom else 0.0
    fp_rate = false_alarm / prod_denom if prod_denom else 0.0

    pass_on_broken = sum(
        1 for r in rows
        if r["kind"] == "broken" and r.get("actual_verdict") == "pass"
    )

    criteria_met = (catch_rate >= CATCH_THRESHOLD
                    and fp_rate <= FP_THRESHOLD
                    and pass_on_broken == 0)

    return Metrics(
        catch_rate=catch_rate, fp_rate=fp_rate,
        pass_on_broken=pass_on_broken, criteria_met=criteria_met,
        total_baselines=len(rows), errors=errors,
    )


_REQUIRED_META_FIELDS = (
    "name", "kind", "archetype",
    "expected_verdict_in", "expected_categories_any_of", "notes",
)


def load_baseline_meta(baseline_dir: Path) -> dict:
    """Read and validate expected.json. Raises FileNotFoundError if absent,
    ValueError if required fields missing."""
    f = baseline_dir / "expected.json"
    if not f.is_file():
        raise FileNotFoundError(f"missing expected.json in {baseline_dir}")
    meta = json.loads(f.read_text())
    missing = [k for k in _REQUIRED_META_FIELDS if k not in meta]
    if missing:
        raise ValueError(f"{f}: required fields missing: {missing}")
    if meta["kind"] not in ("broken", "productized"):
        raise ValueError(f"{f}: kind must be broken|productized, got {meta['kind']!r}")
    return meta


def run_baseline(baseline_dir: Path, verifier_factory, *,
                 dry_run: bool = False,
                 skip_pre_evidence: bool = False) -> dict:
    """Execute one baseline: load meta, collect pre-evidence, build a
    fresh Verifier via `verifier_factory(baseline_dir)`, call verify,
    classify outcome, build row dict. Catches verify exceptions and
    records them as OUTCOME_ERROR rows instead of propagating.

    `verifier_factory` is a callable `(Path) -> Verifier`. The
    Verifier needs a Sandbox rooted at `baseline_dir`, so we cannot
    reuse a single Verifier across baselines — but `verifier_factory`
    can close over a shared router so provider construction is paid
    once."""
    started = time.monotonic()
    try:
        meta = load_baseline_meta(baseline_dir)
    except (FileNotFoundError, ValueError) as e:
        return {
            "name": baseline_dir.name, "kind": "unknown",
            "expected_verdict_in": [], "expected_categories_any_of": [],
            "actual_verdict": "", "actual_categories": [],
            "verdict_match": False, "category_match": None,
            "outcome": OUTCOME_ERROR, "elapsed_seconds": 0.0,
            "verify_result": {}, "error": f"{type(e).__name__}: {e}",
        }

    sandbox = Sandbox(baseline_dir)
    if skip_pre_evidence:
        pre_evidence = PreEvidence()
    else:
        pre_evidence = collect_pre_evidence(sandbox, iter_count=1)

    if dry_run:
        return {
            "name": meta["name"], "kind": meta["kind"],
            "expected_verdict_in": meta["expected_verdict_in"],
            "expected_categories_any_of": meta["expected_categories_any_of"],
            "actual_verdict": "<dry-run>", "actual_categories": [],
            "verdict_match": False, "category_match": None,
            "outcome": OUTCOME_ERROR,
            "elapsed_seconds": time.monotonic() - started,
            "verify_result": {"pre_evidence": pre_evidence.to_dict()},
        }

    claim = VerifyClaim(iter_count=3, no_more_features=True,
                        no_more_bugs=True, qa_corpus_green=True)
    try:
        verifier = verifier_factory(baseline_dir)
        result = verifier.verify(claim, pre_evidence)
    except Exception as e:
        return {
            "name": meta["name"], "kind": meta["kind"],
            "expected_verdict_in": meta["expected_verdict_in"],
            "expected_categories_any_of": meta["expected_categories_any_of"],
            "actual_verdict": "", "actual_categories": [],
            "verdict_match": False, "category_match": None,
            "outcome": OUTCOME_ERROR,
            "elapsed_seconds": time.monotonic() - started,
            "verify_result": {}, "error": f"{type(e).__name__}: {e}",
        }

    actual_cats = [f.category for f in result.new_finding_categories]
    verdict_match = result.verdict in meta["expected_verdict_in"]
    cat_match = match_categories(
        actual=actual_cats,
        expected_substrings=meta["expected_categories_any_of"])
    outcome = classify_outcome(
        kind=meta["kind"], verdict=result.verdict,
        expected_verdict_in=meta["expected_verdict_in"])

    return {
        "name": meta["name"], "kind": meta["kind"],
        "expected_verdict_in": meta["expected_verdict_in"],
        "expected_categories_any_of": meta["expected_categories_any_of"],
        "actual_verdict": result.verdict,
        "actual_categories": actual_cats,
        "verdict_match": verdict_match,
        "category_match": cat_match,
        "outcome": outcome,
        "elapsed_seconds": time.monotonic() - started,
        "verify_result": result.to_dict(),
    }


def write_report(out_dir: Path, rows: list[dict], metrics: Metrics,
                 meta: dict) -> None:
    """Write report.json + report.md to out_dir. Creates out_dir if missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = {
        "harness_version": meta.get("harness_version", "v0"),
        "started_at": meta.get("started_at", ""),
        "elapsed_seconds": meta.get("elapsed_seconds", 0.0),
        "model": meta.get("model", ""),
        "metrics": metrics.to_dict(),
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(full, indent=2,
                                                    default=str))
    (out_dir / "report.md").write_text(_render_md(rows, metrics, meta))


def snapshot_dict(metrics: Metrics, meta: dict) -> dict[str, Any]:
    """The compact 'verifier confidence' record persisted as the canonical
    calibration snapshot and forwarded to the Hub. Flat + JSON-primitive so
    it survives the HTTP contract and a SQLite row without translation."""
    return {
        "catch_rate": metrics.catch_rate,
        "fp_rate": metrics.fp_rate,
        "pass_on_broken": metrics.pass_on_broken,
        "criteria_met": metrics.criteria_met,
        "total_baselines": metrics.total_baselines,
        "errors": metrics.errors,
        "model": meta.get("model", ""),
        "harness_version": meta.get("harness_version", "v0"),
        "calibrated_at": meta.get("started_at", ""),
    }


def write_snapshot(metrics: Metrics, meta: dict,
                   path: Path = SNAPSHOT_PATH) -> None:
    """Overwrite the canonical calibration snapshot. Called by main() after a
    real (non-dry-run) calibration so the committed confidence travels with
    the code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot_dict(metrics, meta), indent=2) + "\n")


def load_snapshot(path: Path = SNAPSHOT_PATH) -> Optional[dict[str, Any]]:
    """Read the canonical calibration snapshot for run-time reporting.
    Fail-safe: returns None if the file is missing or unreadable so a run
    never depends on calibration having been run."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _render_md(rows: list[dict], metrics: Metrics, meta: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Calibration report · {meta.get('started_at', '')}")
    lines.append("")
    lines.append(f"**Model:** {meta.get('model', '')}")
    lines.append(f"**Total baselines:** {metrics.total_baselines}"
                 f" (errors: {metrics.errors})")
    lines.append(f"**Elapsed:** {meta.get('elapsed_seconds', 0.0):.1f}s")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value | §11 target | Met? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| catch_rate | {metrics.catch_rate:.2f} | >= {CATCH_THRESHOLD:.2f} | "
                 f"{'YES' if metrics.catch_rate >= CATCH_THRESHOLD else 'NO'} |")
    lines.append(f"| fp_rate | {metrics.fp_rate:.2f} | <= {FP_THRESHOLD:.2f} | "
                 f"{'YES' if metrics.fp_rate <= FP_THRESHOLD else 'NO'} |")
    lines.append(f"| pass_on_broken | {metrics.pass_on_broken} | == 0 | "
                 f"{'YES' if metrics.pass_on_broken == 0 else 'NO'} |")
    lines.append("")
    lines.append(f"**Single-pass criteria met:** "
                 f"{'YES' if metrics.criteria_met else 'NO'}")
    lines.append("")
    lines.append("## Per-baseline detail")
    lines.append("")
    for r in rows:
        icon = {OUTCOME_CATCH: "[catch]", OUTCOME_CLEAN_PASS: "[clean]",
                OUTCOME_MISS: "[MISS]", OUTCOME_FALSE_ALARM: "[FP]",
                OUTCOME_ERROR: "[err]"}.get(r["outcome"], "?")
        lines.append(f"### {r['kind']} / {r['name']} — {r['outcome']} {icon}")
        lines.append(f"- verdict: `{r['actual_verdict']}` "
                     f"(expected in {r['expected_verdict_in']})")
        lines.append(f"- categories: {r.get('actual_categories', []) or 'none'}")
        if r.get("category_match") is True:
            lines.append("- category match: YES")
        elif r.get("category_match") is False:
            lines.append("- category match: NO")
        lines.append(f"- elapsed: {r.get('elapsed_seconds', 0):.1f}s")
        if r.get("error"):
            lines.append(f"- error: `{r['error']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _exit_code_for(metrics: Metrics) -> int:
    if metrics.errors > 0:
        return 2
    return 0 if metrics.criteria_met else 1


def _build_verifier_factory(model: str):
    """Return a callable `(Path) -> Verifier` that constructs a fresh
    Verifier rooted at the given baseline directory. The provider is
    built once (closed over) and reused; only the Sandbox changes.

    Reads MINIMAX_API_KEY from env. v0 only supports kind=minimax."""
    import os
    from d2p.providers import ProviderSpec, build_router
    from d2p.agents.verifier import Verifier
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY env var not set")
    spec = ProviderSpec(
        kind="minimax", api_key=api_key, default_model=model,
        role_models={"default": model, "verify": model},
    )

    def factory(project_path: Path):
        router = build_router(spec=spec, working_dir=str(project_path))
        return Verifier(router.for_role("verify"), Sandbox(project_path))

    return factory


def _iter_baselines(baselines_dir: Path,
                    kinds: list[str],
                    name_filter: Optional[str]) -> list[Path]:
    out: list[Path] = []
    for kind in kinds:
        kind_dir = baselines_dir / kind
        if not kind_dir.is_dir():
            continue
        for child in sorted(kind_dir.iterdir()):
            if not child.is_dir():
                continue
            if name_filter and name_filter not in child.name:
                continue
            out.append(child)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import datetime as _dt
    parser = argparse.ArgumentParser(
        description="d2p Verifier calibration harness")
    parser.add_argument("--baselines", required=True, type=Path,
                        help="root containing broken/ and productized/ subdirs")
    parser.add_argument("--kind", default="broken,productized",
                        help="comma-separated subset of kinds to run")
    parser.add_argument("--model", default="minimax-m2.7-hs")
    parser.add_argument("--out", required=True, type=Path,
                        help="output directory for report.json + report.md")
    parser.add_argument("--filter", default=None,
                        help="run only baselines whose name contains this")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip Verifier.verify() calls (no LLM)")
    parser.add_argument("--skip-pre-evidence", action="store_true",
                        help="pass empty PreEvidence to verify")
    args = parser.parse_args(argv)

    kinds = [k.strip() for k in args.kind.split(",") if k.strip()]
    baselines = _iter_baselines(args.baselines, kinds, args.filter)
    if not baselines:
        print(f"No baselines found under {args.baselines} for kinds={kinds}",
              file=sys.stderr)
        return 2

    started = time.monotonic()
    started_iso = _dt.datetime.now(_dt.timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")

    rows: list[dict] = []
    verifier_factory = None
    if not args.dry_run:
        try:
            verifier_factory = _build_verifier_factory(args.model)
        except Exception as e:
            print(f"Could not construct verifier factory: {e}", file=sys.stderr)
            return 2

    for b in baselines:
        print(f"[calibrate] {b.parent.name}/{b.name} ...", flush=True)
        rows.append(run_baseline(
            b, verifier_factory,
            dry_run=args.dry_run,
            skip_pre_evidence=args.skip_pre_evidence,
        ))

    metrics = compute_metrics(rows)
    meta = {
        "harness_version": "v0",
        "started_at": started_iso,
        "elapsed_seconds": time.monotonic() - started,
        "model": args.model,
    }
    write_report(args.out, rows, metrics, meta)
    print(f"[calibrate] report -> {args.out}")
    if not args.dry_run:
        write_snapshot(metrics, meta)
        print(f"[calibrate] snapshot -> {SNAPSHOT_PATH}")
    print(f"[calibrate] catch_rate={metrics.catch_rate:.2f} "
          f"fp_rate={metrics.fp_rate:.2f} "
          f"pass_on_broken={metrics.pass_on_broken} "
          f"criteria_met={metrics.criteria_met}")
    return _exit_code_for(metrics)


if __name__ == "__main__":
    sys.exit(main())
