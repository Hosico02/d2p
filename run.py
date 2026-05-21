"""CLI entry-point. Usage: python run.py <target_dir> [--iter N] [--parallel N]"""
from __future__ import annotations

import argparse
import logging
import sys

from d2p.config import Config
from d2p.orchestrator import Orchestrator


_VALID_RACE_ROLES = {"executor", "fix"}


def _parse_race_roles(spec: str) -> set[str]:
    """Parse the --race-mode CLI value.

    "" (no flag) → empty set (race off everywhere).
    "none"       → empty set.
    "all"        → every role we support racing on.
    "fix"        → {"fix"}.
    "fix,exec"   → {"fix", "executor"} (also accepts "executor").
    Unknown role names are dropped with a warning.
    """
    if not spec or spec.lower() == "none":
        return set()
    if spec.lower() == "all":
        return set(_VALID_RACE_ROLES)
    out: set[str] = set()
    for part in spec.split(","):
        name = part.strip().lower()
        # 'exec' is a tolerated alias for 'executor'
        if name == "exec":
            name = "executor"
        if name in _VALID_RACE_ROLES:
            out.add(name)
        elif name:
            logging.warning("ignored unknown race role %r (valid: %s)",
                            name, sorted(_VALID_RACE_ROLES))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="d2p — turn a demo into a product")
    p.add_argument("target", help="path to the demo project directory")
    p.add_argument("--iter", type=int, default=2, help="max iterations (default 2)")
    p.add_argument("--parallel", type=int, default=4, help="parallel executors")
    p.add_argument("--no-qa", action="store_true", help="disable QA stage")
    p.add_argument("--reanalyze-every", type=int, default=0,
                   help="re-run Analyzer every N iters (0=never, default)")
    p.add_argument("--qa-wontfix-after", type=int, default=3,
                   help="retire QA bugs after this many failed fix attempts "
                        "(default 3, 0=never retire)")
    p.add_argument("--max-concurrent-fixes", type=int, default=0,
                   help="cap fix tasks per iter (0=no cap). Lowest-attempt "
                        "bugs go first; the rest roll to next iter.")
    p.add_argument("--race-mode", nargs="?", const="all", default="",
                   metavar="ROLES",
                   help="enable race-mode for the listed roles (comma-separated). "
                        "When primary + fallback are both configured for a role, "
                        "their prepare() calls run in parallel; whichever side "
                        "commits first wins, slow side is abandoned. Costs 2× "
                        "LLM tokens per raced task. Accepts: 'all' (default if "
                        "flag is bare), 'fix', 'executor', 'fix,executor', "
                        "'none'. Without the flag, race is off everywhere. "
                        "Race forces max_fix_attempts=1 to avoid race × retry.")
    p.add_argument("--no-cache-analysis", action="store_true",
                   help="force a fresh Analyzer run, ignoring "
                        ".d2p/analysis_cache.json")
    p.add_argument("--resume", metavar="RUN_DIR",
                   help="resume a previous interrupted run. RUN_DIR is the "
                        "<target>/.d2p/run-* directory created by the prior "
                        "invocation. Rebuilds history from per-iter JSON "
                        "dumps and continues from the first incomplete iter.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config()
    cfg.reanalyze_every = args.reanalyze_every
    cfg.qa_wontfix_after_attempts = args.qa_wontfix_after
    cfg.max_concurrent_fixes = args.max_concurrent_fixes
    cfg.race_roles = _parse_race_roles(args.race_mode)
    orch = Orchestrator(args.target, cfg=cfg, max_iterations=args.iter,
                        parallel=args.parallel, enable_qa=not args.no_qa,
                        use_analyzer_cache=not args.no_cache_analysis,
                        resume_from=args.resume)
    orch.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
