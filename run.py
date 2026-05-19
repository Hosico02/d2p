"""CLI entry-point. Usage: python run.py <target_dir> [--iter N] [--parallel N]"""
from __future__ import annotations

import argparse
import logging
import sys

from d2p.config import Config
from d2p.orchestrator import Orchestrator


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
    p.add_argument("--no-cache-analysis", action="store_true",
                   help="force a fresh Analyzer run, ignoring "
                        ".d2p/analysis_cache.json")
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
    orch = Orchestrator(args.target, cfg=cfg, max_iterations=args.iter,
                        parallel=args.parallel, enable_qa=not args.no_qa,
                        use_analyzer_cache=not args.no_cache_analysis)
    orch.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
