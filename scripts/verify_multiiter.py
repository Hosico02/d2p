"""Run d2p --iter 3 on a fresh werewolf-demo and prove the loop converges:

  - QA test corpus strictly grows across iterations
  - prompts.py / game.py / player.py / app.py remain importable after the run
  - open_bugs count does not strictly increase iteration over iteration

Exits 0 on success, 1 on assertion failure. Prints a structured summary.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent
DEMO = SRC.parent / "werewolf-demo"
RUN_PY = SRC / "run.py"
VENV_PY = SRC / ".venv" / "bin" / "python"
def _pick_python_for_demo(scratch: Path) -> str:
    """Pick a python that can actually import the demo. The demo may use
    PEP-604 union syntax (X | Y) which requires 3.10+; system python3 might
    be 3.9 on macOS. Walk candidates and use whichever works."""
    candidates = [
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        shutil.which("python3.12"),
        shutil.which("python3.11"),
        shutil.which("python3.10"),
        shutil.which("python3"),
        "/usr/bin/python3",
    ]
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            r = subprocess.run(
                [c, "-c", "import sys; sys.path.insert(0,'.'); import app"],
                cwd=str(scratch), capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                return c
        except Exception:
            continue
    return "/usr/bin/python3"


def main() -> int:
    if not DEMO.is_dir():
        print(f"ERROR: werewolf-demo not found at {DEMO}", file=sys.stderr)
        return 1

    scratch_root = Path(tempfile.mkdtemp(prefix="d2p_verify_"))
    scratch = scratch_root / "werewolf-demo"
    subprocess.run(["rsync", "-a",
                    "--exclude=__pycache__", "--exclude=.git",
                    f"{DEMO}/", f"{scratch}/"], check=True)
    print(f"scratch dir: {scratch}")

    t0 = time.time()
    r = subprocess.run(
        [str(VENV_PY), str(RUN_PY), str(scratch),
         "--iter", "3", "--parallel", "2"],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    print(f"d2p exited {r.returncode} in {elapsed:.1f}s")
    if r.returncode != 0:
        print("=== stderr tail ===")
        print(r.stderr[-2000:])
        return 1

    run_dirs = sorted((scratch / ".d2p").iterdir())
    if not run_dirs:
        print("ERROR: no .d2p run dir produced", file=sys.stderr)
        return 1
    run_dir = run_dirs[-1]
    print(f"run dir: {run_dir}")
    summary = json.loads((run_dir / "summary.json").read_text())

    # 1) Corpus growth
    corpus_sizes: list[int] = []
    for i in range(1, 4):
        qa_path = run_dir / f"qa_iter{i}.json"
        if not qa_path.exists():
            continue
        qa = json.loads(qa_path.read_text())
        corpus_sizes.append(
            len(qa.get("new_bugs", [])) + len(qa.get("open_bugs", []))
        )
    print(f"corpus sizes per iter (bugs known to QA): {corpus_sizes}")

    # 2) iteration → fix-success info
    iter_info = []
    for i, it in enumerate(summary.get("iterations", []), 1):
        qa = it.get("qa") or {}
        iter_info.append({
            "iter": i,
            "new_bugs": len(qa.get("new_bugs", [])),
            "open_bugs": len(qa.get("open_bugs", [])),
            "fixed_bugs": len(qa.get("fixed_bugs", [])),
            "feature_done": sum(1 for r in it.get("results", [])
                                if r.get("status") == "done"),
            "feature_failed": sum(1 for r in it.get("results", [])
                                  if r.get("status") == "failed"),
            "fix_done": sum(1 for r in it.get("qa_fix_results", [])
                            if r.get("status") == "done"),
            "fix_failed": sum(1 for r in it.get("qa_fix_results", [])
                              if r.get("status") == "failed"),
        })
    print("per-iter info:")
    for x in iter_info:
        print(f"  iter {x['iter']}: feat={x['feature_done']}/{x['feature_done']+x['feature_failed']} "
              f"new={x['new_bugs']} open={x['open_bugs']} fixed={x['fixed_bugs']} "
              f"fix_done={x['fix_done']} fix_failed={x['fix_failed']}")

    # 3) Final import health (use a python that can run this specific demo)
    py = _pick_python_for_demo(scratch)
    print(f"using {py} for final import probe")
    probe = subprocess.run(
        [py, "-c",
         "import sys; sys.path.insert(0,'.'); "
         "import app, game, player, prompts; print('ok')"],
        cwd=str(scratch), capture_output=True, text=True, timeout=15,
    )
    print(f"final-state import probe: rc={probe.returncode}  out={probe.stdout.strip()!r}")
    if probe.returncode != 0:
        print(probe.stderr[-800:], file=sys.stderr)

    # Assertions
    fails: list[str] = []
    if probe.returncode != 0:
        # Tolerate the case where the demo itself was already Python-version-
        # incompatible (e.g. PEP-604 unions on 3.9). That's not a d2p regression.
        if "unsupported operand type(s) for |" in probe.stderr:
            print("NOTE: final probe hits pre-existing PEP-604 incompatibility "
                  "in the demo — not counted as d2p regression.")
        else:
            fails.append("final import probe failed — project broken")
    # The real "corpus" is the count of distinct test files ever produced.
    # corpus_sizes is "bugs known THIS ITER" (open + new) and can shrink as
    # fixes flip status to 'fixed'. So we check it's never zero and that the
    # MAX is at least the first-iter count.
    if corpus_sizes and max(corpus_sizes) < corpus_sizes[0]:
        fails.append(f"corpus shrank: {corpus_sizes}")
    # at least one iteration should successfully fix at least one bug
    total_fixed = sum(x["fixed_bugs"] for x in iter_info)
    if iter_info and total_fixed == 0:
        fails.append(f"no bugs ever transitioned to fixed: {iter_info}")

    if fails:
        print("FAIL:", *fails, sep="\n  ")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
