"""Pre-pulled execution evidence collector for the Verifier.

Lifted out of orchestrator.py so the calibration harness can call
this directly without the orchestrator's iteration loop. Verify spec
§7 defines the fixed evidence set; this module implements collection
against an arbitrary sandbox directory.

Verifier never invokes commands itself — this module does, and bundles
output into PreEvidence."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from d2p.agents.verifier import PreEvidence
from d2p.fs import Sandbox

DEFAULT_TIMEOUT_S = 90


def collect(sandbox: Sandbox, *, iter_count: int = 1,
            timeout_seconds: int = DEFAULT_TIMEOUT_S) -> PreEvidence:
    """Run tests / build / typecheck / git-diff against the sandbox root
    and bundle outputs into a PreEvidence record."""
    root = str(sandbox.root)
    listing = set(sandbox.listing(max_entries=400))
    is_python = ("pyproject.toml" in listing or "setup.py" in listing
                 or "requirements.txt" in listing
                 or any(p.endswith(".py") for p in listing))
    is_node = "package.json" in listing
    is_rust = "Cargo.toml" in listing
    is_go = "go.mod" in listing

    evidence = PreEvidence()

    # Tests
    test_cmd = None
    if is_python and shutil.which("pytest"):
        test_cmd = ["pytest", "-q", "--maxfail=20"]
    elif is_node:
        mgr = "pnpm" if "pnpm-lock.yaml" in listing else "npm"
        if shutil.which(mgr):
            test_cmd = [mgr, "test", "--silent"] if mgr == "npm" else [mgr, "test"]
    elif is_rust and shutil.which("cargo"):
        test_cmd = ["cargo", "test", "--quiet"]
    elif is_go and shutil.which("go"):
        test_cmd = ["go", "test", "./..."]
    if test_cmd:
        evidence.test_output, evidence.test_exit_code = _run_cmd(
            test_cmd, cwd=root, timeout_seconds=timeout_seconds)

    # Build
    build_cmd = None
    if is_node and shutil.which("pnpm" if "pnpm-lock.yaml" in listing else "npm"):
        mgr = "pnpm" if "pnpm-lock.yaml" in listing else "npm"
        pkg_txt = sandbox.read("package.json")
        if pkg_txt and '"build"' in pkg_txt:
            build_cmd = [mgr, "run", "build"]
    elif is_rust and shutil.which("cargo"):
        build_cmd = ["cargo", "build", "--quiet"]
    elif is_go and shutil.which("go"):
        build_cmd = ["go", "build", "./..."]
    if build_cmd:
        evidence.build_output, evidence.build_exit_code = _run_cmd(
            build_cmd, cwd=root, timeout_seconds=timeout_seconds)

    # Typecheck
    typecheck_cmd = None
    if is_python and shutil.which("mypy"):
        typecheck_cmd = ["mypy", "--ignore-missing-imports",
                         "--no-error-summary", "."]
    elif is_node and "tsconfig.json" in listing:
        mgr = "pnpm" if "pnpm-lock.yaml" in listing else "npx"
        if shutil.which(mgr):
            typecheck_cmd = [mgr, "tsc", "--noEmit"]
    if typecheck_cmd:
        evidence.typecheck_output, evidence.typecheck_exit_code = _run_cmd(
            typecheck_cmd, cwd=root, timeout_seconds=timeout_seconds)

    # Git diff
    if (Path(root) / ".git").is_dir() and shutil.which("git"):
        n = max(1, iter_count)
        out, code = _run_cmd(
            ["git", "diff", f"HEAD~{n}..HEAD"], cwd=root,
            timeout_seconds=timeout_seconds)
        if code != 0:
            out, _ = _run_cmd(["git", "diff", "HEAD"], cwd=root,
                              timeout_seconds=timeout_seconds)
        evidence.git_diff_recent = out

    return evidence


def _run_cmd(cmd: list[str], *, cwd: str,
             timeout_seconds: int) -> tuple[str, int]:
    """Run a command with a hard timeout. Returns (combined_output, exit_code).
    Exit code 124 = timeout (timeout(1) convention); 127 = command not found.
    Never raises — the verifier needs to see whatever happened."""
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout_seconds,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return ((p.stdout + ("\n" + p.stderr if p.stderr else "")).rstrip(),
                int(p.returncode))
    except subprocess.TimeoutExpired as e:
        return (f"<timed out after {timeout_seconds}s: "
                f"{' '.join(cmd)}>\n{(e.stdout or b'').decode(errors='replace')}",
                124)
    except FileNotFoundError:
        return (f"<command not found: {cmd[0]}>", 127)
    except Exception as e:
        return (f"<unexpected error running {cmd!r}: "
                f"{type(e).__name__}: {e}>", 1)
