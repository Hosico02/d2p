"""Python adapter — extracts the existing ProjectHealth/syntax-check logic."""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from typing import List

from ..fs import Sandbox


def _pick_python_310plus() -> str | None:
    """Find the newest Python (>=3.10) on this machine.

    Demos increasingly use PEP-604 unions (`int | None`) which 3.9 can't
    even import. If we fall back to /usr/bin/python3 (3.9 on most macOS),
    our health probe never sees a healthy baseline and rollback never fires.
    Prefer 3.10+ explicitly, scanning common install locations.
    """
    candidates: list[str] = []
    # CONDA envs
    home = os.path.expanduser("~")
    for env_dir in (
        os.path.join(home, "anaconda3", "envs"),
        os.path.join(home, "miniconda3", "envs"),
        os.path.join(home, "miniforge3", "envs"),
    ):
        if os.path.isdir(env_dir):
            for name in os.listdir(env_dir):
                p = os.path.join(env_dir, name, "bin", "python")
                if os.path.isfile(p):
                    candidates.append(p)
    # Homebrew + system pythonX.Y. shutil.which() returns Optional[str], so
    # narrow before appending.
    homebrew_candidates: tuple[str | None, ...] = (
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.11",
        shutil.which("python3.13"),
        shutil.which("python3.12"),
        shutil.which("python3.11"),
        shutil.which("python3.10"),
    )
    for cand in homebrew_candidates:
        if cand is None:
            continue
        if os.path.isfile(cand):
            candidates.append(cand)
    # version probe (>=3.10)
    for c in dict.fromkeys(candidates):
        try:
            r = subprocess.run(
                [c, "-c",
                 "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"],
                capture_output=True, timeout=3,
            )
            if r.returncode == 0:
                return c
        except Exception:
            continue
    return None


class PythonAdapter:
    name = "python"
    test_corpus_dir = "tests/d2p_qa"

    TIMEOUT_S = 8

    def __init__(self, python: str | None = None) -> None:
        self.python = python or _pick_python_310plus() or shutil.which("python3") or sys.executable

    # ---- health ----

    def discover_modules(self, sandbox: Sandbox) -> List[str]:
        out: list[str] = []
        listing = sorted(sandbox.listing(max_entries=400))
        # top-level *.py
        for p in listing:
            if not p.endswith(".py") or "/" in p:
                continue
            stem = p[:-3]
            if stem in {"setup", "conftest"}:
                continue
            out.append(stem)
        # the demo's OWN tests under tests/*.py — catches regressions in
        # pre-existing tests that the demo author shipped
        for p in listing:
            if not p.endswith(".py"):
                continue
            if not p.startswith("tests/") or p.startswith(self.test_corpus_dir + "/"):
                continue
            if p.endswith("__init__.py") or p.endswith("conftest.py"):
                continue
            if p.count("/") > 1:
                continue  # only top-level tests/*.py for now
            out.append(p[:-3].replace("/", "."))
        # QA corpus tests
        for p in listing:
            if not p.startswith(self.test_corpus_dir + "/") or not p.endswith(".py"):
                continue
            if p.endswith("__init__.py") or p.endswith("conftest.py"):
                continue
            out.append(p[:-3].replace("/", "."))
        return out

    def import_probe(self, sandbox: Sandbox, modules: List[str]) -> dict:
        if not modules:
            return {}
        script = (
            "import json\n"
            "out = {}\n"
            f"for m in {modules!r}:\n"
            "    try:\n"
            "        __import__(m)\n"
            "        out[m] = 'ok'\n"
            "    except BaseException as e:\n"
            "        out[m] = f'{type(e).__name__}: {e}'\n"
            "print(json.dumps(out))\n"
        )
        try:
            r = subprocess.run(
                [self.python, "-c", script],
                cwd=str(sandbox.root),
                capture_output=True, text=True, timeout=self.TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return {m: "timeout" for m in modules}
        if r.returncode != 0 or not r.stdout.strip():
            return {m: f"probe-crashed: {r.stderr[:200]}" for m in modules}
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            return {m: f"probe-output-unparseable: {r.stdout[:200]}" for m in modules}

    # ---- write-time safety ----

    def syntax_check(self, sandbox: Sandbox, rel_path: str) -> str:
        if not rel_path.endswith(".py"):
            return ""
        try:
            ast.parse(sandbox.read(rel_path))
        except SyntaxError as e:
            return f"syntax error: {e.msg} (line {e.lineno})"
        return ""

    # ---- QA ----

    def test_template(self) -> str:
        return (
            '"""<one-line bug hypothesis>"""\n'
            "import os, sys, unittest\n"
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))\n"
            "\n"
            "# project imports — adapt to the project\n"
            "from app import app  # if Flask/FastAPI\n"
            "\n"
            "\n"
            "class TestSomething(unittest.TestCase):\n"
            "    def setUp(self):\n"
            "        pass\n"
            "\n"
            "    def test_descriptive_name(self):\n"
            "        # exercise the suspected-buggy path; assertion fails if bug exists\n"
            "        self.assertEqual(...)\n"
            "\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    unittest.main()\n"
        )

    def test_path(self, slug: str) -> str:
        if not slug.endswith(".py"):
            slug = slug + ".py"
        return f"{self.test_corpus_dir}/{slug}"

    def test_runner_cmd(self, rel_path: str, *,
                        sandbox: Sandbox | None = None) -> List[str]:
        """Pick pytest vs unittest based on the test file's actual imports.

        If the file imports pytest (or uses pytest-style bare `def test_*`
        functions instead of unittest.TestCase), use `python -m pytest`.
        Otherwise the default `python -m unittest`.
        """
        use_pytest = False
        if sandbox is not None:
            content = sandbox.read(rel_path) or ""
            if ("import pytest" in content
                    or "from pytest" in content
                    or ("def test_" in content
                        and "unittest.TestCase" not in content)):
                use_pytest = True
        if use_pytest:
            return [self.python, "-m", "pytest", rel_path, "-q", "--no-header"]
        module = rel_path.replace("/", ".").rsplit(".py", 1)[0]
        return [self.python, "-m", "unittest", module, "-v"]
