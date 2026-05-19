"""Python adapter — extracts the existing ProjectHealth/syntax-check logic."""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from typing import List

from ..fs import Sandbox


class PythonAdapter:
    name = "python"
    test_corpus_dir = "tests/d2p_qa"

    TIMEOUT_S = 8

    def __init__(self, python: str | None = None) -> None:
        self.python = python or shutil.which("python3") or sys.executable

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
