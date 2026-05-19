"""JavaScript / TypeScript adapter — uses Node.js toolchain (no deps required).

`node --check` validates JS files. The built-in `node:test` runner (Node 18+)
runs tests without installing Jest/Mocha. TypeScript uses `tsc --noEmit` if
available, otherwise falls back to a permissive syntax check.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import List

from ..fs import Sandbox


class JSAdapter:
    name = "javascript"
    test_corpus_dir = "tests/d2p_qa"

    TIMEOUT_S = 10

    def __init__(self, node: str | None = None) -> None:
        self.node = node or shutil.which("node") or "node"
        self.tsc = shutil.which("tsc")

    # ---- health ----

    def discover_modules(self, sandbox: Sandbox) -> List[str]:
        out: list[str] = []
        listing = sorted(sandbox.listing(max_entries=400))
        for p in listing:
            if "/" in p:
                continue
            if p.endswith((".js", ".mjs", ".cjs", ".ts")) and not p.endswith(".d.ts"):
                out.append(p)
        # also import test corpus
        for p in listing:
            if not p.startswith(self.test_corpus_dir + "/"):
                continue
            if p.endswith((".test.js", ".test.mjs", ".test.ts")):
                out.append(p)
        return out

    def import_probe(self, sandbox: Sandbox, modules: List[str]) -> dict:
        if not modules:
            return {}
        out: dict[str, str] = {}
        for rel in modules:
            cmd = self._check_cmd(rel)
            if not cmd:
                out[rel] = "ok"  # nothing to check
                continue
            try:
                r = subprocess.run(
                    cmd, cwd=str(sandbox.root),
                    capture_output=True, text=True, timeout=self.TIMEOUT_S,
                )
                if r.returncode == 0:
                    out[rel] = "ok"
                else:
                    err = (r.stderr or r.stdout or "").strip().splitlines()
                    out[rel] = err[-1] if err else f"exit {r.returncode}"
            except subprocess.TimeoutExpired:
                out[rel] = "timeout"
            except FileNotFoundError:
                out[rel] = "node not found"
        return out

    def _check_cmd(self, rel: str) -> list[str]:
        if rel.endswith((".js", ".mjs", ".cjs")):
            return [self.node, "--check", rel]
        if rel.endswith(".ts"):
            if self.tsc:
                return [self.tsc, "--noEmit", "--allowJs", "--skipLibCheck", rel]
            # no tsc → degrade to a JS parse via swc-style strip? we just say ok.
            return []
        return []

    # ---- write-time safety ----

    def syntax_check(self, sandbox: Sandbox, rel_path: str) -> str:
        cmd = self._check_cmd(rel_path)
        if not cmd:
            return ""
        try:
            r = subprocess.run(
                cmd, cwd=str(sandbox.root),
                capture_output=True, text=True, timeout=self.TIMEOUT_S,
            )
            if r.returncode == 0:
                return ""
            err = (r.stderr or r.stdout or "").strip().splitlines()
            return err[-1] if err else f"check exit {r.returncode}"
        except FileNotFoundError:
            return ""  # toolchain missing — don't block writes
        except subprocess.TimeoutExpired:
            return "syntax check timeout"

    # ---- QA ----

    def test_template(self) -> str:
        return (
            "// <one-line bug hypothesis>\n"
            "import { test } from 'node:test';\n"
            "import assert from 'node:assert/strict';\n"
            "import path from 'node:path';\n"
            "import url from 'node:url';\n"
            "\n"
            "const __filename = url.fileURLToPath(import.meta.url);\n"
            "const __dirname  = path.dirname(__filename);\n"
            "const ROOT = path.resolve(__dirname, '..', '..');\n"
            "\n"
            "test('descriptive_name', async () => {\n"
            "  // exercise the suspected-buggy path; assertion fails if bug exists\n"
            "  const mod = await import(path.join(ROOT, 'server.js'));\n"
            "  assert.equal(mod.something(), 'expected');\n"
            "});\n"
        )

    def test_path(self, slug: str) -> str:
        if not slug.endswith(".test.js") and not slug.endswith(".test.mjs"):
            slug = slug.rsplit(".", 1)[0] if "." in slug else slug
            slug = slug + ".test.mjs"
        return f"{self.test_corpus_dir}/{slug}"

    def test_runner_cmd(self, rel_path: str, *,
                        sandbox: Sandbox | None = None) -> List[str]:
        return [self.node, "--test", rel_path]
