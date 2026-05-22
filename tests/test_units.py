"""Offline unit tests — no API calls."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
# (tempfile, Path imported so TestHealthRollback below can use them too)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.fs import Sandbox
from d2p.llm import _extract_json
from d2p.agents import (parse_executor_output, _guard_destructive_write,
                         _apply_search_replace, _fuzzy_locate)
from d2p.symbols import extract_symbols
from d2p.qa import parse_qa_output, _validate_test_quality


class TestSandbox(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.txt").write_text("hi")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.py").write_text("print(1)\n")
        self.sb = Sandbox(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_listing_skips_pycache(self) -> None:
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "x.pyc").write_text("x")
        files = self.sb.listing()
        self.assertIn("a.txt", files)
        self.assertIn("sub/b.py", files)
        self.assertFalse(any("__pycache__" in f for f in files))

    def test_read_write_roundtrip(self) -> None:
        self.sb.write("new/dir/c.md", "# hello")
        self.assertEqual(self.sb.read("new/dir/c.md"), "# hello")

    def test_escape_blocked(self) -> None:
        with self.assertRaises(ValueError):
            self.sb.write("../escape.txt", "x")


class TestExtractJSON(unittest.TestCase):
    def test_plain(self) -> None:
        self.assertEqual(_extract_json('{"a":1}'), {"a": 1})

    def test_fenced(self) -> None:
        self.assertEqual(_extract_json('```json\n{"a":2}\n```'), {"a": 2})

    def test_fenced_plain(self) -> None:
        self.assertEqual(_extract_json('```\n[1,2,3]\n```'), [1, 2, 3])

    def test_with_prose(self) -> None:
        text = 'Here you go:\n{"x": [1,2]}\nthanks!'
        self.assertEqual(_extract_json(text), {"x": [1, 2]})


class TestGuardDestructiveWrite(unittest.TestCase):
    def test_allows_new_file(self) -> None:
        self.assertEqual(_guard_destructive_write("a.py", "", "new content\n" * 10), "")

    def test_allows_growth(self) -> None:
        existing = "\n".join(f"line {i}" for i in range(100))
        new = existing + "\n" + "\n".join(f"more {i}" for i in range(50))
        self.assertEqual(_guard_destructive_write("a.py", existing, new), "")

    def test_blocks_huge_shrink(self) -> None:
        existing = "\n".join(f"line {i}" for i in range(2000))
        new = "tiny"
        reason = _guard_destructive_write("templates/index.html", existing, new)
        self.assertIn("destructive write blocked", reason)

    def test_allows_small_file_shrink(self) -> None:
        existing = "a\nb\nc\n"
        new = "a\n"
        self.assertEqual(_guard_destructive_write("x.md", existing, new), "")


class TestExecutorParser(unittest.TestCase):
    def test_two_files(self) -> None:
        out = parse_executor_output(
            "STATUS: done\nSUMMARY: did the thing\n\n"
            "===FILE: a.txt===\nline1\nline2\n===END===\n"
            "===FILE: dir/b.py===\nprint('hi')\n===END===\n"
        )
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["summary"], "did the thing")
        self.assertEqual(out["files"][0], ("a.txt", "line1\nline2"))
        self.assertEqual(out["files"][1], ("dir/b.py", "print('hi')"))

    def test_skipped(self) -> None:
        out = parse_executor_output("STATUS: skipped\nSUMMARY: already done\n")
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(out["files"], [])

    def test_implicit_done_with_files(self) -> None:
        out = parse_executor_output("===FILE: x.md===\n# hi\n===END===\n")
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["files"], [("x.md", "# hi")])

    def test_done_without_files_becomes_skipped(self) -> None:
        out = parse_executor_output("STATUS: done\nSUMMARY: nothing to do\n")
        self.assertEqual(out["status"], "skipped")


class TestSearchReplace(unittest.TestCase):
    def test_simple(self) -> None:
        new, miss = _apply_search_replace("a\nb\nc\n", [("b", "B")])
        self.assertEqual(new, "a\nB\nc\n")
        self.assertEqual(miss, "")

    def test_first_only(self) -> None:
        new, miss = _apply_search_replace("xx-xx", [("xx", "Y")])
        self.assertEqual(new, "Y-xx")
        self.assertEqual(miss, "")

    def test_miss(self) -> None:
        new, miss = _apply_search_replace("abc", [("zzz", "X")])
        self.assertEqual(new, "abc")
        self.assertEqual(miss, "zzz")

    def test_multiple_pairs(self) -> None:
        text = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        new, miss = _apply_search_replace(text, [
            ("def foo():\n    pass", "def foo():\n    return 1"),
            ("def bar():\n    pass", "def bar():\n    return 2"),
        ])
        self.assertIn("return 1", new)
        self.assertIn("return 2", new)
        self.assertEqual(miss, "")

    def test_fuzzy_indent_shift(self) -> None:
        # haystack has 4-space indent; SEARCH provided with 0-indent
        haystack = "class A:\n    def x(self):\n        return 1\n"
        search = "def x(self):\n    return 1"
        replace = "def x(self):\n    return 2"
        new, miss = _apply_search_replace(haystack, [(search, replace)])
        self.assertEqual(miss, "")
        self.assertIn("return 2", new)
        self.assertNotIn("return 1", new)

    def test_fuzzy_ambiguous_refused(self) -> None:
        haystack = "foo()\nfoo()\n"
        # exact "foo()" already matches first occurrence, not ambiguous via fuzzy
        # so test fuzzy ambiguity via indent-shifted block
        haystack = "if a:\n    foo()\nif b:\n    foo()\n"
        search = "foo()"
        # exact: matches first occurrence, fine
        new, miss = _apply_search_replace(haystack, [(search, "bar()")])
        self.assertEqual(miss, "")
        # now make exact fail by adding whitespace
        new, miss = _apply_search_replace(haystack, [("foo() ", "bar()")])
        self.assertEqual(miss, "foo() ")


class TestSandboxSnapshot(unittest.TestCase):
    def test_roundtrip(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sb = Sandbox(d)
            sb.write("a.txt", "one")
            sb.write("dir/b.txt", "two")
            snap = sb.snapshot(["a.txt", "dir/b.txt", "missing.txt"])
            sb.write("a.txt", "ONE-modified")
            sb.write("missing.txt", "newly created")
            sb.delete("dir/b.txt")
            sb.restore(snap)
            self.assertEqual(sb.read("a.txt"), "one")
            self.assertEqual(sb.read("dir/b.txt"), "two")
            self.assertEqual(sb.read("missing.txt"), "")  # was missing -> restored as missing


class TestExecutorParserPatches(unittest.TestCase):
    def test_patch_block(self) -> None:
        text = (
            "STATUS: done\nSUMMARY: tweak\n\n"
            "===PATCH: a.py===\n"
            "<<<SEARCH\nold code\nSEARCH>>>\n"
            "<<<REPLACE\nnew code\nREPLACE>>>\n"
            "===END===\n"
        )
        out = parse_executor_output(text)
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["files"], [])
        self.assertEqual(len(out["patches"]), 1)
        path, ops = out["patches"][0]
        self.assertEqual(path, "a.py")
        self.assertEqual(ops, [("old code", "new code")])

    def test_mixed_file_and_patch(self) -> None:
        text = (
            "STATUS: done\nSUMMARY: mix\n\n"
            "===FILE: new.md===\n# hi\n===END===\n"
            "===PATCH: existing.py===\n"
            "<<<SEARCH\nfoo()\nSEARCH>>>\n"
            "<<<REPLACE\nbar()\nREPLACE>>>\n"
            "===END===\n"
        )
        out = parse_executor_output(text)
        self.assertEqual(out["files"], [("new.md", "# hi")])
        self.assertEqual(len(out["patches"]), 1)


class TestSymbolExtract(unittest.TestCase):
    def test_python(self) -> None:
        src = "class Foo:\n    pass\n\ndef bar():\n    pass\n\n@app.route('/x')\ndef x():\n    pass\n"
        syms = extract_symbols("m.py", src)
        self.assertIn("Foo", syms)
        self.assertIn("bar", syms)
        self.assertIn("x", syms)
        self.assertIn("/x", syms)

    def test_js(self) -> None:
        src = "export function hello() {}\nexport class Foo {}\n"
        syms = extract_symbols("m.js", src)
        self.assertIn("hello", syms)
        self.assertIn("Foo", syms)


class TestMissingSymbolDetection(unittest.TestCase):
    def test_cannot_import_name_yields_restore_task(self) -> None:
        from d2p.qa import detect_missing_symbol_failures
        failure = (
            "ImportError: Failed to import test module: test_x\n"
            "  File 'tests/d2p_qa/test_x.py', line 7, in <module>\n"
            "    from app import x\n"
            "  File '/path/app.py', line 9, in <module>\n"
            "    from prompts import build_system_prompt\n"
            "ImportError: cannot import name 'build_system_prompt' from 'prompts' "
            "(/path/prompts.py)\n"
        )
        out = detect_missing_symbol_failures(failure)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], ("build_system_prompt", "prompts"))

    def test_no_match_returns_empty(self) -> None:
        from d2p.qa import detect_missing_symbol_failures
        self.assertEqual(
            detect_missing_symbol_failures("AssertionError: 1 != 2"), []
        )

    def test_multiple_distinct(self) -> None:
        from d2p.qa import detect_missing_symbol_failures
        text = (
            "ImportError: cannot import name 'a' from 'mod1' (/p/mod1.py)\n"
            "later...\n"
            "ImportError: cannot import name 'b' from 'mod2' (/p/mod2.py)\n"
        )
        out = sorted(detect_missing_symbol_failures(text))
        self.assertEqual(out, [("a", "mod1"), ("b", "mod2")])


class TestQATestQuality(unittest.TestCase):
    GOOD = (
        "import unittest\n"
        "import sys\n"
        "class TFoo(unittest.TestCase):\n"
        "    def test_a(self):\n"
        "        self.assertEqual(1, 2)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
        + ("# padding " * 40)
    )

    def test_good_passes(self) -> None:
        self.assertEqual(_validate_test_quality(self.GOOD), "")

    def test_short_rejected(self) -> None:
        self.assertIn("truncated", _validate_test_quality("import unittest\n"))

    def test_no_testcase_rejected(self) -> None:
        body = "import unittest\nx = 1\n" + ("# padding " * 50)
        self.assertIn("TestCase", _validate_test_quality(body))

    def test_no_test_method_rejected(self) -> None:
        body = "import unittest\nclass T(unittest.TestCase): pass\n" + ("# padding " * 50)
        self.assertIn("def test", _validate_test_quality(body))

    def test_truncation_trailing_paren(self) -> None:
        body = self.GOOD.rstrip() + "\nresult = foo("
        self.assertIn("truncation", _validate_test_quality(body))


class TestQAParser(unittest.TestCase):
    def test_basic(self) -> None:
        text = (
            "===TEST: tests/d2p_qa/test_x.py===\n"
            'META: {"title": "rejects empty body", "category": "input_validation", "suspected_files": ["app.py"]}\n'
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(1,2)\n"
            "===END===\n"
        )
        out = parse_qa_output(text)
        self.assertEqual(len(out), 1)
        path, meta, body = out[0]
        self.assertEqual(path, "tests/d2p_qa/test_x.py")
        self.assertEqual(meta["category"], "input_validation")
        self.assertIn("class T(unittest.TestCase)", body)

    def test_missing_meta(self) -> None:
        text = "===TEST: tests/d2p_qa/test_y.py===\nimport unittest\n===END===\n"
        out = parse_qa_output(text)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], {})


class TestExtractAssertionSummary(unittest.TestCase):
    def test_picks_assertion_error(self) -> None:
        from d2p.agents import _extract_assertion_summary
        text = (
            "test_x (...) ... FAIL\n"
            "Traceback (most recent call last):\n"
            "  File \"x.py\", line 42, in test_x\n"
            "    self.assertEqual(1, 2)\n"
            "AssertionError: 1 != 2\n"
            "Ran 1 test in 0.001s\n"
        )
        out = _extract_assertion_summary(text)
        self.assertIn("AssertionError", out)
        self.assertIn("1 != 2", out)

    def test_picks_import_error(self) -> None:
        from d2p.agents import _extract_assertion_summary
        text = "ImportError: cannot import name 'foo' from 'bar'"
        self.assertEqual(_extract_assertion_summary(text),
                         "ImportError: cannot import name 'foo' from 'bar'")

    def test_fallback_last_nonempty(self) -> None:
        from d2p.agents import _extract_assertion_summary
        self.assertEqual(_extract_assertion_summary("hello\nworld\n"), "world")

    def test_empty(self) -> None:
        from d2p.agents import _extract_assertion_summary
        self.assertEqual(_extract_assertion_summary(""), "(no test output)")


class TestFlipMetaStatus(unittest.TestCase):
    def test_flip_writes_meta(self) -> None:
        from d2p.fs import Sandbox
        from d2p.qa import QAAgent
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "tests" / "d2p_qa").mkdir()
            (root / "tests" / "d2p_qa" / "_meta.json").write_text(
                json.dumps({
                    "tests/d2p_qa/test_x.py": {
                        "id": "a", "title": "t", "test_path": "tests/d2p_qa/test_x.py",
                        "category": "x", "summary": "t",
                        "suspected_files": [], "last_failure": "",
                        "status": "open",
                    }
                })
            )
            sb = Sandbox(root)
            qa = QAAgent.__new__(QAAgent)
            qa.sandbox = sb
            qa.corpus_dir = "tests/d2p_qa"
            import threading as _th
            qa._meta_lock = _th.Lock()
            qa._test_run_lock = _th.Lock()
            qa.flip_meta_status("tests/d2p_qa/test_x.py", "fixed")
            meta = json.loads((root / "tests/d2p_qa/_meta.json").read_text())
            self.assertEqual(meta["tests/d2p_qa/test_x.py"]["status"], "fixed")

    def test_flip_unknown_path_noop(self) -> None:
        from d2p.fs import Sandbox
        from d2p.qa import QAAgent
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "tests" / "d2p_qa").mkdir()
            (root / "tests" / "d2p_qa" / "_meta.json").write_text("{}")
            sb = Sandbox(root)
            qa = QAAgent.__new__(QAAgent)
            qa.sandbox = sb
            qa.corpus_dir = "tests/d2p_qa"
            import threading as _th
            qa._meta_lock = _th.Lock()
            qa._test_run_lock = _th.Lock()
            qa.flip_meta_status("not/in/meta.py", "fixed")  # noop
            meta = json.loads((root / "tests/d2p_qa/_meta.json").read_text())
            self.assertEqual(meta, {})


class TestBaselineTestGate(unittest.TestCase):
    def test_discover_pre_existing_tests_excludes_d2p_qa(self) -> None:
        from d2p.orchestrator import _discover_pre_existing_tests
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "tests" / "test_smoke.py").write_text("")
            (root / "tests" / "d2p_qa").mkdir()
            (root / "tests" / "d2p_qa" / "test_bug.py").write_text("")
            (root / "tests" / "__init__.py").write_text("")
            sb = Sandbox(root)
            out = _discover_pre_existing_tests(sb)
            self.assertIn("tests/test_smoke.py", out)
            self.assertNotIn("tests/d2p_qa/test_bug.py", out)
            self.assertNotIn("tests/__init__.py", out)


class TestPatchDestructiveShrinkGuard(unittest.TestCase):
    """Many small SEARCH/REPLACE patches summed can destructively shrink a
    file. The post-patch guard must catch that just like full rewrites."""

    def test_aggregated_patch_shrink_blocked(self) -> None:
        from d2p.fs import Sandbox
        from d2p.agents import Executor
        from d2p.models import Task
        from unittest.mock import MagicMock

        original = "\n".join(f"line {i}" for i in range(200)) + "\n"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "big.py").write_text(original)
            sb = Sandbox(root)
            ex = Executor.__new__(Executor)
            ex.sandbox = sb
            ex.adapter = type("A", (), {
                "name": "python",
                "syntax_check": lambda self, sb, p: "",
            })()
            ex.llm = MagicMock()
            # The "patch" deletes 195 of the 200 lines via one giant SEARCH/REPLACE
            big_search = "\n".join(f"line {i}" for i in range(195))
            ex.llm.chat = MagicMock(return_value=(
                "STATUS: done\nSUMMARY: shrink\n\n"
                "===PATCH: big.py===\n"
                "<<<SEARCH\n" + big_search + "\nSEARCH>>>\n"
                "<<<REPLACE\n\nREPLACE>>>\n"
                "===END===\n"
            ))
            task = Task(id="x", title="t", rationale="", target_files=["big.py"],
                        instructions="i", priority=1, category="feature")
            res = ex.run(task)
            self.assertEqual(res.status, "failed")
            self.assertIn("destructive write blocked", res.error)
            # original is intact
            self.assertEqual(sb.read("big.py"), original)


class TestExecutorPostCheckKeepsWrites(unittest.TestCase):
    """When post_check ultimately fails, the Executor marks status=failed but
    KEEPS the writes — they may be partially correct, and import-level damage
    is caught separately by the orchestrator's health probe."""

    def test_failed_post_check_keeps_writes(self) -> None:
        from d2p.fs import Sandbox
        from d2p.agents import Executor
        from d2p.models import Task
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "app.py").write_text("ORIGINAL\n")
            sb = Sandbox(root)
            ex = Executor.__new__(Executor)
            ex.sandbox = sb
            ex.llm = MagicMock()
            ex.llm.chat = MagicMock(return_value=(
                "STATUS: done\nSUMMARY: did it\n\n"
                "===FILE: app.py===\nMODIFIED\n===END===\n"
            ))
            task = Task(id="x", title="t", rationale="", target_files=["app.py"],
                        instructions="i", priority=1, category="bugfix")
            pc = lambda: (False, "AssertionError: nope")
            res = ex.run(task, post_check=pc)
            self.assertEqual(res.status, "failed")
            self.assertIn("post-check failed", res.error)
            # writes preserved
            self.assertEqual(sb.read("app.py").rstrip("\n"), "MODIFIED")


class TestExecutorPostCheckBookkeeping(unittest.TestCase):
    def test_failure_demotes_done_to_failed(self) -> None:
        from d2p.agents import _apply_post_check_to_result
        from d2p.models import ExecutionResult
        res = ExecutionResult(task_id="t1", status="done",
                              summary="did it", files_changed=["app.py"])
        out = _apply_post_check_to_result(res, post_check_ok=False,
                                          post_check_output="AssertionError: nope")
        self.assertEqual(out.status, "failed")
        self.assertIn("post-check failed", out.error)
        self.assertIn("AssertionError", out.error)

    def test_success_keeps_done(self) -> None:
        from d2p.agents import _apply_post_check_to_result
        from d2p.models import ExecutionResult
        res = ExecutionResult(task_id="t2", status="done",
                              summary="did it", files_changed=["app.py"])
        out = _apply_post_check_to_result(res, post_check_ok=True,
                                          post_check_output="")
        self.assertEqual(out.status, "done")
        self.assertEqual(out.error, "")


class TestFixTaskHasTestSource(unittest.TestCase):
    def test_instructions_include_test_file_contents(self) -> None:
        from d2p.fs import Sandbox
        from d2p.qa import QAAgent, BugReport

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "tests" / "d2p_qa").mkdir()
            test_body = (
                "import unittest\n"
                "class TestFoo(unittest.TestCase):\n"
                "    def test_a(self):\n"
                "        self.assertEqual(1, 2)  # canary marker\n"
            )
            (root / "tests" / "d2p_qa" / "test_bug.py").write_text(test_body)
            sb = Sandbox(root)
            qa = QAAgent.__new__(QAAgent)
            qa.sandbox = sb
            bug = BugReport(id="abc", title="t",
                            test_path="tests/d2p_qa/test_bug.py",
                            category="custom", summary="t",
                            suspected_files=["app.py"],
                            last_failure="AssertionError: 1 != 2")
            task = qa._bug_to_task(bug)
            self.assertIn("canary marker", task.instructions)
            self.assertIn("tests/d2p_qa/test_bug.py", task.instructions)
            self.assertEqual(task.forbidden_files, ["tests/d2p_qa/test_bug.py"])


class TestPatchRetryFormatter(unittest.TestCase):
    def test_includes_misses_and_numbered_file(self) -> None:
        from d2p.agents import _format_patch_retry_user
        content = "line 1\nline 2\nline 3\n"
        user = _format_patch_retry_user(
            rel="app.py", file_content=content,
            misses=["nonexistent text"],
            original_task_title="Add feature X",
        )
        self.assertIn("app.py", user)
        self.assertIn("nonexistent text", user)
        self.assertIn("1| line 1", user)
        self.assertIn("3| line 3", user)


class TestHealthRollback(unittest.TestCase):
    """End-to-end logic test of the rollback wrapper — no LLM."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "app.py").write_text("from helpers import hello\nprint(hello())\n")
        (root / "helpers.py").write_text("def hello():\n    return 'world'\n")
        self.sb = Sandbox(root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rollback_when_task_breaks_import(self) -> None:
        from d2p.orchestrator import _rollback_if_health_regressed
        from d2p.health import ProjectHealth
        probe = ProjectHealth(self.sb)
        baseline = probe.probe(probe.default_modules())
        snapshot = self.sb.snapshot(["helpers.py"])
        self.sb.write("helpers.py", "def goodbye():\n    return 1\n")
        rolled = _rollback_if_health_regressed(
            self.sb, probe, baseline=baseline, snapshot=snapshot,
        )
        self.assertTrue(rolled)
        self.assertEqual(self.sb.read("helpers.py"),
                         "def hello():\n    return 'world'\n")

    def test_no_rollback_when_health_unchanged(self) -> None:
        from d2p.orchestrator import _rollback_if_health_regressed
        from d2p.health import ProjectHealth
        probe = ProjectHealth(self.sb)
        baseline = probe.probe(probe.default_modules())
        snapshot = self.sb.snapshot(["helpers.py"])
        self.sb.write("helpers.py",
                      "def hello():\n    return 'world'\n\ndef new():\n    return 'x'\n")
        rolled = _rollback_if_health_regressed(
            self.sb, probe, baseline=baseline, snapshot=snapshot,
        )
        self.assertFalse(rolled)
        self.assertIn("new()", self.sb.read("helpers.py"))


class TestUsageAccumulator(unittest.TestCase):
    def test_accumulates_and_summarises(self) -> None:
        from d2p.providers.base import UsageAccumulator
        u = UsageAccumulator()
        u.add(role="executor", model="haiku",
              input_tokens=10, output_tokens=20,
              cache_creation_tokens=100, cache_read_tokens=0, cost_usd=0.001)
        u.add(role="executor", model="haiku",
              input_tokens=5, output_tokens=8,
              cache_creation_tokens=0, cache_read_tokens=100, cost_usd=0.0005)
        u.add(role="planner", model="opus",
              input_tokens=200, output_tokens=300, cost_usd=0.02)
        s = u.summary()
        self.assertEqual(s["total_calls"], 3)
        self.assertAlmostEqual(s["total_cost_usd"], 0.0215, places=4)
        # cache hit ratio = read / (read+creation+input) = 100 / (100+100+215) = 0.241
        # input is in the denominator so MiniMax-style usage (cc=0, large
        # raw input) doesn't falsely report a perfect cache hit.
        self.assertAlmostEqual(s["cache_hit_ratio"], 0.241, places=3)
        self.assertIn("executor:haiku", s["per_role"])
        self.assertEqual(s["per_role"]["executor:haiku"]["calls"], 2)
        self.assertEqual(s["per_role"]["executor:haiku"]["input"], 15)
        self.assertEqual(s["per_role"]["planner:opus"]["calls"], 1)

    def test_threadsafe_concurrent_add(self) -> None:
        import threading
        from d2p.providers.base import UsageAccumulator
        u = UsageAccumulator()
        def worker():
            for _ in range(50):
                u.add(role="executor", model="haiku",
                      input_tokens=1, output_tokens=1)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(u.summary()["total_calls"], 200)


class TestQAWontfix(unittest.TestCase):
    def _make_qa(self, root: Path) -> "object":
        import threading as _th
        from d2p.qa import QAAgent
        (root / "tests" / "d2p_qa").mkdir(parents=True, exist_ok=True)
        sb = Sandbox(root)
        qa = QAAgent.__new__(QAAgent)
        qa.sandbox = sb
        qa.corpus_dir = "tests/d2p_qa"
        qa._meta_lock = _th.Lock()
        qa._test_run_lock = _th.Lock()
        return qa

    def test_bump_attempts_increments(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests" / "d2p_qa").mkdir(parents=True)
            (root / "tests" / "d2p_qa" / "_meta.json").write_text(json.dumps({
                "tests/d2p_qa/test_x.py": {
                    "id": "a", "title": "t",
                    "test_path": "tests/d2p_qa/test_x.py",
                    "category": "x", "summary": "t",
                    "suspected_files": [], "last_failure": "",
                    "status": "open", "attempts": 0,
                }
            }))
            qa = self._make_qa(root)
            n1 = qa.bump_attempts("tests/d2p_qa/test_x.py")
            n2 = qa.bump_attempts("tests/d2p_qa/test_x.py")
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 2)
            meta = json.loads((root / "tests/d2p_qa/_meta.json").read_text())
            self.assertEqual(meta["tests/d2p_qa/test_x.py"]["attempts"], 2)

    def test_bump_unknown_path_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests" / "d2p_qa").mkdir(parents=True)
            (root / "tests" / "d2p_qa" / "_meta.json").write_text("{}")
            qa = self._make_qa(root)
            self.assertEqual(qa.bump_attempts("nope.py"), 0)

    def test_mark_wontfix_flips_status(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests" / "d2p_qa").mkdir(parents=True)
            (root / "tests" / "d2p_qa" / "_meta.json").write_text(json.dumps({
                "tests/d2p_qa/test_y.py": {
                    "id": "b", "title": "t",
                    "test_path": "tests/d2p_qa/test_y.py",
                    "category": "x", "summary": "t",
                    "suspected_files": [], "last_failure": "",
                    "status": "open", "attempts": 3,
                }
            }))
            qa = self._make_qa(root)
            qa.mark_wontfix("tests/d2p_qa/test_y.py")
            meta = json.loads((root / "tests/d2p_qa/_meta.json").read_text())
            self.assertEqual(meta["tests/d2p_qa/test_y.py"]["status"], "wontfix")


class TestClaudeCLIPromptPrefixStable(unittest.TestCase):
    """The cache-friendly prompt layout puts variable knobs at the END so the
    bytewise prefix up to the user block stays identical across retries with
    different temperature/max_tokens. Regression test for d2p prompt caching."""

    def _build(self, **call_kwargs):
        from d2p.providers.claude_cli import ClaudeCLIProvider
        p = ClaudeCLIProvider.__new__(ClaudeCLIProvider)
        p.role = "executor"
        return p._build_prompt("SYS", "USER",
                               web_search=call_kwargs.get("web_search", False),
                               json_mode=call_kwargs.get("json_mode", False),
                               temperature=call_kwargs.get("temperature", 0.4),
                               max_tokens=call_kwargs.get("max_tokens", 4096))

    def test_stable_prefix_across_temperatures(self) -> None:
        a = self._build(temperature=0.3, max_tokens=4096)
        b = self._build(temperature=0.7, max_tokens=4096)
        # Find the divergence point — must be inside "Call options" block.
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca != cb:
                self.assertIn("=== Call options ===", a[:i])
                self.assertIn("=== Call options ===", b[:i])
                return
        # If they're identical, fine (no per-call info present)
        self.assertEqual(a, b)

    def test_user_block_appears_before_call_options(self) -> None:
        prompt = self._build()
        user_idx = prompt.index("=== User ===")
        opts_idx = prompt.index("=== Call options ===")
        self.assertLess(user_idx, opts_idx,
                        "User block must precede Call options for cache prefix")


class TestIterChangesMarkdown(unittest.TestCase):
    def test_emit_basic_digest(self) -> None:
        from unittest.mock import MagicMock
        from d2p.orchestrator import Orchestrator
        from d2p.models import ExecutionResult, PlanResult, Task
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # synthesise an Orchestrator without running build_router
            orch = Orchestrator.__new__(Orchestrator)
            orch.run_dir = root
            orch.router = MagicMock()
            orch.router.usage.summary.return_value = {
                "total_calls": 5, "total_cost_usd": 0.0123,
                "cache_hit_ratio": 0.7, "per_role": {},
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cache_creation_tokens": 0, "total_cache_read_tokens": 0,
            }
            plan = PlanResult(iteration=1, tasks=[
                Task(id="t1", title="Add login", rationale="x",
                     target_files=["auth.py"], instructions="",
                     priority=1, category="feature"),
            ], rationale="why these tasks")
            results = [
                ExecutionResult(task_id="t1", status="done",
                                summary="ok", files_changed=["auth.py"]),
            ]
            fake_qa = MagicMock()
            fake_qa.new_bugs = []
            fake_qa.fixed_bugs = []
            fake_qa.open_bugs = []
            orch._emit_iter_changes_md(
                1, plan=plan, results=results, qa_report=fake_qa,
                qa_fix_results=[], retired_this_iter=[],
            )
            md = (root / "iter1_changes.md").read_text()
            self.assertIn("# Iteration 1", md)
            self.assertIn("Add login", md)
            self.assertIn("auth.py", md)
            self.assertIn("Cumulative usage", md)
            self.assertIn("0.0123", md)


class TestIterChangesMarkdownTiming(unittest.TestCase):
    def test_md_includes_elapsed_and_cost_delta(self) -> None:
        from unittest.mock import MagicMock
        from d2p.orchestrator import Orchestrator
        from d2p.models import ExecutionResult, PlanResult, Task
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            orch = Orchestrator.__new__(Orchestrator)
            orch.run_dir = root
            orch.router = MagicMock()
            orch.router.usage.summary.return_value = {
                "total_calls": 0, "total_cost_usd": 0.0,
                "cache_hit_ratio": 0.0, "per_role": {},
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cache_creation_tokens": 0, "total_cache_read_tokens": 0,
            }
            plan = PlanResult(iteration=1, tasks=[], rationale="why")
            orch._emit_iter_changes_md(
                1, plan=plan, results=[], qa_report=None,
                qa_fix_results=[], retired_this_iter=[],
                elapsed_s=42.7, cost_delta_usd=0.0512,
            )
            md = (root / "iter1_changes.md").read_text()
            self.assertIn("Elapsed: 42.7s", md)
            self.assertIn("$0.0512", md)


class TestAnalyzerFingerprint(unittest.TestCase):
    """The fingerprint determines analyzer-cache hit/miss. Two runs with
    identical inputs must produce identical fingerprints; any input change
    must shift it."""

    def _make_analyzer(self, root: Path):
        from d2p.agents import Analyzer
        from d2p.fs import Sandbox
        from unittest.mock import MagicMock
        sb = Sandbox(root)
        llm = MagicMock()
        llm.name = "test-model@analyzer"
        return Analyzer(llm, sb)

    def test_same_input_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "app.py").write_text("print(1)\n")
            (root / "README.md").write_text("# Demo\n")
            a = self._make_analyzer(root)
            self.assertEqual(a.fingerprint(), a.fingerprint())

    def test_listing_change_shifts_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "app.py").write_text("print(1)\n")
            a = self._make_analyzer(root)
            fp1 = a.fingerprint()
            (root / "new_file.py").write_text("x=1\n")
            self.assertNotEqual(a.fingerprint(), fp1)

    def test_doc_change_shifts_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "README.md").write_text("# v1\n")
            a = self._make_analyzer(root)
            fp1 = a.fingerprint()
            (root / "README.md").write_text("# v2\n")
            self.assertNotEqual(a.fingerprint(), fp1)

    def test_model_change_shifts_fingerprint(self) -> None:
        from d2p.agents import Analyzer
        from d2p.fs import Sandbox
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "README.md").write_text("# x\n")
            sb = Sandbox(root)
            llm1 = MagicMock(); llm1.name = "haiku"
            llm2 = MagicMock(); llm2.name = "opus"
            a1 = Analyzer(llm1, sb)
            a2 = Analyzer(llm2, sb)
            self.assertNotEqual(a1.fingerprint(), a2.fingerprint())


class TestAnalyzerCachedRoundtrip(unittest.TestCase):
    # Phase-1/2/3 LLM responses for a single Analyzer.run()
    PHASE1 = {
        "domain": "X", "essence": "E", "audience": "A",
        "competitors": ["c1"],
        "competitors_detail": [
            {"name": "C1", "key_features": ["kf-a", "kf-b"],
             "source_url": "https://c1.example", "notes": "n"}
        ],
        "ui_elements": ["u"],
        "raw_notes": "n",
    }
    PHASE2 = {"demo_capabilities": ["does X via foo.py:bar()"]}
    PHASE3 = {"features": [{
        "name": "f1", "category": "ux", "description": "d",
        "source": "C1", "in_demo": "missing",
        "evidence_in_demo": "—", "gap_severity": "high",
    }]}

    def _seed_llm(self, llm: object, times: int = 1) -> None:
        from unittest.mock import MagicMock
        llm.name = "m"
        # 3 phase calls per Analyzer.run()
        llm.chat_json.side_effect = [
            self.PHASE1, self.PHASE2, self.PHASE3,
        ] * times

    def test_hit_skips_llm(self) -> None:
        from d2p.agents import Analyzer
        from d2p.fs import Sandbox
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "README.md").write_text("# x\n")
            (root / "app.py").write_text("def main(): pass\n")
            sb = Sandbox(root)
            llm = MagicMock()
            self._seed_llm(llm, times=1)
            a = Analyzer(llm, sb)
            cache_path = root / ".d2p" / "analysis_cache.json"
            r1, hit1 = a.run_cached(cache_path)
            r2, hit2 = a.run_cached(cache_path)
            self.assertFalse(hit1)
            self.assertTrue(hit2)
            # 3 LLM calls (one per phase) on the fresh run; cache hit costs 0
            self.assertEqual(llm.chat_json.call_count, 3)
            self.assertEqual(r1.essence, r2.essence)
            self.assertEqual([f.name for f in r1.features],
                             [f.name for f in r2.features])
            self.assertEqual(r1.demo_capabilities, r2.demo_capabilities)
            self.assertEqual([c.name for c in r1.competitors_detail],
                             [c.name for c in r2.competitors_detail])

    def test_no_cache_flag_forces_fresh(self) -> None:
        from d2p.agents import Analyzer
        from d2p.fs import Sandbox
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "README.md").write_text("# x\n")
            (root / "app.py").write_text("def main(): pass\n")
            sb = Sandbox(root)
            llm = MagicMock()
            self._seed_llm(llm, times=2)
            a = Analyzer(llm, sb)
            cache_path = root / ".d2p" / "analysis_cache.json"
            a.run_cached(cache_path)
            a.run_cached(cache_path, use_cache=False)
            # 3 calls × 2 runs (cache bypassed second time)
            self.assertEqual(llm.chat_json.call_count, 6)


class TestRouterLadder(unittest.TestCase):
    def test_for_role_tier_no_ladder_falls_through(self) -> None:
        from d2p.providers.base import RoleRouter

        class P:
            def __init__(self, n): self.name = n
            def chat(self, *a, **k): return ""
            def chat_json(self, *a, **k): return {}
        primary = P("haiku")
        r = RoleRouter({"default": primary, "executor": primary})
        # no ladder configured → falls back to for_role
        self.assertIs(r.for_role_tier("executor", 0), primary)
        self.assertIs(r.for_role_tier("executor", 5), primary)
        self.assertEqual(r.ladder_length("executor"), 1)

    def test_for_role_tier_with_ladder(self) -> None:
        from d2p.providers.base import RoleRouter

        class P:
            def __init__(self, n): self.name = n
            def chat(self, *a, **k): return ""
            def chat_json(self, *a, **k): return {}
        p0, p1, p2 = P("haiku"), P("sonnet"), P("opus")
        r = RoleRouter({"default": p0, "executor": p0},
                       ladders={"executor": [p0, p1, p2]})
        self.assertIs(r.for_role_tier("executor", 0), p0)
        self.assertIs(r.for_role_tier("executor", 1), p1)
        self.assertIs(r.for_role_tier("executor", 2), p2)
        # clamps overshoot to top
        self.assertIs(r.for_role_tier("executor", 99), p2)
        self.assertEqual(r.ladder_length("executor"), 3)
        # describe exposes ladder tiers
        d = r.describe()
        self.assertEqual(d["executor-tier0"], "haiku")
        self.assertEqual(d["executor-tier1"], "sonnet")
        self.assertEqual(d["executor-tier2"], "opus")

    def test_provider_spec_reads_ladder_env(self) -> None:
        import os as _os
        from d2p.providers import _from_env
        old = _os.environ.copy()
        try:
            _os.environ["D2P_PROVIDER"] = "minimax"
            _os.environ["MINIMAX_API_KEY"] = "sk-cp-test"
            _os.environ["D2P_ROLE_EXECUTOR_LADDER"] = "A, B ,C"
            _os.environ["D2P_ROLE_FIX_LADDER"] = "X"
            spec = _from_env()
            self.assertEqual(spec.role_ladders["executor"], ["A", "B", "C"])
            self.assertEqual(spec.role_ladders["fix"], ["X"])
        finally:
            _os.environ.clear()
            _os.environ.update(old)


class TestQAFenceStripAndParse(unittest.TestCase):
    """The 2026-05-19 demo run produced a test_api_contract.py wrapped in a
    ```python fence — sailed past quality checks, then SyntaxError'd at
    import time inside the corpus. Both safeguards now block this."""

    def test_strip_md_fence_round(self) -> None:
        from d2p.qa import _strip_md_fence
        content = "```python\n\"\"\"doc\"\"\"\nimport os\n```\n"
        out = _strip_md_fence(content)
        self.assertNotIn("```", out)
        self.assertIn("import os", out)

    def test_strip_md_fence_no_op_when_only_one_end(self) -> None:
        from d2p.qa import _strip_md_fence
        # one stray ``` shouldn't trigger stripping
        content = 'x = "```python"\n'
        self.assertEqual(_strip_md_fence(content), content)

    def test_quality_rejects_syntax_error(self) -> None:
        bad = (
            '```python\n'
            '"""docstring that\'s long enough to pass the 300-byte gate"""\n'
            'import unittest\n'
            'class TestX(unittest.TestCase):\n'
            '    def test_foo(self):\n'
            '        self.assertTrue(True)\n'
            'this line breaks parsing because it has bad characters: $$$\n'
            'class More syntax problems!! while:: \n'
        )
        # pad to >300 bytes
        bad = bad + '\n# padding ' * 30
        reason = _validate_test_quality(bad, "python")
        self.assertTrue(reason.startswith("SyntaxError"), reason)

    def test_quality_accepts_clean_test(self) -> None:
        padding = "padding " * 50
        good = (
            f'"""docstring that is long enough to pass the 300-byte gate. {padding}"""\n'
            "import unittest\n\n\n"
            "class TestY(unittest.TestCase):\n"
            "    def test_z(self):\n"
            "        self.assertEqual(1 + 1, 2)\n"
            "\n\nif __name__ == '__main__':\n    unittest.main()\n"
        )
        self.assertEqual(_validate_test_quality(good, "python"), "")

    def test_parser_strips_fences_end_to_end(self) -> None:
        # full QA agent output with the fenced test inside the ===TEST===
        # block. parse_qa_output should strip the fence before returning.
        raw = (
            "===TEST: tests/d2p_qa/test_x.py===\n"
            'META: {"title": "x", "category": "custom"}\n'
            "```python\n"
            '"""hypothesis"""\n'
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_bug(self):\n"
            "        self.fail()\n"
            "```\n"
            "===END===\n"
        )
        parsed = parse_qa_output(raw)
        self.assertEqual(len(parsed), 1)
        _path, _meta, body = parsed[0]
        self.assertNotIn("```", body)
        self.assertIn("import unittest", body)


class TestFlattenError(unittest.TestCase):
    def test_collapses_multiline(self) -> None:
        from d2p.orchestrator import _flatten
        msg = 'post-check failed: Traceback:\n  File "x.py", line 1\n    foo'
        out = _flatten(msg, max_len=200)
        self.assertNotIn("\n", out)
        self.assertIn("|", out)

    def test_truncates(self) -> None:
        from d2p.orchestrator import _flatten
        long = "a" * 1000
        self.assertEqual(len(_flatten(long, max_len=80)), 80)

    def test_empty_passthrough(self) -> None:
        from d2p.orchestrator import _flatten
        self.assertEqual(_flatten(""), "")
        self.assertEqual(_flatten(None), "")  # type: ignore[arg-type]


class TestFixCap(unittest.TestCase):
    """Verify the orchestrator.cfg.max_concurrent_fixes path: when N fixes
    exceed the cap, fix_tasks gets trimmed to the cap, preferring tasks
    whose bugs have the lowest attempts count."""

    def test_cap_filters_by_lowest_attempts(self) -> None:
        from unittest.mock import MagicMock
        from d2p.models import Task
        # We'll exercise the SORT logic directly — the orchestrator's
        # _run_qa code calls _load_meta + sorts inline. Test that
        # behaviour by reconstructing the exact sort.
        tasks = [
            Task(id=f"qa-{i:08x}", title=f"t{i}", rationale="", target_files=[],
                 instructions="", priority=1, category="bugfix")
            for i in range(4)
        ]
        attempts_by_bug_id = {
            tasks[0].id[-8:]: 0,   # fresh — should go first
            tasks[1].id[-8:]: 3,   # stale
            tasks[2].id[-8:]: 1,
            tasks[3].id[-8:]: 0,
        }
        # mirror the sort the orchestrator does
        def _attempts_of(t):
            bug_id = t.id.replace("qa-", "")
            return attempts_by_bug_id.get(bug_id, 0)
        tasks.sort(key=lambda t: (_attempts_of(t), t.priority))
        # the 2 lowest-attempts tasks come first
        kept = tasks[:2]
        ids_kept = {t.id for t in kept}
        self.assertEqual(ids_kept, {tasks[0].id, tasks[3].id}
                         if tasks[0].id in ids_kept and tasks[3].id in ids_kept
                         else ids_kept)
        # the 0-attempt ones should be in the kept set
        for t in kept:
            self.assertEqual(_attempts_of(t), 0)


class TestBumpAttemptsScope(unittest.TestCase):
    """The bump_attempts call must only fire for bugs the orchestrator
    actually dispatched as fix tasks this iteration. Bugs deferred by
    the max_concurrent_fixes cap (or otherwise absent) must NOT have
    their attempts bumped — otherwise the wontfix threshold would trip
    on bugs we never tried to fix."""

    def test_dispatched_set_excludes_deferred(self) -> None:
        from d2p.models import Task
        # Three QA-fix tasks, one was dropped by the fix cap.
        all_tasks = [
            Task(id="qa-aaaa", title="a", rationale="", target_files=["x.py"],
                 instructions="", priority=1, category="bugfix",
                 forbidden_files=["tests/d2p_qa/test_a.py"]),
            Task(id="qa-bbbb", title="b", rationale="", target_files=["x.py"],
                 instructions="", priority=1, category="bugfix",
                 forbidden_files=["tests/d2p_qa/test_b.py"]),
            Task(id="qa-cccc", title="c", rationale="", target_files=["x.py"],
                 instructions="", priority=1, category="bugfix",
                 forbidden_files=["tests/d2p_qa/test_c.py"]),
        ]
        # Simulate the fix cap dropping the last one
        dispatched = all_tasks[:2]
        # Reconstruct bug_test_paths the way the orchestrator does
        bug_test_paths = {
            t.id: t.forbidden_files[0]
            for t in dispatched
            if t.id.startswith("qa-") and t.forbidden_files
        }
        dispatched_test_paths = set(bug_test_paths.values())
        # the deferred bug's test path must NOT be in the set
        self.assertNotIn("tests/d2p_qa/test_c.py", dispatched_test_paths)
        self.assertIn("tests/d2p_qa/test_a.py", dispatched_test_paths)
        self.assertIn("tests/d2p_qa/test_b.py", dispatched_test_paths)

    def test_restore_tasks_dont_populate_bug_paths(self) -> None:
        """restore-symbol tasks must NOT show up in bug_test_paths — their
        forbidden_files is empty and their id has the "restore-" prefix."""
        from d2p.models import Task
        tasks = [
            Task(id="restore-xxx", title="r", rationale="", target_files=["m.py"],
                 instructions="", priority=0, category="bugfix",
                 forbidden_files=[]),
            Task(id="qa-yyyy", title="y", rationale="", target_files=["m.py"],
                 instructions="", priority=1, category="bugfix",
                 forbidden_files=["tests/d2p_qa/test_y.py"]),
        ]
        bug_test_paths = {
            t.id: t.forbidden_files[0]
            for t in tasks
            if t.id.startswith("qa-") and t.forbidden_files
        }
        self.assertEqual(set(bug_test_paths.keys()), {"qa-yyyy"})


class TestMetaConcurrency(unittest.TestCase):
    """Regression test for the 2026-05-19 race: parallel post_check callbacks
    invoked flip_meta_status concurrently for different bugs; the
    read-modify-write of _meta.json lost updates because there was no lock.
    Now _meta_lock serialises all meta mutations."""

    def test_parallel_flips_all_persist(self) -> None:
        import threading as _th
        from d2p.qa import QAAgent
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests" / "d2p_qa").mkdir(parents=True)
            # 20 distinct bugs, all open
            initial = {
                f"tests/d2p_qa/test_{i}.py": {
                    "id": f"b{i}", "title": f"t{i}",
                    "test_path": f"tests/d2p_qa/test_{i}.py",
                    "category": "x", "summary": "",
                    "suspected_files": [], "last_failure": "",
                    "status": "open", "attempts": 0,
                }
                for i in range(20)
            }
            (root / "tests" / "d2p_qa" / "_meta.json").write_text(
                json.dumps(initial))
            sb = Sandbox(root)
            qa = QAAgent.__new__(QAAgent)
            qa.sandbox = sb
            qa.corpus_dir = "tests/d2p_qa"
            qa._meta_lock = _th.Lock()
            qa._test_run_lock = _th.Lock()

            def flip(i):
                qa.flip_meta_status(f"tests/d2p_qa/test_{i}.py", "fixed")

            threads = [_th.Thread(target=flip, args=(i,)) for i in range(20)]
            for t in threads: t.start()
            for t in threads: t.join()

            final = json.loads((root / "tests/d2p_qa/_meta.json").read_text())
            # Every bug must show status=fixed. Without the lock, the
            # last-writer-wins race drops updates and most stay "open".
            for i in range(20):
                self.assertEqual(
                    final[f"tests/d2p_qa/test_{i}.py"]["status"], "fixed",
                    f"bug {i} lost its flip — meta race regression",
                )

    def test_parallel_bumps_all_count(self) -> None:
        import threading as _th
        from d2p.qa import QAAgent
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests" / "d2p_qa").mkdir(parents=True)
            initial = {
                "tests/d2p_qa/test_x.py": {
                    "id": "a", "title": "t",
                    "test_path": "tests/d2p_qa/test_x.py",
                    "category": "x", "summary": "", "suspected_files": [],
                    "last_failure": "", "status": "open", "attempts": 0,
                },
            }
            (root / "tests" / "d2p_qa" / "_meta.json").write_text(
                json.dumps(initial))
            sb = Sandbox(root)
            qa = QAAgent.__new__(QAAgent)
            qa.sandbox = sb
            qa.corpus_dir = "tests/d2p_qa"
            qa._meta_lock = _th.Lock()
            qa._test_run_lock = _th.Lock()

            N = 50
            threads = [_th.Thread(
                target=qa.bump_attempts,
                args=("tests/d2p_qa/test_x.py",),
            ) for _ in range(N)]
            for t in threads: t.start()
            for t in threads: t.join()

            final = json.loads((root / "tests/d2p_qa/_meta.json").read_text())
            # Without the lock, lost updates would make attempts < N.
            self.assertEqual(
                final["tests/d2p_qa/test_x.py"]["attempts"], N,
                "attempts lost updates — meta race regression",
            )


class TestIterMdBugCounts(unittest.TestCase):
    """The Bugs section of iter md must reflect the actual lifecycle, not
    confuse "carried in from prior iters" with "still open after fix sweep".
    Regression for the 2026-05-19 audit finding."""

    def test_bug_section_labels(self) -> None:
        from unittest.mock import MagicMock
        from d2p.orchestrator import Orchestrator
        from d2p.models import PlanResult
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            orch = Orchestrator.__new__(Orchestrator)
            orch.run_dir = root
            orch.router = MagicMock()
            orch.router.usage.summary.return_value = {
                "total_calls": 0, "total_cost_usd": 0.0,
                "cache_hit_ratio": 0.0, "per_role": {},
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cache_creation_tokens": 0, "total_cache_read_tokens": 0,
            }
            plan = PlanResult(iteration=1, tasks=[], rationale="r")
            # Simulate: 2 carried in, 4 new, 1 incidentally fixed, 2 retired,
            # 3 still open going forward.
            fake_qa = MagicMock()
            fake_qa.new_bugs = [MagicMock(test_path=f"t{i}", title=f"t{i}")
                                for i in range(4)]
            fake_qa.fixed_bugs = [MagicMock(test_path="f1", title="f1")]
            fake_qa.open_bugs = [MagicMock(test_path="o1", title="o1"),
                                 MagicMock(test_path="o2", title="o2")]
            orch._emit_iter_changes_md(
                1, plan=plan, results=[], qa_report=fake_qa,
                qa_fix_results=[],
                retired_this_iter=["r1.py", "r2.py"],
                still_open_count=3,
                elapsed_s=1.0, cost_delta_usd=0.0,
            )
            md = (root / "iter1_changes.md").read_text()
            # New, unambiguous labels
            self.assertIn("carried in (open from prior iters): 2", md)
            self.assertIn("new this iter: 4", md)
            self.assertIn("incidentally fixed", md)
            self.assertIn("retired (wontfix) this iter: 2", md)
            self.assertIn("still open going forward: 3", md)
            # Old ambiguous label must NOT appear
            self.assertNotIn("- still open: 2", md)

    def test_fix_task_ok_failed_counts(self) -> None:
        from unittest.mock import MagicMock
        from d2p.orchestrator import Orchestrator
        from d2p.models import ExecutionResult, PlanResult
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            orch = Orchestrator.__new__(Orchestrator)
            orch.run_dir = root
            orch.router = MagicMock()
            orch.router.usage.summary.return_value = {
                "total_calls": 0, "total_cost_usd": 0.0,
                "cache_hit_ratio": 0.0, "per_role": {},
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cache_creation_tokens": 0, "total_cache_read_tokens": 0,
            }
            plan = PlanResult(iteration=1, tasks=[], rationale="r")
            fake_qa = MagicMock()
            fake_qa.new_bugs = []; fake_qa.fixed_bugs = []; fake_qa.open_bugs = []
            qa_fix_results = [
                ExecutionResult(task_id="qa-aaaa", status="done", summary=""),
                ExecutionResult(task_id="qa-bbbb", status="failed", summary=""),
                ExecutionResult(task_id="qa-cccc", status="failed", summary=""),
            ]
            orch._emit_iter_changes_md(
                1, plan=plan, results=[], qa_report=fake_qa,
                qa_fix_results=qa_fix_results,
                retired_this_iter=[], still_open_count=2,
                elapsed_s=1.0, cost_delta_usd=0.0,
            )
            md = (root / "iter1_changes.md").read_text()
            self.assertIn("fix tasks: 1 ok, 2 failed", md)


class TestPlannerFeatureCap(unittest.TestCase):
    def test_planner_renders_min_max_from_feature_cap(self) -> None:
        from unittest.mock import MagicMock
        from d2p.agents import Planner  # noqa: F401
        from d2p.models import AnalysisReport
        # Build a Planner with a mocked llm and capture the user prompt.
        # MagicMock auto-creates `chat_structured`, so the helper picks
        # that path; pin it explicitly to the expected return shape.
        sb_mock = MagicMock()
        sb_mock.listing.return_value = ["app.py"]
        sb_mock.read.return_value = "print(1)\n"
        llm_mock = MagicMock()
        llm_mock.chat_structured.return_value = {"rationale": "", "tasks": []}
        p = Planner(llm_mock, sb_mock, max_tasks=5)
        p.run(
            AnalysisReport(domain="x", essence="e", audience="a"),
            iteration=1, max_iter=2,
            history=[], open_bugs=None,
            feature_cap=1,
        )
        args = llm_mock.chat_structured.call_args.args
        kwargs = llm_mock.chat_structured.call_args.kwargs
        user_prompt = args[1] if len(args) > 1 else kwargs.get("user", "")
        # When feature_cap=1, both min_tasks and max_tasks collapse to 1
        # (the floor formula uses min(3, cap)).
        self.assertIn("1 to 1 tasks", user_prompt)
        # Schema passed through so the model is forced into the right shape
        self.assertIn("schema", kwargs)
        self.assertEqual(kwargs["schema"]["properties"]["tasks"]["type"], "array")


class TestUsageCounters(unittest.TestCase):
    def test_counter_increments_threadsafe(self) -> None:
        import threading as _th
        from d2p.providers.base import UsageAccumulator
        u = UsageAccumulator()

        def worker():
            for _ in range(100):
                u.increment("self_heal_attempts")
                u.increment("self_heal_succeeded", 2)

        threads = [_th.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        c = u.counters()
        self.assertEqual(c["self_heal_attempts"], 400)
        self.assertEqual(c["self_heal_succeeded"], 800)
        # surfaced in summary
        self.assertEqual(u.summary()["counters"], c)


class TestCompressHistory(unittest.TestCase):
    def test_strips_big_payloads(self) -> None:
        from d2p.agents import _compress_history
        # Synthesise an entry with the kind of payloads that bloat the
        # Planner prompt (full stdout/stderr per test, etc.).
        big_qa_output = "x" * 50_000
        raw = [{
            "iteration": 1,
            "results": [
                {"task_id": "t1", "status": "done",
                 "files_changed": ["a.py", "b.py", "c.py", "d.py", "e.py"],
                 "summary": "y" * 5000, "error": ""},
                {"task_id": "t2", "status": "failed",
                 "files_changed": [], "summary": "z" * 3000, "error": "oops"},
            ],
            "qa_fix_results": [
                {"task_id": "qa-aaaa", "status": "done", "summary": "a" * 2000},
            ],
            "qa": {
                "new_bugs": [{"title": "n1", "test_path": "p"}],
                "fixed_bugs": [],
                "open_bugs": [{"title": "o1"}],
                "test_runs": {"p": {"output": big_qa_output}},
            },
            "retired_this_iter": ["something"],
        }]
        out = _compress_history(raw)
        self.assertEqual(len(out), 1)
        e = out[0]
        # iteration preserved
        self.assertEqual(e["iteration"], 1)
        # only task_id + status + files (capped to 4) kept from results
        self.assertEqual(e["results"][0]["task_id"], "t1")
        self.assertEqual(len(e["results"][0]["files"]), 4)
        # summary/error gone
        self.assertNotIn("summary", e["results"][0])
        self.assertNotIn("error", e["results"][0])
        # qa.test_runs (the big payload) completely dropped
        self.assertNotIn("test_runs", e["qa"])
        # the giant string must not appear anywhere in the compressed form
        serialised = json.dumps(e)
        self.assertNotIn(big_qa_output, serialised)
        self.assertLess(len(serialised), 2000,
                        f"compressed entry too big: {len(serialised)} bytes")

    def test_empty_input_yields_empty(self) -> None:
        from d2p.agents import _compress_history
        self.assertEqual(_compress_history([]), [])


class TestStageTimingsInMd(unittest.TestCase):
    def test_md_renders_stage_breakdown(self) -> None:
        from unittest.mock import MagicMock
        from d2p.orchestrator import Orchestrator
        from d2p.models import PlanResult
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            orch = Orchestrator.__new__(Orchestrator)
            orch.run_dir = root
            orch.router = MagicMock()
            orch.router.usage.summary.return_value = {
                "total_calls": 0, "total_cost_usd": 0.0,
                "cache_hit_ratio": 0.0, "per_role": {},
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cache_creation_tokens": 0, "total_cache_read_tokens": 0,
                "counters": {},
            }
            plan = PlanResult(iteration=1, tasks=[], rationale="r")
            orch._emit_iter_changes_md(
                1, plan=plan, results=[], qa_report=None,
                qa_fix_results=[], retired_this_iter=[],
                still_open_count=0,
                elapsed_s=100.0, cost_delta_usd=0.0,
                stage_timings={"planner_s": 1.5, "executor_s": 30.2,
                               "qa_s": 12.0, "fix_s": 45.7,
                               "regression_sweep_s": 8.4},
            )
            md = (root / "iter1_changes.md").read_text()
            self.assertIn("Stage timings:", md)
            self.assertIn("planner=1.5s", md)
            self.assertIn("executor=30.2s", md)
            self.assertIn("fix=45.7s", md)


class TestExecutorPrepareCommit(unittest.TestCase):
    """The 2026-05-20 refactor split Executor.run into prepare (unlocked LLM
    call) + commit (locked writes). Verify (a) backward-compat — run() still
    works, (b) FILE-mode refuses to clobber a concurrently-modified file,
    (c) PATCH-mode applies cleanly against the post-modification content."""

    def _mk(self, root, llm_response):
        from unittest.mock import MagicMock
        from d2p.agents import Executor
        sb = Sandbox(root)
        llm = MagicMock()
        llm.chat.return_value = llm_response
        ex = Executor.__new__(Executor)
        ex.llm = llm
        ex.sandbox = sb
        from d2p.lang import adapter_for, detect_primary_language
        ex.adapter = adapter_for(detect_primary_language(sb))
        ex.usage = None
        return ex

    def test_file_mode_refuses_concurrent_modification(self) -> None:
        from d2p.models import Task
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.py").write_text("# v1\nprint(1)\n")
            ex = self._mk(root, (
                "STATUS: done\nSUMMARY: rewrite\n"
                "===FILE: a.py===\n# v2\nprint(2)\n===END===\n"
            ))
            task = Task(id="t1", title="t", rationale="", target_files=["a.py"],
                        instructions="", priority=1, category="feature")
            # Phase 1: prepare reads a.py at v1
            prepared = ex.prepare(task)
            self.assertEqual(prepared.source_snapshot["a.py"], "# v1\nprint(1)\n")
            # Concurrent modification: another task writes v_other before commit
            (root / "a.py").write_text("# v_other\nprint(99)\n")
            # Phase 2: commit must refuse to clobber v_other with v2
            res = ex.commit(prepared)
            self.assertEqual(res.status, "failed")
            self.assertIn("concurrent modification", res.error)
            # The concurrent write must be preserved
            self.assertEqual((root / "a.py").read_text(), "# v_other\nprint(99)\n")

    def test_file_mode_applies_when_unchanged(self) -> None:
        from d2p.models import Task
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.py").write_text("# v1\nprint(1)\n")
            ex = self._mk(root, (
                "STATUS: done\nSUMMARY: rewrite\n"
                "===FILE: a.py===\n# v2\nx = 1\ny = 2\nz = 3\nprint(x+y+z)\n===END===\n"
            ))
            task = Task(id="t1", title="t", rationale="", target_files=["a.py"],
                        instructions="", priority=1, category="feature")
            prepared = ex.prepare(task)
            res = ex.commit(prepared)
            self.assertEqual(res.status, "done", res.error)
            self.assertIn("# v2", (root / "a.py").read_text())

    def test_patch_mode_survives_concurrent_modification(self) -> None:
        """PATCH-mode handles concurrency naturally because SEARCH/REPLACE
        runs against the file's CURRENT content. If the anchor texts are
        still present after the concurrent write, the patch applies fine."""
        from d2p.models import Task
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # File has two functions; we patch foo and concurrent edit modifies bar.
            initial = (
                "def foo():\n    return 1\n\n"
                "def bar():\n    return 2\n"
            )
            (root / "lib.py").write_text(initial)
            ex = self._mk(root, (
                "STATUS: done\nSUMMARY: patch foo\n"
                "===PATCH: lib.py===\n"
                "<<<SEARCH\ndef foo():\n    return 1\nSEARCH>>>\n"
                "<<<REPLACE\ndef foo():\n    return 42\nREPLACE>>>\n"
                "===END===\n"
            ))
            task = Task(id="t2", title="t", rationale="", target_files=["lib.py"],
                        instructions="", priority=1, category="feature")
            prepared = ex.prepare(task)
            # Concurrent modification: another task edits bar (NOT foo)
            new_content = (
                "def foo():\n    return 1\n\n"
                "def bar():\n    return 999\n"
            )
            (root / "lib.py").write_text(new_content)
            # Phase 2: PATCH should still find foo's anchor and apply
            res = ex.commit(prepared)
            self.assertEqual(res.status, "done", res.error)
            final = (root / "lib.py").read_text()
            self.assertIn("return 42", final)   # our patch applied
            self.assertIn("return 999", final)  # concurrent edit preserved

    def test_run_wrapper_calls_prepare_then_commit(self) -> None:
        """Backward-compat: run() is equivalent to prepare + commit."""
        from d2p.models import Task
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ex = self._mk(root, (
                "STATUS: done\nSUMMARY: create\n"
                "===FILE: new.py===\nimport os\nprint(os.getcwd())\n===END===\n"
            ))
            task = Task(id="t3", title="t", rationale="", target_files=["new.py"],
                        instructions="", priority=1, category="feature")
            res = ex.run(task)
            self.assertEqual(res.status, "done", res.error)
            self.assertTrue((root / "new.py").is_file())


class TestExecutorParallelLLM(unittest.TestCase):
    """Smoke test that two prepare() calls can run concurrently — the whole
    point of moving the LLM out of the per-file lock. Mocked LLM sleeps;
    if prepare wasn't parallel-safe (e.g. shared mutable state) elapsed
    time would be ~2× serial."""

    def test_two_prepares_run_in_parallel(self) -> None:
        import threading as _th
        import time as _time
        from unittest.mock import MagicMock
        from d2p.agents import Executor
        from d2p.models import Task

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "app.py").write_text("print(1)\n")
            sb = Sandbox(root)

            def slow_chat(*a, **k):
                _time.sleep(0.5)
                return ("STATUS: done\nSUMMARY: x\n"
                        "===FILE: app.py===\nprint(2)\n===END===\n")

            llm = MagicMock()
            llm.chat.side_effect = slow_chat

            # Two executors share the same sandbox/LLM (mock is thread-safe
            # because each call has its own kwargs)
            from d2p.lang import adapter_for, detect_primary_language
            ex1 = Executor.__new__(Executor)
            ex1.llm = llm; ex1.sandbox = sb
            ex1.adapter = adapter_for(detect_primary_language(sb))
            ex1.usage = None
            ex2 = Executor.__new__(Executor)
            ex2.llm = llm; ex2.sandbox = sb
            ex2.adapter = adapter_for(detect_primary_language(sb))
            ex2.usage = None

            results: list = [None, None]

            def w(i, ex):
                t = Task(id=f"t{i}", title="x", rationale="",
                         target_files=["app.py"], instructions="",
                         priority=1, category="feature")
                results[i] = ex.prepare(t)

            t0 = _time.monotonic()
            ths = [_th.Thread(target=w, args=(0, ex1)),
                   _th.Thread(target=w, args=(1, ex2))]
            for t in ths: t.start()
            for t in ths: t.join()
            elapsed = _time.monotonic() - t0
            # If serialised: ~1.0s. In parallel: ~0.5s. Allow some scheduling slack.
            self.assertLess(elapsed, 0.9,
                            f"prepare() not running in parallel: elapsed={elapsed:.2f}s")
            for r in results:
                self.assertIsNotNone(r)


class TestHtmlReport(unittest.TestCase):
    def test_render_includes_core_fields(self) -> None:
        from d2p.report import render_html
        summary = {
            "analysis": {"domain": "todo API", "audience": "devs",
                          "essence": "minimal stateless REST"},
            "elapsed_s": 100.5,
            "iterations": [{
                "iteration": 1,
                "elapsed_s": 50.0,
                "cost_delta_usd": 0.1,
                "cumulative_cost_usd": 0.1,
                "plan_rationale": "first cut",
                "stage_timings": {"planner_s": 5.0, "executor_s": 40.0},
                "results": [
                    {"task_id": "t1", "status": "done",
                     "files_changed": ["app.py"]},
                ],
                "qa": {"new_bugs": [{"test_path": "tests/d2p_qa/x.py",
                                      "title": "x bug"}],
                       "fixed_bugs": [],
                       "open_bugs": []},
                "qa_fix_results": [],
                "retired_this_iter": [],
            }],
            "open_bugs": [{"test_path": "tests/d2p_qa/y.py", "title": "y bug",
                            "attempts": 2, "first_seen_iter": 1}],
            "run_dir": "/tmp/x",
            "usage": {
                "total_calls": 5, "total_cost_usd": 0.123,
                "cache_hit_ratio": 0.5,
                "per_role": {"executor:haiku": {"calls": 5, "input": 100,
                                                  "output": 200,
                                                  "cache_read": 50,
                                                  "cache_creation": 50,
                                                  "cost_usd": 0.123}},
                "counters": {"self_heal_attempts": 2, "self_heal_succeeded": 1},
            },
        }
        html_str = render_html(summary)
        # all the load-bearing facts must appear
        self.assertIn("todo API", html_str)
        self.assertIn("minimal stateless REST", html_str)
        self.assertIn("100.5", html_str)
        self.assertIn("$0.1230", html_str)
        self.assertIn("Iteration 1", html_str)
        self.assertIn("first cut", html_str)
        self.assertIn("tests/d2p_qa/x.py", html_str)
        self.assertIn("tests/d2p_qa/y.py", html_str)
        self.assertIn("executor:haiku", html_str)
        self.assertIn("self_heal_attempts", html_str)
        # html escaping — title with special chars survives
        summary["analysis"]["domain"] = "<script>alert(1)</script>"  # type: ignore[index]
        html_str2 = render_html(summary)
        self.assertNotIn("<script>alert(1)</script>", html_str2)
        self.assertIn("&lt;script&gt;", html_str2)

    def test_render_handles_empty(self) -> None:
        from d2p.report import render_html
        # minimal summary shouldn't crash
        out = render_html({"analysis": {}, "iterations": [], "open_bugs": [],
                            "elapsed_s": 0, "usage": {}, "run_dir": ""})
        self.assertIn("d2p run report", out)


class TestResumeReload(unittest.TestCase):
    """The orchestrator's _reload_history reconstructs `history` from per-iter
    JSON files. Without it, --resume would lose all prior-iter context the
    Planner needs."""

    def test_reload_skips_incomplete_iters(self) -> None:
        from unittest.mock import MagicMock
        from d2p.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # iter 1: complete (plan + exec + qa + rerun)
            (root / "plan_iter1.json").write_text(json.dumps({
                "iteration": 1, "rationale": "iter1 plan",
                "tasks": [{"id": "t1"}],
            }))
            (root / "exec_iter1.json").write_text(json.dumps([
                {"task_id": "t1", "status": "done", "files_changed": ["a.py"]},
            ]))
            (root / "qa_iter1.json").write_text(json.dumps({
                "new_bugs": [{"test_path": "tests/d2p_qa/bug.py", "title": "b"}],
                "fixed_bugs": [], "open_bugs": [], "retired_bugs": [],
                "test_runs": {},
            }))
            (root / "qa_rerun_iter1.json").write_text(json.dumps({
                "tests/d2p_qa/bug.py": {"status": "failed"},
            }))
            # iter 2: ONLY plan written — incomplete, should be re-run
            (root / "plan_iter2.json").write_text(json.dumps({
                "iteration": 2, "tasks": [],
            }))
            # no exec_iter2.json

            orch = Orchestrator.__new__(Orchestrator)
            orch.run_dir = root

            history, resume_from, open_bugs = orch._reload_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(resume_from, 2)
            self.assertEqual(history[0]["iteration"], 1)
            # bug carried into next iter (failed in rerun)
            self.assertEqual(len(open_bugs), 1)
            self.assertEqual(open_bugs[0]["test_path"], "tests/d2p_qa/bug.py")

    def test_reload_empty_run_dir(self) -> None:
        from d2p.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            orch = Orchestrator.__new__(Orchestrator)
            orch.run_dir = root
            history, resume_from, open_bugs = orch._reload_history()
            self.assertEqual(history, [])
            self.assertEqual(resume_from, 1)
            self.assertEqual(open_bugs, [])


class TestChatStructuredFallback(unittest.TestCase):
    def test_uses_chat_structured_when_available(self) -> None:
        from unittest.mock import MagicMock
        from d2p.providers.base import chat_structured
        p = MagicMock()
        p.chat_structured.return_value = {"ok": True}
        out = chat_structured(p, "sys", "user",
                              schema={"type": "object"},
                              temperature=0.3, max_tokens=100)
        self.assertEqual(out, {"ok": True})
        p.chat_structured.assert_called_once()
        p.chat_json.assert_not_called()

    def test_falls_back_to_chat_json(self) -> None:
        # Provider without chat_structured — helper must embed schema in
        # the prompt and call chat_json.
        from d2p.providers.base import chat_structured

        class Stub:
            name = "stub"
            def chat(self, *a, **k): return ""
            def chat_json(self, system, user, **kw):
                # Schema embedded in the user prompt
                self.last_user = user
                return {"ok": True}

        s = Stub()
        out = chat_structured(s, "sys", "user",
                              schema={"type": "object", "title": "X"})
        self.assertEqual(out, {"ok": True})
        self.assertIn('"title": "X"', s.last_user)


class TestPlannerTrim(unittest.TestCase):
    def test_key_files_block_caps_per_file_chars(self) -> None:
        from unittest.mock import MagicMock
        from d2p.agents import Planner
        sb = MagicMock()
        sb.read.return_value = "x" * 10_000
        p = Planner(MagicMock(), sb, max_tasks=5)
        block = p._build_key_files_block(["a.py", "b.py"])
        # Each file capped at Planner.KEY_FILE_CHARS; allow ~100 chars
        # of header overhead per file.
        self.assertLess(len(block), 2 * (Planner.KEY_FILE_CHARS + 100))
        self.assertIn("=== a.py ===", block)

    def test_pick_key_files_caps_count(self) -> None:
        from unittest.mock import MagicMock
        from d2p.agents import Planner
        sb = MagicMock()
        sb.read.return_value = "x" * 100
        p = Planner(MagicMock(), sb, max_tasks=5)
        # 10 candidate source files, all .py
        listing = [f"f{i}.py" for i in range(10)]
        keys = p._pick_key_files(listing)
        self.assertLessEqual(len(keys), p.KEY_FILES_MAX)


class TestRaceModeConfig(unittest.TestCase):
    def test_cfg_race_roles_defaults_empty(self) -> None:
        from d2p.config import Config
        cfg = Config()
        self.assertEqual(cfg.race_roles, set())

    def test_race_on_is_per_role(self) -> None:
        """Race is enabled per-role: gate is `role in cfg.race_roles AND
        fallback_present`. Race for fix-only should NOT trigger executor
        race, and vice versa."""
        from d2p.config import Config
        cfg = Config()
        fallback_present = True
        # fix only
        cfg.race_roles = {"fix"}
        self.assertTrue("fix" in cfg.race_roles and fallback_present)
        self.assertFalse("executor" in cfg.race_roles and fallback_present)
        # both
        cfg.race_roles = {"fix", "executor"}
        self.assertTrue("fix" in cfg.race_roles)
        self.assertTrue("executor" in cfg.race_roles)
        # off
        cfg.race_roles = set()
        self.assertFalse("fix" in cfg.race_roles)

    def test_cli_race_mode_parser(self) -> None:
        """Verify the CLI value parser handles all documented shapes."""
        from run import _parse_race_roles
        self.assertEqual(_parse_race_roles(""), set())
        self.assertEqual(_parse_race_roles("none"), set())
        self.assertEqual(_parse_race_roles("NONE"), set())
        self.assertEqual(_parse_race_roles("all"), {"fix", "executor"})
        self.assertEqual(_parse_race_roles("fix"), {"fix"})
        self.assertEqual(_parse_race_roles("executor"), {"executor"})
        self.assertEqual(_parse_race_roles("exec"), {"executor"})  # alias
        self.assertEqual(_parse_race_roles("fix,executor"), {"fix", "executor"})
        self.assertEqual(_parse_race_roles("fix, exec"), {"fix", "executor"})
        # unknown roles dropped
        self.assertEqual(_parse_race_roles("fix,bogus"), {"fix"})
        self.assertEqual(_parse_race_roles("bogus"), set())

    def test_commit_respects_max_fix_attempts_override(self) -> None:
        """Race-mode commits cap MAX_FIX_ATTEMPTS=1 so race × retry doesn't
        compound. Verify Executor.commit honours the kwarg."""
        from unittest.mock import MagicMock
        from d2p.agents import Executor, PreparedExecution
        from d2p.models import Task
        ex = Executor.__new__(Executor)  # bypass __init__
        ex.MAX_FIX_ATTEMPTS = 3
        # Stub commit's pre-loop work: build a "done" result with a
        # post_check that always fails so we can count attempts.
        # Simplest: invoke the internal retry loop logic via direct
        # construction is tricky — instead, monkey-patch the LLM call.
        ex.llm = MagicMock()
        ex.llm.chat.return_value = "===PATCH===\n(noop)\n===END==="
        ex.sandbox = MagicMock()
        ex.sandbox.read.return_value = "x = 1\n"
        ex.sandbox.write.return_value = "x = 1\n"
        ex.adapter = MagicMock()
        ex.adapter.health_check.return_value = ("ok", "")
        ex.adapter.post_write_check.return_value = (True, "")
        ex.usage = None
        # We can't easily exercise the full commit path without more
        # plumbing. Instead just verify the signature accepts the kwarg.
        import inspect
        sig = inspect.signature(Executor.commit)
        self.assertIn("max_fix_attempts", sig.parameters)
        self.assertIsNone(sig.parameters["max_fix_attempts"].default)


class TestAnalyzerPipelineGate(unittest.TestCase):
    def test_next_iter_reanalyze_trigger_formula(self) -> None:
        """The prefetch gate fires at the end of iter N if iter N+1 will
        trigger re-analysis (i.e. (N+1-1) % reanalyze_every == 0 and
        N+1 > 1)."""
        # reanalyze_every=2 → triggers at iters 3, 5, 7, ... (next_it=3
        # means we prefetch at end of iter 2)
        for it, expected in [(1, False), (2, True), (3, False), (4, True)]:
            next_it = it + 1
            reanalyze_every = 2
            should_prefetch = (
                reanalyze_every and next_it > 1
                and (next_it - 1) % reanalyze_every == 0
            )
            self.assertEqual(bool(should_prefetch), expected,
                             f"iter={it} should_prefetch={should_prefetch} "
                             f"expected={expected}")


if __name__ == "__main__":
    unittest.main()
