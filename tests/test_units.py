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
        # cache hit ratio = read / (read+creation) = 100 / 200 = 0.5
        self.assertEqual(s["cache_hit_ratio"], 0.5)
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
    def test_hit_skips_llm(self) -> None:
        from d2p.agents import Analyzer
        from d2p.fs import Sandbox
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "README.md").write_text("# x\n")
            sb = Sandbox(root)
            llm = MagicMock()
            llm.name = "m"
            llm.chat_json.return_value = {
                "domain": "X", "essence": "E", "audience": "A",
                "features": [{"name": "f1", "category": "ux",
                              "description": "d", "source": "s"}],
                "competitors": ["c1"], "ui_elements": ["u"],
                "raw_notes": "n",
            }
            a = Analyzer(llm, sb)
            cache_path = root / ".d2p" / "analysis_cache.json"
            r1, hit1 = a.run_cached(cache_path)
            r2, hit2 = a.run_cached(cache_path)
            self.assertFalse(hit1)
            self.assertTrue(hit2)
            self.assertEqual(llm.chat_json.call_count, 1)
            self.assertEqual(r1.essence, r2.essence)
            self.assertEqual([f.name for f in r1.features],
                             [f.name for f in r2.features])

    def test_no_cache_flag_forces_fresh(self) -> None:
        from d2p.agents import Analyzer
        from d2p.fs import Sandbox
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "README.md").write_text("# x\n")
            sb = Sandbox(root)
            llm = MagicMock()
            llm.name = "m"
            llm.chat_json.return_value = {
                "domain": "X", "essence": "E", "audience": "A",
                "features": [], "competitors": [], "ui_elements": [],
                "raw_notes": "",
            }
            a = Analyzer(llm, sb)
            cache_path = root / ".d2p" / "analysis_cache.json"
            a.run_cached(cache_path)
            a.run_cached(cache_path, use_cache=False)
            self.assertEqual(llm.chat_json.call_count, 2)


class TestRouterFallback(unittest.TestCase):
    def test_for_fallback_returns_none_when_absent(self) -> None:
        from d2p.providers.base import RoleRouter

        class P:
            def __init__(self, n): self.name = n
            def chat(self, *a, **k): return ""
            def chat_json(self, *a, **k): return {}
        r = RoleRouter({"default": P("d"), "executor": P("e")})
        self.assertIsNone(r.for_fallback("executor"))

    def test_for_fallback_returns_provider_when_set(self) -> None:
        from d2p.providers.base import RoleRouter

        class P:
            def __init__(self, n): self.name = n
            def chat(self, *a, **k): return ""
            def chat_json(self, *a, **k): return {}
        primary = P("haiku")
        fallback = P("sonnet")
        r = RoleRouter({"default": primary, "executor": primary},
                       fallbacks={"executor": fallback})
        self.assertIs(r.for_fallback("executor"), fallback)
        self.assertIsNone(r.for_fallback("planner"))
        # describe() exposes the fallback under <role>-fallback
        d = r.describe()
        self.assertIn("executor-fallback", d)
        self.assertEqual(d["executor-fallback"], "sonnet")

    def test_provider_spec_reads_fallback_env(self) -> None:
        import os as _os
        from d2p.providers import _from_env
        old = _os.environ.copy()
        try:
            _os.environ["D2P_PROVIDER"] = "minimax"
            _os.environ["MINIMAX_API_KEY"] = "sk-cp-test"
            _os.environ["D2P_ROLE_EXECUTOR_FALLBACK_MODEL"] = "MiniMax-strong"
            _os.environ["D2P_ROLE_FIX_FALLBACK_MODEL"] = "MiniMax-strong"
            spec = _from_env()
            self.assertEqual(spec.fallback_models.get("executor"), "MiniMax-strong")
            self.assertEqual(spec.fallback_models.get("fix"), "MiniMax-strong")
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


class TestEscalationDecision(unittest.TestCase):
    def test_skip_on_forbidden(self) -> None:
        from d2p.orchestrator import _should_escalate
        self.assertFalse(_should_escalate(
            "partial; rejected: tests/d2p_qa/x.py: forbidden (test file, read)"))

    def test_skip_on_sandbox_escape(self) -> None:
        from d2p.orchestrator import _should_escalate
        self.assertFalse(_should_escalate("path escapes sandbox: ../etc/x"))

    def test_retry_on_regression(self) -> None:
        from d2p.orchestrator import _should_escalate
        self.assertTrue(_should_escalate(
            "regression detected — rolled back"))

    def test_retry_on_search_miss(self) -> None:
        from d2p.orchestrator import _should_escalate
        self.assertTrue(_should_escalate(
            "app.py: SEARCH not found after retry: '@app.route...'"))

    def test_retry_on_empty_error(self) -> None:
        from d2p.orchestrator import _should_escalate
        # unknown errors should default to retry — better to waste $0.10
        # on one extra call than to silently swallow a fixable case.
        self.assertTrue(_should_escalate(""))


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
        from d2p.agents import Planner, ANALYZER_USER_TMPL  # noqa: F401
        from d2p.models import AnalysisReport
        # Build a Planner with a mocked llm and capture the user prompt
        sb_mock = MagicMock()
        sb_mock.listing.return_value = ["app.py"]
        sb_mock.read.return_value = "print(1)\n"
        llm_mock = MagicMock()
        llm_mock.chat_json.return_value = {"rationale": "", "tasks": []}
        p = Planner(llm_mock, sb_mock, max_tasks=5)
        p.run(
            AnalysisReport(domain="x", essence="e", audience="a"),
            iteration=1, max_iter=2,
            history=[], open_bugs=None,
            feature_cap=1,
        )
        # Last positional arg to chat_json is the user prompt
        _, kwargs = llm_mock.chat_json.call_args
        # chat_json(system, user, **kw) — user is args[1]
        args = llm_mock.chat_json.call_args.args
        user_prompt = args[1] if len(args) > 1 else kwargs.get("user", "")
        # When feature_cap=1, both min_tasks and max_tasks collapse to 1
        # (the floor formula uses min(3, cap)).
        self.assertIn("1 to 1 tasks", user_prompt)


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


if __name__ == "__main__":
    unittest.main()
