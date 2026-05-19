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


if __name__ == "__main__":
    unittest.main()
