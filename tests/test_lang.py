"""Unit tests for language detection + adapters — no LLM."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.fs import Sandbox
from d2p.lang.detect import detect_primary_language


class TestDetect(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sb = Sandbox(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pure_python(self) -> None:
        (self.root / "app.py").write_text("")
        (self.root / "lib.py").write_text("")
        (self.root / "main.py").write_text("")
        self.assertEqual(detect_primary_language(self.sb), "python")

    def test_pure_js(self) -> None:
        (self.root / "server.js").write_text("")
        (self.root / "lib.js").write_text("")
        (self.root / "package.json").write_text('{"name":"x"}')
        self.assertEqual(detect_primary_language(self.sb), "javascript")

    def test_typescript_over_js(self) -> None:
        (self.root / "a.ts").write_text("")
        (self.root / "b.ts").write_text("")
        (self.root / "c.js").write_text("")
        self.assertEqual(detect_primary_language(self.sb), "typescript")

    def test_python_wins_over_minor_js(self) -> None:
        for n in range(5):
            (self.root / f"file{n}.py").write_text("")
        (self.root / "build.js").write_text("")  # one supporting JS
        self.assertEqual(detect_primary_language(self.sb), "python")

    def test_empty_project(self) -> None:
        self.assertEqual(detect_primary_language(self.sb), "unknown")

    def test_go(self) -> None:
        (self.root / "main.go").write_text("")
        (self.root / "lib.go").write_text("")
        self.assertEqual(detect_primary_language(self.sb), "go")

    def test_adapter_selection_by_detected_language(self) -> None:
        from d2p.lang import adapter_for, PythonAdapter, JSAdapter, NullAdapter
        self.assertIsInstance(adapter_for("python"), PythonAdapter)
        self.assertIsInstance(adapter_for("javascript"), JSAdapter)
        self.assertIsInstance(adapter_for("typescript"), JSAdapter)
        self.assertIsInstance(adapter_for("go"), NullAdapter)  # not yet wired

    def test_python_adapter_syntax_check(self) -> None:
        from d2p.lang import PythonAdapter
        ad = PythonAdapter()
        (self.root / "ok.py").write_text("x = 1\n")
        (self.root / "bad.py").write_text("def f(:\n  pass\n")
        self.assertEqual(ad.syntax_check(self.sb, "ok.py"), "")
        self.assertIn("syntax error", ad.syntax_check(self.sb, "bad.py"))

    def test_python_adapter_includes_project_own_tests(self) -> None:
        from d2p.lang import PythonAdapter
        ad = PythonAdapter()
        (self.root / "app.py").write_text("x = 1\n")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "__init__.py").write_text("")
        (self.root / "tests" / "test_smoke.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n"
        )
        mods = ad.discover_modules(self.sb)
        self.assertIn("tests.test_smoke", mods)

    def test_js_adapter_test_path(self) -> None:
        from d2p.lang import JSAdapter
        ad = JSAdapter()
        self.assertTrue(ad.test_path("bug_x").endswith(".test.mjs"))
        self.assertIn("d2p_qa", ad.test_path("bug_x"))

    def test_python_runner_uses_pytest_if_test_imports_it(self) -> None:
        from d2p.lang import PythonAdapter
        ad = PythonAdapter()
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests" / "test_foo.py").write_text(
            "import pytest\ndef test_a(): assert True\n"
        )
        cmd = ad.test_runner_cmd("tests/test_foo.py", sandbox=self.sb)
        self.assertIn("-m", cmd)
        self.assertIn("pytest", cmd)

    def test_python_runner_uses_unittest_for_unittest_tests(self) -> None:
        from d2p.lang import PythonAdapter
        ad = PythonAdapter()
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests" / "test_bar.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n"
        )
        cmd = ad.test_runner_cmd("tests/test_bar.py", sandbox=self.sb)
        self.assertIn("unittest", cmd)

    def test_rust(self) -> None:
        (self.root / "Cargo.toml").write_text("")
        (self.root / "src").mkdir()
        (self.root / "src" / "main.rs").write_text("")
        (self.root / "src" / "lib.rs").write_text("")
        self.assertEqual(detect_primary_language(self.sb), "rust")
