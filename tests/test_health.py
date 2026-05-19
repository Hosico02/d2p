"""Unit tests for ProjectHealth — no LLM, no network."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.fs import Sandbox
from d2p.health import ProjectHealth


class TestProjectHealth(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text("def hello():\n    return 1\n")
        (self.root / "helpers.py").write_text("import app\nVALUE = app.hello()\n")
        self.sb = Sandbox(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_clean_project_all_ok(self) -> None:
        h = ProjectHealth(self.sb)
        result = h.probe(["app", "helpers"])
        self.assertEqual(result["app"], "ok")
        self.assertEqual(result["helpers"], "ok")

    def test_broken_dependency_reports_error(self) -> None:
        (self.root / "app.py").write_text("def goodbye():\n    return 0\n")
        result = ProjectHealth(self.sb).probe(["app", "helpers"])
        self.assertEqual(result["app"], "ok")
        self.assertNotEqual(result["helpers"], "ok")
        self.assertIn("hello", result["helpers"])

    def test_syntax_error_in_module(self) -> None:
        (self.root / "app.py").write_text("def hello(:\n    pass\n")
        result = ProjectHealth(self.sb).probe(["app"])
        self.assertNotEqual(result["app"], "ok")

    def test_default_modules_includes_qa_corpus(self) -> None:
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests" / "__init__.py").write_text("")
        (self.root / "tests" / "d2p_qa").mkdir()
        (self.root / "tests" / "d2p_qa" / "__init__.py").write_text("")
        (self.root / "tests" / "d2p_qa" / "test_bug_x.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n"
        )
        h = ProjectHealth(self.sb)
        mods = h.default_modules()
        self.assertIn("tests.d2p_qa.test_bug_x", mods)
        result = h.probe(mods)
        self.assertEqual(result["tests.d2p_qa.test_bug_x"], "ok")
