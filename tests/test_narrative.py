"""Unit tests for the per-iteration narrative builder. Pure logic, no LLM."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.narrative import (
    analyzer_line, planner_line, executor_line, qa_line,
    build_iter_narrative,
)


def _res(task_id, status, files=(), error=""):
    return SimpleNamespace(task_id=task_id, status=status,
                           files_changed=list(files), error=error, summary="")


def _bug(title, test_path):
    return SimpleNamespace(title=title, test_path=test_path)


def _analysis(domain="multi-agent sim", essence="agent-vs-agent harness",
              features=2, competitors=3):
    return SimpleNamespace(domain=domain, essence=essence,
                           features=list(range(features)),
                           competitors=list(range(competitors)))


def _plan(rationale="close competitor gaps", titles=("Add Dockerfile",)):
    tasks = [SimpleNamespace(id=f"t{i}", title=t) for i, t in enumerate(titles)]
    return SimpleNamespace(rationale=rationale, tasks=tasks)


class TestAnalyzerLine(unittest.TestCase):
    def test_includes_domain_and_essence(self):
        line = analyzer_line(_analysis(), reanalyzed=False)
        self.assertIn("multi-agent sim", line)
        self.assertIn("agent-vs-agent harness", line)
        self.assertNotIn("本轮重新分析", line)

    def test_reanalyzed_prefix(self):
        line = analyzer_line(_analysis(), reanalyzed=True)
        self.assertTrue(line.startswith("(本轮重新分析)"))


class TestPlannerLine(unittest.TestCase):
    def test_empty_tasks(self):
        self.assertIn("无新特性任务", planner_line(_plan(titles=())))

    def test_lists_titles_and_count(self):
        line = planner_line(_plan(titles=("A", "B")))
        self.assertIn("2 个任务", line)
        self.assertIn("A", line)
        self.assertIn("B", line)

    def test_truncates_to_three_titles(self):
        line = planner_line(_plan(titles=("A", "B", "C", "D")))
        self.assertIn("4 个任务", line)
        self.assertIn("…", line)
        self.assertNotIn("D", line)


class TestExecutorLine(unittest.TestCase):
    def test_no_results(self):
        self.assertIn("未执行", executor_line([], {}))

    def test_done_and_failed(self):
        results = [
            _res("t0", "done", files=["Dockerfile", ".github/workflows/ci.yml"]),
            _res("t1", "failed", error="SEARCH block did not match"),
        ]
        title_by_id = {"t0": "Add Dockerfile", "t1": "Login retry"}
        line = executor_line(results, title_by_id)
        self.assertIn("完成 1/2", line)
        self.assertIn("Dockerfile", line)
        self.assertIn("失败", line)
        self.assertIn("Login retry", line)
        self.assertIn("SEARCH block", line)

    def test_truncates_file_list(self):
        results = [_res("t0", "done",
                        files=[f"f{i}.py" for i in range(6)])]
        line = executor_line(results, {})
        self.assertIn("等 6 个文件", line)


class TestQaLine(unittest.TestCase):
    def test_none_report(self):
        self.assertEqual(qa_line(None, [], still_open=0), "本轮未跑 QA")

    def test_still_open_emits_marker(self):
        report = SimpleNamespace(new_bugs=[], open_bugs=[], fixed_bugs=[])
        line = qa_line(report, [], still_open=3)
        self.assertIn("未解决", line)

    def test_all_clear_no_marker(self):
        report = SimpleNamespace(new_bugs=[], open_bugs=[], fixed_bugs=[])
        line = qa_line(report, [], still_open=0)
        self.assertNotIn("未解决", line)
        self.assertIn("全部清零", line)

    def test_lists_new_bug_titles(self):
        report = SimpleNamespace(
            new_bugs=[_bug("login crashes on empty pw", "tests/d2p_qa/test_login.py")],
            open_bugs=[], fixed_bugs=[])
        line = qa_line(report, [], still_open=1)
        self.assertIn("login crashes", line)
        self.assertIn("test_login.py", line)


class TestBuildIterNarrative(unittest.TestCase):
    def test_returns_four_keys(self):
        out = build_iter_narrative(
            analysis=_analysis(), plan=_plan(),
            results=[_res("t0", "done", files=["README.md"])],
            qa_report=None, qa_fix_results=[], still_open_count=0,
            reanalyzed=False)
        self.assertEqual(set(out.keys()),
                         {"analyzer_summary", "planner_summary",
                          "executor_summary", "qa_summary"})
        self.assertTrue(all(isinstance(v, str) for v in out.values()))


if __name__ == "__main__":
    unittest.main()
