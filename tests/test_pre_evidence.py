"""Direct tests for the extracted pre-evidence collector.
These tests do NOT depend on orchestrator.py — they verify that
`collect()` shells out to the right commands given a sandbox layout
and bundles the result into PreEvidence."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.agents.pre_evidence import collect
from d2p.agents.verifier import PreEvidence
from d2p.fs import Sandbox


class TestCollectPreEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(self.id().replace(".", "_"))
        self.tmpdir.mkdir(exist_ok=True)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_collect_returns_empty_pre_evidence_when_no_runners_present(self) -> None:
        sandbox = Sandbox(self.tmpdir)
        with mock.patch("d2p.agents.pre_evidence.shutil.which", return_value=None):
            evidence = collect(sandbox, iter_count=1)
        self.assertIsInstance(evidence, PreEvidence)
        self.assertEqual(evidence.test_output, "")
        self.assertIsNone(evidence.test_exit_code)
        self.assertIsNone(evidence.build_exit_code)
        self.assertIsNone(evidence.typecheck_exit_code)
        self.assertEqual(evidence.git_diff_recent, "")

    def test_collect_invokes_pytest_when_requirements_present(self) -> None:
        (self.tmpdir / "requirements.txt").write_text("flask\n")
        sandbox = Sandbox(self.tmpdir)
        with mock.patch("d2p.agents.pre_evidence.shutil.which",
                        side_effect=lambda c: "/usr/bin/" + c if c == "pytest" else None):
            with mock.patch("d2p.agents.pre_evidence.subprocess.run") as runner:
                runner.return_value = mock.Mock(
                    stdout="5 passed\n", stderr="", returncode=0)
                evidence = collect(sandbox, iter_count=1)
        runner.assert_called_once()
        invoked_cmd = runner.call_args.args[0]
        self.assertEqual(invoked_cmd[0], "pytest")
        self.assertEqual(evidence.test_exit_code, 0)
        self.assertIn("5 passed", evidence.test_output)

    def test_collect_records_127_when_runner_missing(self) -> None:
        (self.tmpdir / "package.json").write_text('{"name":"x"}\n')
        sandbox = Sandbox(self.tmpdir)
        with mock.patch("d2p.agents.pre_evidence.shutil.which", return_value=None):
            evidence = collect(sandbox, iter_count=1)
        self.assertEqual(evidence.test_output, "")
        self.assertIsNone(evidence.test_exit_code)

    def test_collect_falls_back_to_head_diff_when_history_shorter(self) -> None:
        (self.tmpdir / ".git").mkdir()
        sandbox = Sandbox(self.tmpdir)
        responses = [
            mock.Mock(stdout="", stderr="fatal: bad revision", returncode=128),
            mock.Mock(stdout="diff --git a/x b/x\n+y", stderr="", returncode=0),
        ]
        with mock.patch("d2p.agents.pre_evidence.shutil.which",
                        side_effect=lambda c: "/usr/bin/" + c if c == "git" else None):
            with mock.patch("d2p.agents.pre_evidence.subprocess.run",
                            side_effect=responses) as runner:
                evidence = collect(sandbox, iter_count=5)
        self.assertEqual(runner.call_count, 2)
        first_call_cmd = runner.call_args_list[0].args[0]
        second_call_cmd = runner.call_args_list[1].args[0]
        self.assertEqual(first_call_cmd, ["git", "diff", "HEAD~5..HEAD"])
        self.assertEqual(second_call_cmd, ["git", "diff", "HEAD"])
        self.assertIn("diff --git", evidence.git_diff_recent)
