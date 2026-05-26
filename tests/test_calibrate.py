"""Unit tests for the calibration harness. No real LLM calls — all
Verifier interactions are mocked."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.calibration import (
    Metrics,
    classify_outcome,
    compute_metrics,
    match_categories,
    load_baseline_meta,
    OUTCOME_CATCH, OUTCOME_MISS, OUTCOME_CLEAN_PASS,
    OUTCOME_FALSE_ALARM, OUTCOME_ERROR,
)


class TestClassifyOutcome(unittest.TestCase):
    def test_broken_verdict_in_expected_set_is_catch(self) -> None:
        outcome = classify_outcome(
            kind="broken", verdict="needs_repair",
            expected_verdict_in=["needs_repair", "fail"])
        self.assertEqual(outcome, OUTCOME_CATCH)

    def test_broken_verdict_not_in_expected_set_is_miss(self) -> None:
        outcome = classify_outcome(
            kind="broken", verdict="pass",
            expected_verdict_in=["needs_repair", "fail"])
        self.assertEqual(outcome, OUTCOME_MISS)

    def test_productized_verdict_in_expected_set_is_clean_pass(self) -> None:
        outcome = classify_outcome(
            kind="productized", verdict="pass",
            expected_verdict_in=["pass", "no_new_findings"])
        self.assertEqual(outcome, OUTCOME_CLEAN_PASS)

    def test_productized_verdict_not_in_expected_set_is_false_alarm(self) -> None:
        outcome = classify_outcome(
            kind="productized", verdict="needs_repair",
            expected_verdict_in=["pass", "no_new_findings"])
        self.assertEqual(outcome, OUTCOME_FALSE_ALARM)

    def test_unknown_kind_returns_error(self) -> None:
        outcome = classify_outcome(
            kind="something_weird", verdict="pass",
            expected_verdict_in=["pass"])
        self.assertEqual(outcome, OUTCOME_ERROR)


class TestMatchCategories(unittest.TestCase):
    def test_empty_expected_returns_none_skip(self) -> None:
        # Empty list means "don't check categories" -> match is None.
        self.assertIsNone(match_categories(
            actual=["readme_command_mismatch"], expected_substrings=[]))

    def test_substring_match_is_case_insensitive(self) -> None:
        self.assertTrue(match_categories(
            actual=["README_Command_Mismatch"],
            expected_substrings=["readme"]))

    def test_no_substring_overlap_is_false(self) -> None:
        self.assertFalse(match_categories(
            actual=["missing_tests"],
            expected_substrings=["readme", "documentation"]))

    def test_any_actual_matching_any_substring_passes(self) -> None:
        self.assertTrue(match_categories(
            actual=["missing_tests", "readme_command_mismatch"],
            expected_substrings=["readme"]))

    def test_no_actual_categories_with_non_empty_expected_is_false(self) -> None:
        self.assertFalse(match_categories(
            actual=[], expected_substrings=["readme"]))


class TestComputeMetrics(unittest.TestCase):
    def _row(self, kind: str, outcome: str) -> dict:
        return {"kind": kind, "outcome": outcome, "actual_verdict": "pass"
                if outcome in (OUTCOME_MISS, OUTCOME_CLEAN_PASS)
                else "needs_repair"}

    def test_all_catches_and_clean_passes_meets_criteria(self) -> None:
        rows = [
            self._row("broken", OUTCOME_CATCH),
            self._row("broken", OUTCOME_CATCH),
            self._row("productized", OUTCOME_CLEAN_PASS),
        ]
        m = compute_metrics(rows)
        self.assertEqual(m.catch_rate, 1.0)
        self.assertEqual(m.fp_rate, 0.0)
        self.assertEqual(m.pass_on_broken, 0)
        self.assertTrue(m.criteria_met)
        self.assertEqual(m.total_baselines, 3)
        self.assertEqual(m.errors, 0)

    def test_catch_rate_below_threshold_fails_criteria(self) -> None:
        rows = [
            self._row("broken", OUTCOME_CATCH),
            self._row("broken", OUTCOME_MISS),
            self._row("broken", OUTCOME_MISS),
            self._row("productized", OUTCOME_CLEAN_PASS),
        ]
        m = compute_metrics(rows)
        self.assertAlmostEqual(m.catch_rate, 1/3, places=3)
        self.assertFalse(m.criteria_met)

    def test_false_alarm_above_threshold_fails_criteria(self) -> None:
        rows = [
            self._row("broken", OUTCOME_CATCH),
            self._row("productized", OUTCOME_FALSE_ALARM),
            self._row("productized", OUTCOME_FALSE_ALARM),
            self._row("productized", OUTCOME_CLEAN_PASS),
        ]
        m = compute_metrics(rows)
        self.assertAlmostEqual(m.fp_rate, 2/3, places=3)
        self.assertFalse(m.criteria_met)

    def test_pass_verdict_on_broken_is_hard_fail(self) -> None:
        # MISS with verdict "pass" specifically is pass_on_broken,
        # which is criteria-failing even if other metrics are ok.
        rows = [
            {"kind": "broken", "outcome": OUTCOME_MISS, "actual_verdict": "pass"},
            self._row("broken", OUTCOME_CATCH),
            self._row("broken", OUTCOME_CATCH),
            self._row("broken", OUTCOME_CATCH),
            self._row("broken", OUTCOME_CATCH),
            self._row("productized", OUTCOME_CLEAN_PASS),
        ]
        m = compute_metrics(rows)
        self.assertEqual(m.pass_on_broken, 1)
        self.assertFalse(m.criteria_met)

    def test_errors_excluded_from_rate_denominators(self) -> None:
        rows = [
            self._row("broken", OUTCOME_CATCH),
            {"kind": "broken", "outcome": OUTCOME_ERROR, "actual_verdict": ""},
            self._row("productized", OUTCOME_CLEAN_PASS),
        ]
        m = compute_metrics(rows)
        self.assertEqual(m.catch_rate, 1.0)
        self.assertEqual(m.fp_rate, 0.0)
        self.assertEqual(m.errors, 1)
        self.assertEqual(m.total_baselines, 3)


class TestLoadBaselineMeta(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmpdir,
                                                            ignore_errors=True))

    def test_valid_expected_json_loads(self) -> None:
        baseline = self.tmpdir / "x"
        baseline.mkdir()
        (baseline / "expected.json").write_text(json.dumps({
            "name": "x", "kind": "broken", "archetype": "python-cli",
            "expected_verdict_in": ["needs_repair"],
            "expected_categories_any_of": ["readme"],
            "notes": "n",
        }))
        meta = load_baseline_meta(baseline)
        self.assertEqual(meta["kind"], "broken")
        self.assertEqual(meta["expected_verdict_in"], ["needs_repair"])

    def test_missing_expected_json_raises(self) -> None:
        baseline = self.tmpdir / "y"
        baseline.mkdir()
        with self.assertRaises(FileNotFoundError):
            load_baseline_meta(baseline)

    def test_missing_required_field_raises_value_error(self) -> None:
        baseline = self.tmpdir / "z"
        baseline.mkdir()
        (baseline / "expected.json").write_text(json.dumps({
            "name": "z", "kind": "broken",  # missing required fields
        }))
        with self.assertRaises(ValueError) as ctx:
            load_baseline_meta(baseline)
        self.assertIn("required", str(ctx.exception).lower())


class TestRunBaseline(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmpdir,
                                                            ignore_errors=True))

    def _baseline(self, name: str, kind: str, expected_in: list[str],
                  expected_subs: list[str]) -> Path:
        b = self.tmpdir / name
        b.mkdir()
        (b / "expected.json").write_text(json.dumps({
            "name": name, "kind": kind, "archetype": "python-cli",
            "expected_verdict_in": expected_in,
            "expected_categories_any_of": expected_subs,
            "notes": "test",
        }))
        return b

    def test_run_baseline_produces_catch_row_when_verifier_reports_needs_repair(self) -> None:
        from d2p.calibration import run_baseline
        baseline = self._baseline("flask-bad-readme", "broken",
                                  ["needs_repair", "fail"], ["readme"])
        mock_verifier = mock.Mock()
        mock_result = mock.Mock()
        mock_result.verdict = "needs_repair"
        mock_result.new_finding_categories = [
            mock.Mock(category="readme_command_mismatch"),
        ]
        mock_result.to_dict.return_value = {"verdict": "needs_repair"}
        mock_verifier.verify.return_value = mock_result
        verifier_factory = mock.Mock(return_value=mock_verifier)

        with mock.patch("d2p.calibration.collect_pre_evidence") as cpe:
            cpe.return_value = mock.Mock(to_dict=lambda: {})
            row = run_baseline(baseline, verifier_factory)

        verifier_factory.assert_called_once_with(baseline)
        # Verifier.verify takes (claim, pre_evidence) positionally; no project_path kwarg
        called_args = mock_verifier.verify.call_args
        self.assertEqual(called_args.kwargs, {})
        self.assertEqual(len(called_args.args), 2)
        from d2p.agents.verifier import VerifyClaim
        self.assertIsInstance(called_args.args[0], VerifyClaim)
        self.assertEqual(row["name"], "flask-bad-readme")
        self.assertEqual(row["outcome"], OUTCOME_CATCH)
        self.assertEqual(row["actual_verdict"], "needs_repair")
        self.assertTrue(row["verdict_match"])
        self.assertTrue(row["category_match"])

    def test_run_baseline_records_error_when_verify_raises(self) -> None:
        from d2p.calibration import run_baseline
        baseline = self._baseline("flask-bad-readme", "broken",
                                  ["needs_repair", "fail"], ["readme"])
        mock_verifier = mock.Mock()
        mock_verifier.verify.side_effect = RuntimeError("api down")
        verifier_factory = mock.Mock(return_value=mock_verifier)
        with mock.patch("d2p.calibration.collect_pre_evidence") as cpe:
            cpe.return_value = mock.Mock(to_dict=lambda: {})
            row = run_baseline(baseline, verifier_factory)
        self.assertEqual(row["outcome"], OUTCOME_ERROR)
        self.assertIn("api down", row.get("error", ""))

    def test_run_baseline_dry_run_skips_verifier_factory(self) -> None:
        from d2p.calibration import run_baseline
        baseline = self._baseline("x", "broken", ["needs_repair"], [])
        verifier_factory = mock.Mock()
        with mock.patch("d2p.calibration.collect_pre_evidence") as cpe:
            cpe.return_value = mock.Mock(to_dict=lambda: {})
            row = run_baseline(baseline, verifier_factory, dry_run=True)
        verifier_factory.assert_not_called()
        self.assertEqual(row["actual_verdict"], "<dry-run>")
        self.assertEqual(row["outcome"], OUTCOME_ERROR)  # dry-run can't be classified


class TestWriteReport(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmpdir,
                                                            ignore_errors=True))

    def test_write_report_creates_json_and_md_files(self) -> None:
        from d2p.calibration import write_report
        rows = [
            {"name": "x", "kind": "broken", "outcome": OUTCOME_CATCH,
             "actual_verdict": "needs_repair",
             "expected_verdict_in": ["needs_repair"],
             "expected_categories_any_of": ["readme"],
             "actual_categories": ["readme_x"],
             "verdict_match": True, "category_match": True,
             "elapsed_seconds": 5.0, "verify_result": {}},
        ]
        meta = {"model": "minimax-m2.7-hs", "started_at": "2026-05-26T16:00:00Z",
                "elapsed_seconds": 5.0, "harness_version": "v0"}
        metrics = compute_metrics(rows)
        write_report(self.tmpdir, rows, metrics, meta)
        self.assertTrue((self.tmpdir / "report.json").exists())
        self.assertTrue((self.tmpdir / "report.md").exists())

        report = json.loads((self.tmpdir / "report.json").read_text())
        self.assertEqual(report["model"], "minimax-m2.7-hs")
        self.assertEqual(report["metrics"]["catch_rate"], 1.0)
        self.assertEqual(len(report["rows"]), 1)

        md = (self.tmpdir / "report.md").read_text()
        self.assertIn("catch_rate", md)
        self.assertIn("fp_rate", md)
        self.assertIn("pass_on_broken", md)
        self.assertIn("minimax-m2.7-hs", md)
        self.assertIn("x", md)  # baseline name surfaces


class TestMainExitCode(unittest.TestCase):
    def test_criteria_met_returns_0(self) -> None:
        from d2p.calibration import _exit_code_for
        m = Metrics(catch_rate=0.9, fp_rate=0.1, pass_on_broken=0,
                    criteria_met=True, total_baselines=8, errors=0)
        self.assertEqual(_exit_code_for(m), 0)

    def test_criteria_failed_returns_1(self) -> None:
        from d2p.calibration import _exit_code_for
        m = Metrics(catch_rate=0.5, fp_rate=0.1, pass_on_broken=0,
                    criteria_met=False, total_baselines=8, errors=0)
        self.assertEqual(_exit_code_for(m), 1)

    def test_any_errors_force_exit_2(self) -> None:
        from d2p.calibration import _exit_code_for
        m = Metrics(catch_rate=1.0, fp_rate=0.0, pass_on_broken=0,
                    criteria_met=True, total_baselines=8, errors=1)
        self.assertEqual(_exit_code_for(m), 2)


if __name__ == "__main__":
    unittest.main()
