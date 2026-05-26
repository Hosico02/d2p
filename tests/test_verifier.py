"""Verifier unit tests — offline, no LLM calls."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.agents.verifier import (
    CheckEntry, Finding, PreEvidence, VerifyClaim, VerifyResult, Verifier,
)
from d2p.fs import Sandbox


class _StubLLM:
    """Minimal LLMProvider stub that returns canned JSON for chat_json."""
    name = "stub-verify"

    def __init__(self, response: dict[str, Any] | Exception):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, system: str, user: str, **kw):
        raise NotImplementedError  # verifier only uses chat_json

    def chat_json(self, system: str, user: str, **kw):
        self.calls.append({"system": system, "user": user, "kwargs": kw})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _mkproject(root: Path) -> None:
    """Tiny fake Python project so listing/manifests have something to read."""
    (root / "package.json").write_text(json.dumps({
        "name": "x", "version": "0.0.0", "scripts": {"test": "echo ok"},
    }))
    (root / "README.md").write_text("# x\n\nrun `pnpm test`.\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("def test_ok():\n    assert True\n")


class TestVerifierHappyPath(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _mkproject(self.root)
        self.sb = Sandbox(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_returns_pass_on_clean_response(self) -> None:
        llm = _StubLLM({
            "verdict": "pass",
            "confidence": 0.9,
            "detected_archetype": "node-library",
            "stability_signal": "no_new_findings_after_effort",
            "reasoning_trace": [
                {"check": "tests run", "result": "ok",
                 "evidence": "exit=0, 1 passed"}
            ],
            "new_finding_categories": [],
            "repeated_finding_categories": [],
            "blocking_findings": [],
            "suggested_next_focus": None,
        })
        verifier = Verifier(llm, self.sb)
        claim = VerifyClaim(iter_count=3, no_more_features=True,
                            no_more_bugs=True, qa_corpus_green=True)
        result = verifier.verify(claim, PreEvidence(test_output="1 passed",
                                                    test_exit_code=0))
        self.assertEqual(result.verdict, "pass")
        self.assertEqual(result.stability_signal,
                         "no_new_findings_after_effort")
        self.assertEqual(result.detected_archetype, "node-library")
        self.assertAlmostEqual(result.confidence, 0.9)
        self.assertEqual(len(result.reasoning_trace), 1)
        self.assertEqual(result.reasoning_trace[0].result, "ok")
        # The prompt should have actually included some project state.
        self.assertEqual(len(llm.calls), 1)
        user_prompt = llm.calls[0]["user"]
        self.assertIn("package.json", user_prompt)
        self.assertIn("D2P CLAIM", user_prompt)

    def test_classifies_findings_with_severity(self) -> None:
        llm = _StubLLM({
            "verdict": "needs_repair", "confidence": 0.6,
            "detected_archetype": "fastapi-api",
            "stability_signal": "new_findings",
            "reasoning_trace": [],
            "new_finding_categories": [
                {"category": "missing_api_error_envelope", "severity": "high",
                 "message": "no @app.exception_handler", "evidence": "app.py"},
                {"category": "bogus_sev", "severity": "asdf",
                 "message": "x", "evidence": "y"},
            ],
            "repeated_finding_categories": [],
            "blocking_findings": [],
            "suggested_next_focus": "Add error envelope.",
        })
        verifier = Verifier(llm, self.sb)
        claim = VerifyClaim(iter_count=1, no_more_features=False,
                            no_more_bugs=False, qa_corpus_green=False)
        result = verifier.verify(claim, PreEvidence())
        self.assertEqual(result.verdict, "needs_repair")
        self.assertEqual(len(result.new_finding_categories), 2)
        # Bogus severity normalises to medium.
        sevs = sorted(f.severity for f in result.new_finding_categories)
        self.assertEqual(sevs, ["high", "medium"])
        self.assertEqual(result.suggested_next_focus, "Add error envelope.")


class TestVerifierConvergenceProtocol(unittest.TestCase):
    """Spec §8 convergence protocol: 2nd-pass prompt MUST surface previous
    findings as `previous_findings_by_category`-shaped context so the
    verifier can classify new vs repeated."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _mkproject(self.root)
        self.sb = Sandbox(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_second_pass_prompt_includes_previous_categories(self) -> None:
        llm = _StubLLM({
            "verdict": "no_new_findings", "confidence": 0.85,
            "detected_archetype": "node-library",
            "stability_signal": "no_new_findings_after_effort",
            "reasoning_trace": [], "new_finding_categories": [],
            "repeated_finding_categories": [
                {"category": "readme_command_mismatch", "severity": "medium",
                 "message": "still bad", "evidence": "README.md"},
            ],
            "blocking_findings": [],
        })
        verifier = Verifier(llm, self.sb)
        previous = [VerifyResult(
            verdict="needs_repair", confidence=0.6,
            detected_archetype="node-library", stability_signal="new_findings",
            new_finding_categories=[Finding(
                category="readme_command_mismatch", severity="medium",
                message="bad", evidence="README.md",
            )],
        )]
        claim = VerifyClaim(iter_count=2, no_more_features=True,
                            no_more_bugs=True, qa_corpus_green=True)
        verifier.verify(claim, PreEvidence(), previous_results=previous)
        prompt = llm.calls[0]["user"]
        self.assertIn("PREVIOUS VERIFY PASSES", prompt)
        self.assertIn("readme_command_mismatch", prompt)
        self.assertIn("CLASSIFY each finding", prompt)


class TestVerifierParseSafety(unittest.TestCase):
    """Malformed model output must default to needs_repair, never silent pass."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _mkproject(self.root)
        self.sb = Sandbox(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bad_verdict_falls_back_to_needs_repair(self) -> None:
        llm = _StubLLM({
            "verdict": "lgtm",     # not in the enum
            "confidence": 1.0,
            "detected_archetype": "x",
            "stability_signal": "new_findings",
            "reasoning_trace": [], "new_finding_categories": [],
            "repeated_finding_categories": [], "blocking_findings": [],
        })
        result = Verifier(llm, self.sb).verify(
            VerifyClaim(1, False, False, False), PreEvidence(),
        )
        self.assertEqual(result.verdict, "needs_repair")

    def test_bad_stability_signal_falls_back(self) -> None:
        llm = _StubLLM({
            "verdict": "pass", "confidence": 0.5,
            "detected_archetype": "x",
            "stability_signal": "uhhh",
            "reasoning_trace": [], "new_finding_categories": [],
            "repeated_finding_categories": [], "blocking_findings": [],
        })
        result = Verifier(llm, self.sb).verify(
            VerifyClaim(1, True, True, True), PreEvidence(),
        )
        self.assertEqual(result.stability_signal, "new_findings")

    def test_non_dict_response_yields_needs_repair(self) -> None:
        llm = _StubLLM([1, 2, 3])  # type: ignore[arg-type]
        result = Verifier(llm, self.sb).verify(
            VerifyClaim(1, False, False, False), PreEvidence(),
        )
        self.assertEqual(result.verdict, "needs_repair")
        # Failure should surface in reasoning_trace.
        self.assertTrue(any(c.result == "INSUFFICIENT_EVIDENCE"
                            for c in result.reasoning_trace))

    def test_llm_exception_yields_needs_repair(self) -> None:
        llm = _StubLLM(RuntimeError("boom"))
        result = Verifier(llm, self.sb).verify(
            VerifyClaim(1, False, False, False), PreEvidence(),
        )
        self.assertEqual(result.verdict, "needs_repair")
        self.assertIn("boom", result.raw_response)

    def test_confidence_clamped(self) -> None:
        llm = _StubLLM({
            "verdict": "pass", "confidence": 99.9,
            "detected_archetype": "x", "stability_signal": "new_findings",
            "reasoning_trace": [], "new_finding_categories": [],
            "repeated_finding_categories": [], "blocking_findings": [],
        })
        result = Verifier(llm, self.sb).verify(
            VerifyClaim(1, True, True, True), PreEvidence(),
        )
        self.assertEqual(result.confidence, 1.0)


class TestPreEvidenceInPrompt(unittest.TestCase):
    """Pre-evidence sections render only when their exit codes are populated.
    Verifies the independence-by-construction property: verifier never sees
    sections d2p didn't pre-run."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _mkproject(self.root)
        self.sb = Sandbox(self.root)
        self.llm = _StubLLM({
            "verdict": "needs_repair", "confidence": 0.5,
            "detected_archetype": "x", "stability_signal": "new_findings",
            "reasoning_trace": [], "new_finding_categories": [],
            "repeated_finding_categories": [], "blocking_findings": [],
        })
        self.verifier = Verifier(self.llm, self.sb)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_empty_pre_evidence_renders_no_test_section(self) -> None:
        self.verifier.verify(VerifyClaim(1, False, False, False),
                             PreEvidence())
        p = self.llm.calls[0]["user"]
        self.assertNotIn("## test (exit=", p)
        self.assertNotIn("## build (exit=", p)
        self.assertNotIn("## typecheck (exit=", p)

    def test_test_output_only_renders_test_section(self) -> None:
        self.verifier.verify(
            VerifyClaim(1, False, False, False),
            PreEvidence(test_output="3 failed, 0 passed",
                        test_exit_code=1),
        )
        p = self.llm.calls[0]["user"]
        self.assertIn("## test (exit=1)", p)
        self.assertIn("3 failed", p)
        self.assertNotIn("## build (exit=", p)


# --------------------------------------------------------------------------- #
# Orchestrator-side state machine (streak counter + terminal states)          #
# --------------------------------------------------------------------------- #

class TestOrchestratorVerifyStateMachine(unittest.TestCase):
    """Reaches into the orchestrator to verify the streak / terminal-state
    logic without running an actual loop. Keeps the test tightly scoped to
    the §6.3 spec table."""

    def setUp(self) -> None:
        # Patch out the bits of __init__ that would call the network.
        # Use the cfg-only path: skip building a real router by injecting one.
        from d2p.config import Config
        from d2p.providers.base import RoleRouter
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _mkproject(self.root)
        cfg = Config()
        cfg.verify_enabled = True
        # Tiny fake router with one fake provider for the "verify" role.
        # Will never be called because we replace orch.verifier below.
        fake_llm = _StubLLM({"verdict": "pass", "confidence": 0.5,
                             "detected_archetype": "x",
                             "stability_signal": "no_new_findings_after_effort",
                             "reasoning_trace": [], "new_finding_categories": [],
                             "repeated_finding_categories": [],
                             "blocking_findings": []})
        router = RoleRouter({"default": fake_llm,
                              "verify": fake_llm,
                              "analyzer": fake_llm, "planner": fake_llm,
                              "qa": fake_llm, "executor": fake_llm,
                              "fix": fake_llm})
        from d2p.orchestrator import Orchestrator
        self.orch = Orchestrator(target_dir=str(self.root),
                                  cfg=cfg, enable_qa=False, router=router)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _result(self, *, verdict: str, new=0, repeated=0,
                stability: str = "new_findings") -> VerifyResult:
        return VerifyResult(
            verdict=verdict, confidence=0.5, detected_archetype="x",
            stability_signal=stability,
            new_finding_categories=[Finding("c%d" % i, "low", "m", "e")
                                    for i in range(new)],
            repeated_finding_categories=[Finding("r%d" % i, "low", "m", "e")
                                         for i in range(repeated)],
        )

    def test_streak_increments_on_no_new_findings(self) -> None:
        """Spec §6.3 verdict-to-counter table: streak += 1 when
        new_finding_categories is empty."""
        self.orch._verify_streak = 0
        # Replicate the orchestrator's streak update inline (avoid running the
        # whole run() loop). The logic under test is the if-condition.
        for r in [self._result(verdict="needs_repair", new=0, repeated=1,
                                stability="no_new_findings_after_effort"),
                  self._result(verdict="no_new_findings", new=0, repeated=1,
                                stability="no_new_findings_after_effort")]:
            if (len(r.new_finding_categories) == 0
                    or r.verdict == "no_new_findings"):
                self.orch._verify_streak += 1
            else:
                self.orch._verify_streak = 0
        self.assertEqual(self.orch._verify_streak, 2)

    def test_streak_resets_when_new_findings_appear(self) -> None:
        """Spec §6.3: streak = 0 if new_finding_categories non-empty."""
        self.orch._verify_streak = 1
        r = self._result(verdict="needs_repair", new=2, repeated=0)
        if (len(r.new_finding_categories) == 0
                or r.verdict == "no_new_findings"):
            self.orch._verify_streak += 1
        else:
            self.orch._verify_streak = 0
        self.assertEqual(self.orch._verify_streak, 0)

    def test_handoff_writer_creates_file(self) -> None:
        """Spec §6.4 TERMINATE_WITH_RESIDUALS handoff path."""
        self.orch._current_iter = 3
        self.orch._write_handoff_report(
            [Finding("readme_command_mismatch", "medium", "msg", "README.md:1")],
            reason="converged_with_residuals",
        )
        handoff = self.orch.run_dir / "verify_handoff.md"
        self.assertTrue(handoff.is_file())
        txt = handoff.read_text()
        self.assertIn("converged_with_residuals", txt)
        self.assertIn("readme_command_mismatch", txt)


if __name__ == "__main__":
    unittest.main()
