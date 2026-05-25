"""Verifier agent — adversarial independence check on top of Analyzer/Planner/QA.

Single LLM call per pass. NOT a tool-using loop. The verifier sees only the
pre-evidence d2p hands it (project tree paths + manifests + README + test
output + build/typecheck output + recent git diff). It never reads .d2p/
internal state, by construction.

Design spec:
  ../demo2project/docs/superpowers/specs/2026-05-22-d2p-verify-agent-design.md

Termination authority lives in `orchestrator.py` — the verifier itself is
stateless across calls. The orchestrator persists previous_results and feeds
them back in on subsequent passes so the verifier can classify findings as
new vs repeated.
"""
from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ..fs import Sandbox
from ..providers.base import LLMProvider

log = logging.getLogger("d2p.agents.verifier")


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class VerifyClaim:
    """d2p's claim that the project is done. The single fact verify is given
    about d2p's internal state — the rest is pure project observation."""
    iter_count: int
    no_more_features: bool
    no_more_bugs: bool
    qa_corpus_green: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreEvidence:
    """Pre-pulled execution evidence. Verifier never invokes commands itself —
    d2p runs them and bundles the output here."""
    test_output: str = ""
    test_exit_code: Optional[int] = None
    build_output: Optional[str] = None
    build_exit_code: Optional[int] = None
    typecheck_output: Optional[str] = None
    typecheck_exit_code: Optional[int] = None
    git_diff_recent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """A single defect found by the verifier."""
    category: str          # e.g. "missing_api_error_envelope"
    severity: str          # "blocker" | "high" | "medium" | "low"
    message: str
    evidence: str          # file path + snippet, output excerpt, or absence statement

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckEntry:
    """One reasoning-trace row. Forces verify to attach evidence per check
    rather than handwave 'looks ok'."""
    check: str
    result: str            # "ok" | "fail" | "INSUFFICIENT_EVIDENCE"
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyResult:
    verdict: str                                # pass | needs_repair | fail | no_new_findings
    confidence: float
    detected_archetype: str
    stability_signal: str                       # new_findings | no_new_findings_after_effort
    reasoning_trace: list[CheckEntry] = field(default_factory=list)
    new_finding_categories: list[Finding] = field(default_factory=list)
    repeated_finding_categories: list[Finding] = field(default_factory=list)
    blocking_findings: list[Finding] = field(default_factory=list)
    suggested_next_focus: Optional[str] = None
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "detected_archetype": self.detected_archetype,
            "stability_signal": self.stability_signal,
            "reasoning_trace": [c.to_dict() for c in self.reasoning_trace],
            "new_finding_categories": [f.to_dict() for f in self.new_finding_categories],
            "repeated_finding_categories": [f.to_dict() for f in self.repeated_finding_categories],
            "blocking_findings": [f.to_dict() for f in self.blocking_findings],
            "suggested_next_focus": self.suggested_next_focus,
            "raw_response": self.raw_response,
        }


# --------------------------------------------------------------------------- #
# System prompt                                                               #
# --------------------------------------------------------------------------- #

# Adversarial + anti-fabrication. Default verdict is `needs_repair`, NOT pass.
# The convergence-protocol section is what makes fixed-point termination work:
# without `no_new_findings_after_effort` as an explicit escape hatch, the loop
# can never cleanly terminate because Opus will always produce "something".
SYSTEM_PROMPT = """You are an independent verifier. You are NOT part of the d2p productization
pipeline. You do not assist users. Your sole purpose is to determine whether
a project that d2p claims to have productized actually meets the bar.

Your default verdict is `needs_repair`. You upgrade to `pass` ONLY when you
find specific positive evidence covering every relevant category for the
project's detected archetype. You upgrade to `fail` when you find at least
one blocker — a defect that would make this project unfit to ship to even
internal users.

Relevant categories (apply only those matching the detected archetype):
- Tests actually run and pass (not skipped, not empty, not assert(True))
- Test invocation in README matches the package manifest
- Build / typecheck pass (use pre-pulled evidence; do not infer)
- Error envelope exists for HTTP surfaces (404 + unhandled exception
  both return structured JSON with a stable shape)
- Runtime contracts exist for detected surfaces (API routes, CLI args,
  worker entries, config loading)
- Dependencies are pinned (no `*`, caret-only on production deps)
- Deployment artifacts are real (Dockerfile uses a prod server, not
  `python app.py`; wsgi.py exists if framework needs it)
- README claims map to actual files / scripts / commands

For each category you assess, you MUST emit a `reasoning_trace` entry
with:
- `check`: human-readable name of what you checked
- `result`: `ok` | `fail` | `INSUFFICIENT_EVIDENCE`
- `evidence`: a specific file path + content snippet, or a test output
  excerpt, or an explicit absence statement

"Looks ok" without an evidence field is not acceptable. If you cannot
locate evidence for a category that matters for this archetype, emit
INSUFFICIENT_EVIDENCE — this downgrades the overall verdict to
`needs_repair`.

You will receive d2p's claim that the project is done. Treat this claim
as a hypothesis to refute, not a fact to confirm. Refusing to refute
without evidence is acceptable; confirming without evidence is not.

If your verdict is `needs_repair`, you must populate
`suggested_next_focus` with the single most important category for
d2p's next iteration to address. Be specific.

If your verdict is `fail`, you must list at least one
`blocking_findings` entry with a clear explanation of why this is
unrecoverable in the next iteration and requires human review.

---

CONVERGENCE PROTOCOL (applies on the 2nd and later verify passes for
the same run):

You will be given a list of `previous_findings_by_category` summarizing
what earlier verify passes already surfaced. Your job on this pass:

1. Independently re-derive findings as you would on a fresh project.
   Do NOT just copy the previous list.
2. For each finding you produce, classify it:
   - `new` — this category did NOT appear in any previous pass for this
     run. These are what the loop has not yet seen.
   - `repeated` — this category was already flagged previously and is
     still present. d2p either couldn't fix it or decided not to.
3. After genuine adversarial effort, if you cannot identify ANY new
   category beyond previously-surfaced ones, you MUST set
   `stability_signal = "no_new_findings_after_effort"` and use verdict
   `no_new_findings`. This is the correct response when the project
   has genuinely converged.

This `no_new_findings` outcome is REQUIRED to be available — it is the
signal the pipeline uses to detect convergence. Without it, the loop
cannot terminate cleanly. Fabricating "new" findings to look productive
is a failure mode, not success. The system depends on you being honest
about whether you genuinely found something new.

If you DO find genuinely new categories: list them in
`new_finding_categories`, classify the rest as `repeated_finding_categories`,
and set `stability_signal = "new_findings"`. Verdict should be
`needs_repair` (or `fail` if a blocker is among the new).

Two checks on yourself before you finalize:
- "Am I marking something as new just because the prompt asked me to
  find something?" If yes, downgrade it.
- "Did I genuinely look at this with fresh eyes, or did I anchor on
  the previous pass?" If anchored, re-examine from scratch.

Output STRICT JSON only — no markdown fences, no commentary."""


# --------------------------------------------------------------------------- #
# Schema (used both for response_format coercion and for documenting shape)   #
# --------------------------------------------------------------------------- #

VERIFY_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "needs_repair", "fail", "no_new_findings"],
        },
        "confidence": {"type": "number"},
        "detected_archetype": {"type": "string"},
        "stability_signal": {
            "type": "string",
            "enum": ["new_findings", "no_new_findings_after_effort"],
        },
        "reasoning_trace": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "check": {"type": "string"},
                    "result": {
                        "type": "string",
                        "enum": ["ok", "fail", "INSUFFICIENT_EVIDENCE"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["check", "result", "evidence"],
            },
        },
        "new_finding_categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "high", "medium", "low"],
                    },
                    "message": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["category", "severity", "message", "evidence"],
            },
        },
        "repeated_finding_categories": {
            "type": "array",
            "items": {"$ref": "#/properties/new_finding_categories/items"},
        },
        "blocking_findings": {
            "type": "array",
            "items": {"$ref": "#/properties/new_finding_categories/items"},
        },
        "suggested_next_focus": {"type": ["string", "null"]},
    },
    "required": [
        "verdict", "confidence", "detected_archetype", "stability_signal",
        "reasoning_trace", "new_finding_categories",
        "repeated_finding_categories", "blocking_findings",
    ],
}


# --------------------------------------------------------------------------- #
# Verifier agent                                                              #
# --------------------------------------------------------------------------- #

class Verifier:
    """B+ independence: sees project state + d2p's done-claim + pre-pulled
    test/build/typecheck output. Never sees Analyzer/Planner/QA outputs.

    Implementation: single chat_json call per verify pass. No tool loop.
    """

    # File-collection limits (§7 of spec). Total prompt target: 10-20k tokens.
    MAX_TREE_ENTRIES = 400
    MAX_FILE_CHARS = 4000          # per manifest / entry file
    MAX_TEST_FILE_CHARS = 6000     # first 200ish lines per test file
    MAX_TEST_FILES = 12
    MAX_TEST_OUTPUT_CHARS = 6000
    MAX_BUILD_OUTPUT_CHARS = 3000
    MAX_TYPECHECK_OUTPUT_CHARS = 3000
    MAX_GIT_DIFF_CHARS = 8000

    # Standard files we always try to read (skip silently if absent).
    MANIFEST_FILES = (
        "package.json", "pyproject.toml", "setup.py", "setup.cfg",
        "requirements.txt", "Cargo.toml", "go.mod", "Gemfile",
        "pom.xml", "build.gradle", "Dockerfile", "docker-compose.yml",
        "docker-compose.yaml", ".env.example", "LICENSE",
        "README.md", "README.rst", "README",
    )

    ENTRY_HINTS_BY_LANG = {
        "python": ("app.py", "main.py", "wsgi.py", "asgi.py", "cli.py",
                   "manage.py", "src/main.py", "src/app.py"),
        "node":   ("index.js", "index.ts", "index.mjs", "server.js",
                   "server.ts", "app.js", "app.ts", "src/index.ts",
                   "src/index.js", "src/server.ts"),
        "rust":   ("src/main.rs", "src/lib.rs"),
        "go":     ("main.go", "cmd/main.go"),
    }

    CI_PATHS = (
        ".github/workflows/ci.yml",
        ".github/workflows/test.yml",
        ".github/workflows/main.yml",
        ".gitlab-ci.yml",
        ".circleci/config.yml",
    )

    def __init__(self, llm: LLMProvider, sandbox: Sandbox) -> None:
        self.llm = llm
        self.sandbox = sandbox

    # ---- public API ----------------------------------------------------- #

    def verify(
        self,
        claim: VerifyClaim,
        pre_evidence: PreEvidence,
        previous_results: Optional[list[VerifyResult]] = None,
    ) -> VerifyResult:
        """Single Opus-class call. Returns a VerifyResult that classifies
        findings against `previous_results` so the orchestrator can drive
        the fixed-point convergence streak."""
        user_prompt = self._build_user_prompt(claim, pre_evidence,
                                              previous_results or [])
        log.info("Verifier: calling LLM (iter %d, previous_results=%d)",
                 claim.iter_count, len(previous_results or []))
        try:
            data = self.llm.chat_json(
                SYSTEM_PROMPT, user_prompt,
                web_search=False,
                temperature=0.2,    # lower than analyzer/planner — verdict stability
                max_tokens=4000,
            )
        except Exception as e:
            log.warning("Verifier LLM call failed: %s — defaulting to needs_repair",
                        e)
            return VerifyResult(
                verdict="needs_repair",
                confidence=0.0,
                detected_archetype="unknown",
                stability_signal="new_findings",
                reasoning_trace=[CheckEntry(
                    check="verifier llm call",
                    result="INSUFFICIENT_EVIDENCE",
                    evidence=f"verifier call raised {type(e).__name__}: {e}",
                )],
                suggested_next_focus=(
                    "verifier failed; rerun or inspect logs"),
                raw_response=str(e),
            )
        return self._parse_result(data)

    # ---- prompt construction ------------------------------------------- #

    def _build_user_prompt(self, claim: VerifyClaim,
                           pre_evidence: PreEvidence,
                           previous_results: list[VerifyResult]) -> str:
        parts: list[str] = []

        parts.append("D2P CLAIM (verify hypothesis to refute):")
        parts.append(_json.dumps(claim.to_dict(), ensure_ascii=False, indent=2))
        parts.append("")

        # Pre-pulled evidence — always include sections that have data.
        parts.append("PRE-EVIDENCE (commands d2p ran before calling verify):")
        if pre_evidence.test_exit_code is not None:
            parts.append(f"\n## test (exit={pre_evidence.test_exit_code})")
            parts.append("```")
            parts.append(_truncate(pre_evidence.test_output,
                                   self.MAX_TEST_OUTPUT_CHARS))
            parts.append("```")
        if pre_evidence.build_exit_code is not None:
            parts.append(f"\n## build (exit={pre_evidence.build_exit_code})")
            parts.append("```")
            parts.append(_truncate(pre_evidence.build_output or "",
                                   self.MAX_BUILD_OUTPUT_CHARS))
            parts.append("```")
        if pre_evidence.typecheck_exit_code is not None:
            parts.append(f"\n## typecheck (exit={pre_evidence.typecheck_exit_code})")
            parts.append("```")
            parts.append(_truncate(pre_evidence.typecheck_output or "",
                                   self.MAX_TYPECHECK_OUTPUT_CHARS))
            parts.append("```")
        if pre_evidence.git_diff_recent:
            parts.append("\n## git diff (recent iterations)")
            parts.append("```diff")
            parts.append(_truncate(pre_evidence.git_diff_recent,
                                   self.MAX_GIT_DIFF_CHARS))
            parts.append("```")
        parts.append("")

        # Project tree (paths only).
        parts.append("PROJECT TREE (paths only, .d2p/.git/node_modules etc excluded):")
        listing = self.sandbox.listing(max_entries=self.MAX_TREE_ENTRIES)
        parts.append("\n".join(listing) if listing else "(empty)")
        parts.append("")

        # Manifest files — read directly.
        parts.append("MANIFEST / TOP-LEVEL DOCS:")
        for fname in self.MANIFEST_FILES:
            txt = self.sandbox.read(fname)
            if txt:
                parts.append(f"\n=== {fname} ===")
                parts.append(_truncate(txt, self.MAX_FILE_CHARS))
        parts.append("")

        # Entry files — pick by detected language.
        entry_files = self._pick_entry_files(listing)
        if entry_files:
            parts.append("ENTRY FILES:")
            for fname in entry_files:
                txt = self.sandbox.read(fname)
                if txt:
                    parts.append(f"\n=== {fname} ===")
                    parts.append(_truncate(txt, self.MAX_FILE_CHARS))
            parts.append("")

        # CI configs.
        ci_blocks: list[str] = []
        for fname in self.CI_PATHS:
            txt = self.sandbox.read(fname)
            if txt:
                ci_blocks.append(f"\n=== {fname} ===\n{_truncate(txt, self.MAX_FILE_CHARS)}")
        if ci_blocks:
            parts.append("CI CONFIGS:")
            parts.extend(ci_blocks)
            parts.append("")

        # Test files — first MAX_TEST_FILE_CHARS each.
        test_files = self._pick_test_files(listing)
        if test_files:
            parts.append(f"TEST FILES (first {self.MAX_TEST_FILE_CHARS} chars each, "
                         f"up to {self.MAX_TEST_FILES} files):")
            for fname in test_files[: self.MAX_TEST_FILES]:
                txt = self.sandbox.read(fname)
                if txt:
                    parts.append(f"\n=== {fname} ===")
                    parts.append(_truncate(txt, self.MAX_TEST_FILE_CHARS))
            parts.append("")

        # Previous verify passes (convergence protocol).
        if previous_results:
            parts.append("PREVIOUS VERIFY PASSES (this run):")
            for i, prev in enumerate(previous_results, 1):
                cats_new = [f.category for f in prev.new_finding_categories]
                cats_rep = [f.category for f in prev.repeated_finding_categories]
                parts.append(f"\n## pass {i}: verdict={prev.verdict}, "
                             f"stability={prev.stability_signal}")
                parts.append(f"  new categories: {cats_new or '(none)'}")
                parts.append(f"  repeated categories: {cats_rep or '(none)'}")
                if prev.suggested_next_focus:
                    parts.append(f"  suggested_next_focus: {prev.suggested_next_focus}")
            parts.append("")
            parts.append(
                "CLASSIFY each finding you produce this pass as `new` (category "
                "absent above) or `repeated` (category present above). If you "
                "genuinely cannot find any new category after adversarial "
                "effort, set verdict=no_new_findings, "
                "stability_signal=no_new_findings_after_effort, and emit only "
                "repeated_finding_categories."
            )
        else:
            parts.append("PREVIOUS VERIFY PASSES: (none — this is pass 1)")

        parts.append("")
        parts.append("Return STRICT JSON conforming to the VerifyResult schema "
                     "(no markdown fences, no commentary).")
        return "\n".join(parts)

    # ---- helpers ------------------------------------------------------- #

    def _pick_entry_files(self, listing: list[str]) -> list[str]:
        """Pick entry-point files. Detect language by what's actually in the
        listing rather than by manifest priority — handles mixed repos."""
        in_tree = set(listing)
        out: list[str] = []
        # Order matters: try language families based on which manifest is
        # present, but always also collect any entry-point that exists.
        for hints in self.ENTRY_HINTS_BY_LANG.values():
            for h in hints:
                if h in in_tree:
                    out.append(h)
        return out

    def _pick_test_files(self, listing: list[str]) -> list[str]:
        """Heuristic test-file picker. Covers pytest, jest, mocha, cargo
        conventions. Sorted by path for stable prompts."""
        tests: list[str] = []
        for p in listing:
            base = Path(p).name.lower()
            if (p.startswith("tests/") or p.startswith("test/")
                    or "/tests/" in p or "/test/" in p
                    or base.startswith("test_")
                    or base.endswith(".test.ts") or base.endswith(".test.js")
                    or base.endswith(".spec.ts") or base.endswith(".spec.js")
                    or base.endswith("_test.go")
                    or "/tests.rs" in p):
                # Skip __pycache__/.snap noise.
                if "__pycache__" in p or p.endswith(".snap"):
                    continue
                tests.append(p)
        return sorted(tests)

    # ---- parsing -------------------------------------------------------- #

    def _parse_result(self, data: Any) -> VerifyResult:
        """Tolerant JSON → VerifyResult. Defaults to needs_repair on shape
        problems so a malformed model output never silently looks like pass."""
        if not isinstance(data, dict):
            log.warning("Verifier returned non-dict: %r", type(data).__name__)
            return VerifyResult(
                verdict="needs_repair", confidence=0.0,
                detected_archetype="unknown", stability_signal="new_findings",
                reasoning_trace=[CheckEntry(
                    check="parse model output", result="INSUFFICIENT_EVIDENCE",
                    evidence=f"non-dict response: {type(data).__name__}",
                )],
                raw_response=_json.dumps(data) if data is not None else "",
            )

        verdict = str(data.get("verdict") or "needs_repair").strip()
        if verdict not in {"pass", "needs_repair", "fail", "no_new_findings"}:
            verdict = "needs_repair"

        stability = str(data.get("stability_signal") or "new_findings").strip()
        if stability not in {"new_findings", "no_new_findings_after_effort"}:
            stability = "new_findings"

        # Confidence: coerce to 0..1.
        try:
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        return VerifyResult(
            verdict=verdict,
            confidence=confidence,
            detected_archetype=str(data.get("detected_archetype") or "unknown"),
            stability_signal=stability,
            reasoning_trace=[_parse_check(c)
                             for c in (data.get("reasoning_trace") or [])
                             if isinstance(c, dict)],
            new_finding_categories=[_parse_finding(f)
                                    for f in (data.get("new_finding_categories") or [])
                                    if isinstance(f, dict)],
            repeated_finding_categories=[_parse_finding(f)
                                         for f in (data.get("repeated_finding_categories") or [])
                                         if isinstance(f, dict)],
            blocking_findings=[_parse_finding(f)
                               for f in (data.get("blocking_findings") or [])
                               if isinstance(f, dict)],
            suggested_next_focus=(str(data["suggested_next_focus"]).strip()
                                  if data.get("suggested_next_focus") else None),
            raw_response=_json.dumps(data, ensure_ascii=False),
        )


# --------------------------------------------------------------------------- #
# Module-level parse helpers                                                  #
# --------------------------------------------------------------------------- #

def _parse_finding(f: dict[str, Any]) -> Finding:
    sev = str(f.get("severity", "medium")).strip().lower()
    if sev not in {"blocker", "high", "medium", "low"}:
        sev = "medium"
    return Finding(
        category=str(f.get("category", "")).strip() or "unspecified",
        severity=sev,
        message=str(f.get("message", "")).strip(),
        evidence=str(f.get("evidence", "")).strip(),
    )


def _parse_check(c: dict[str, Any]) -> CheckEntry:
    res = str(c.get("result", "INSUFFICIENT_EVIDENCE")).strip()
    if res not in {"ok", "fail", "INSUFFICIENT_EVIDENCE"}:
        res = "INSUFFICIENT_EVIDENCE"
    return CheckEntry(
        check=str(c.get("check", "")).strip(),
        result=res,
        evidence=str(c.get("evidence", "")).strip(),
    )


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n... (truncated; original {len(s)} chars)"
