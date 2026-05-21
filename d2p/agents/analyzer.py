"""Analyzer agent — reads the demo and produces a competitor-grounded gap matrix.

The Analyzer runs THREE LLM calls in sequence:

  Phase 1 — Competitor discovery (web_search=ON)
      Identifies domain/essence/audience and finds 3-5 real competitor products
      with concrete key_features. Does NOT emit features yet.

  Phase 2 — Capability extraction (web_search=OFF)
      Reads the actual contents of key source files (not just paths/README)
      and lists what the demo CURRENTLY does. Grounds later gap analysis in
      real evidence, not file-name guesses.

  Phase 3 — Gap analysis (web_search=OFF)
      Cross-references competitor key_features against demo_capabilities,
      emits features with `in_demo`, `evidence_in_demo`, `gap_severity` so the
      Planner can prioritize gaps over already-present functionality.

Caching: results are persisted under <target>/.d2p/analysis_cache.json keyed
by a fingerprint of (listing + docs + capability-input + system prompts +
model identity). Pass `--no-cache-analysis` to force fresh.
"""
from __future__ import annotations

import hashlib
import json as _json
import logging
from pathlib import Path
from typing import Any

from ..fs import Sandbox
from ..providers.base import LLMProvider
from ..models import AnalysisReport, CompetitorDetail, Feature

log = logging.getLogger("d2p.agents.analyzer")


# --------------------------------------------------------------------------- #
# Phase 1 — Competitor discovery (web search)
# --------------------------------------------------------------------------- #
COMPETITOR_SYS = """You are the Competitor-Discovery phase of the d2p Analyzer.

Your job — and ONLY your job — is to:

1. From the demo listing + docs, identify:
   - "domain": the problem area (e.g. "social deduction game", "speech-to-text")
   - "essence": the demo's CORE NATURE that must be preserved. Captures what
     KIND of artifact this demo IS — who its real audience is, what makes it
     distinct from typical products in the same domain.
   - "audience": one short phrase, e.g. "LLM agents", "developers via API",
     "research notebook users", "humans on a web UI".

   Examples:
     * werewolf demo where 6 LLM-driven players debate each other
         -> essence: "Agent-vs-agent simulation harness; LLM players debate,
            vote, reason; humans are spectators, not players"
         -> audience: "LLM agents (humans only observe / analyze)"
     * a Whisper-based offline transcriber CLI
         -> essence: "offline batch-processing CLI; not a real-time web app"
         -> audience: "CLI users / scripted pipelines"

2. Search the web for 3-5 MATURE COMPETITOR PRODUCTS in the same DOMAIN.
   For EACH competitor, record:
   - name
   - 3-8 concrete key_features you actually found on their page/paper (not
     generic — name specific capabilities like "agent reasoning trace export"
     not "logging")
   - source_url (the page you got the info from)
   - notes (max 200 chars) on why this competitor matters as a reference

   If web_search returned nothing or is unavailable, say so explicitly in
   notes and leave key_features empty rather than inventing.

3. Also list `ui_elements` you'd expect such products to surface (short
   strings, max 12).

DO NOT emit a feature list at this stage. That happens in a later phase.

Output STRICT JSON only — no markdown, no commentary.
"""

COMPETITOR_USER_TMPL = """Demo project files:
{listing}

Top-level documents (truncated):
{docs}

Return a JSON object with this exact shape:
{{
  "domain": "<one sentence>",
  "essence": "<one or two sentences — the demo's core nature that must NOT change>",
  "audience": "<one short phrase>",
  "competitors": ["<product name + 1-line desc>", ...],
  "competitors_detail": [
    {{"name": "...", "key_features": ["...", "..."], "source_url": "...", "notes": "..."}}
  ],
  "ui_elements": ["...", ...],
  "raw_notes": "<short freeform notes about the discovery, max 500 chars>"
}}
"""


# --------------------------------------------------------------------------- #
# Phase 2 — Capability extraction (read actual code)
# --------------------------------------------------------------------------- #
CAPABILITY_SYS = """You are the Capability-Extraction phase of the d2p Analyzer.

You will be given the ACTUAL CONTENTS of the demo's main source files (not
just paths). Your job is to list what the demo CURRENTLY does — concrete,
user-visible behaviors and surfaced API/endpoints/CLI subcommands.

Rules:
- Be evidence-grounded. Each capability must be something you can point at
  in the code (a function, endpoint, CLI command, file, config flag).
- Be concrete. "Logs reasoning to stdout via print() in player.py" beats
  "has logging".
- Be exhaustive within reason: 10-25 capabilities is the target range.
- Include both happy-path features ("websocket /stream endpoint") and ops
  surfaces ("Dockerfile present", "tests in tests/").
- Do NOT speculate about what would be nice to add. That's a later phase.

Output STRICT JSON only.
"""

CAPABILITY_USER_TMPL = """Demo source files (truncated to {chars} chars each):

{files}

Return a JSON object with this exact shape:
{{
  "demo_capabilities": [
    "<one concrete capability — include file:symbol when known>",
    ...
  ]
}}
"""


# --------------------------------------------------------------------------- #
# Phase 3 — Gap analysis
# --------------------------------------------------------------------------- #
GAP_SYS = """You are the Gap-Analysis phase of the d2p Analyzer.

You are given:
  - the demo's essence + audience (immutable constraints),
  - demo_capabilities: what the demo CURRENTLY does (evidence-grounded),
  - competitors_detail: 3-5 competitors with their concrete key_features.

Your job: produce a GAP MATRIX of features the demo SHOULD have, with each
entry classified relative to current demo state.

Rules:

1. ESSENCE PRESERVATION. Every feature must respect essence + audience. If
   a competitor key_feature would change who the demo is for, either reject
   it or translate it into an essence-preserving analogue (e.g. for an
   agent-vs-agent harness, "voice chat" becomes "structured agent-to-agent
   private channel for reasoning").

2. EVIDENCE-GROUNDED. For each feature you propose:
   - `in_demo` is one of: "missing" | "partial" | "present"
   - `evidence_in_demo`: if partial/present, cite the demo_capability that
     covers it (or "—" if missing).
   - `source`: must match the `name` of a competitor in competitors_detail.
   - `gap_severity`: "high" if missing AND central to the product category;
     "medium" if partial OR a strong polish; "low" if nice-to-have.

3. SKIP features that are already fully present unless severity demands a
   visible enhancement.

4. PREFER small, focused features (1-2 files of impact) over sweeping new
   modules. The Planner picks 4-5 of these per iteration; large cross-cutting
   features cause regression cascades.

5. Provide 8-15 features total. Output STRICT JSON.
"""

GAP_USER_TMPL = """Essence (must be preserved):
{essence}

Audience (must be preserved):
{audience}

Demo currently does these things (capabilities):
{capabilities}

Competitors and their key features:
{competitors}

Return a JSON object with this exact shape:
{{
  "features": [
    {{
      "name": "...",
      "category": "backend|frontend|ux|ops|docs",
      "description": "<2-3 sentences; concrete, not generic>",
      "source": "<exact competitor name from the list above>",
      "in_demo": "missing|partial|present",
      "evidence_in_demo": "<demo capability quote or '—' if missing>",
      "gap_severity": "low|medium|high"
    }}
  ]
}}
"""


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #
_VALID_IN_DEMO = {"missing", "partial", "present"}
_VALID_SEVERITY = {"low", "medium", "high"}


def _normalize_feature(f: dict[str, Any]) -> dict[str, Any]:
    in_demo = str(f.get("in_demo", "")).strip().lower()
    if in_demo not in _VALID_IN_DEMO:
        in_demo = ""
    severity = str(f.get("gap_severity", "")).strip().lower()
    if severity not in _VALID_SEVERITY:
        severity = ""
    return {
        "name": str(f.get("name", "")).strip() or "unnamed",
        "category": str(f.get("category", "other")).strip().lower(),
        "description": str(f.get("description", "")).strip(),
        "source": str(f.get("source", "")).strip(),
        "in_demo": in_demo,
        "evidence_in_demo": str(f.get("evidence_in_demo", "")).strip(),
        "gap_severity": severity,
    }


def _normalize_competitor(c: dict[str, Any]) -> CompetitorDetail:
    raw_kf = c.get("key_features") or []
    if not isinstance(raw_kf, list):
        raw_kf = []
    return CompetitorDetail(
        name=str(c.get("name", "")).strip() or "unnamed",
        key_features=[str(x).strip() for x in raw_kf if str(x).strip()],
        source_url=str(c.get("source_url", "")).strip(),
        notes=str(c.get("notes", "")).strip(),
    )


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
class Analyzer:
    # Capability-extraction phase: how many source files to read, and how
    # many chars per file. Code is denser per char than READMEs so 3000 is
    # plenty for the call to identify capabilities.
    CAPABILITY_MAX_FILES = 6
    CAPABILITY_FILE_CHARS = 3000
    CAPABILITY_EXTENSIONS = (".py", ".ts", ".tsx", ".js", ".jsx",
                             ".go", ".rs", ".java", ".rb")

    def __init__(self, llm: LLMProvider, sandbox: Sandbox) -> None:
        self.llm = llm
        self.sandbox = sandbox

    # ---- input gathering ----------------------------------------------------

    def _gather_docs(self) -> str:
        candidates = ["README.md", "README", "readme.md", "package.json",
                      "pyproject.toml", "requirements.txt", "main.py", "app.py",
                      "index.js", "index.ts", "src/main.ts", "Cargo.toml"]
        chunks = []
        for c in candidates:
            txt = self.sandbox.read(c)
            if txt:
                chunks.append(f"=== {c} ===\n{txt[:2500]}")
        return "\n\n".join(chunks) or "(no obvious entry/doc files found)"

    def _build_input(self) -> tuple[str, str]:
        """Phase-1 input: (listing, docs)."""
        listing = "\n".join(self.sandbox.listing(max_entries=120))
        docs = self._gather_docs()
        return listing, docs

    def _pick_capability_files(self, listing: list[str]) -> list[str]:
        """Top-N source files by size — these tend to hold the real logic."""
        sizes: list[tuple[int, str]] = []
        for p in listing:
            if not p.endswith(self.CAPABILITY_EXTENSIONS):
                continue
            if "/test" in p or p.startswith("test"):
                continue
            try:
                sz = len(self.sandbox.read(p))
            except Exception:
                continue
            if sz > 0:
                sizes.append((sz, p))
        sizes.sort(reverse=True)
        return [p for _, p in sizes[: self.CAPABILITY_MAX_FILES]]

    def _build_capability_input(self) -> str:
        listing = self.sandbox.listing(max_entries=300)
        files = self._pick_capability_files(listing)
        chunks: list[str] = []
        for p in files:
            try:
                txt = self.sandbox.read(p)
            except Exception:
                continue
            if not txt:
                continue
            chunks.append(f"=== {p} ===\n{txt[: self.CAPABILITY_FILE_CHARS]}")
        if not chunks:
            return "(no source files found)"
        return "\n\n".join(chunks)

    # ---- fingerprint --------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable hash over (listing + docs + capability-input + all three
        system prompts + model identity). Bumped to v2 with the 3-phase
        pipeline so old caches don't return shape-mismatched data."""
        listing, docs = self._build_input()
        capability_input = self._build_capability_input()
        h = hashlib.sha256()
        h.update(b"d2p-analyzer-v2\n")
        h.update(COMPETITOR_SYS.encode("utf-8"))
        h.update(CAPABILITY_SYS.encode("utf-8"))
        h.update(GAP_SYS.encode("utf-8"))
        h.update(b"\n--listing--\n")
        h.update(listing.encode("utf-8"))
        h.update(b"\n--docs--\n")
        h.update(docs.encode("utf-8"))
        h.update(b"\n--capability-input--\n")
        h.update(capability_input.encode("utf-8"))
        h.update(b"\n--model--\n")
        h.update(getattr(self.llm, "name", "?").encode("utf-8"))
        return h.hexdigest()[:16]

    # ---- the three phases ---------------------------------------------------

    def _phase1_competitors(self, listing: str, docs: str) -> dict[str, Any]:
        user = COMPETITOR_USER_TMPL.format(listing=listing, docs=docs)
        return self.llm.chat_json(COMPETITOR_SYS, user, web_search=True,
                                  temperature=0.3, max_tokens=5000)

    def _phase2_capabilities(self, code: str) -> list[str]:
        user = CAPABILITY_USER_TMPL.format(
            chars=self.CAPABILITY_FILE_CHARS, files=code,
        )
        data = self.llm.chat_json(CAPABILITY_SYS, user, web_search=False,
                                  temperature=0.2, max_tokens=2500)
        caps = data.get("demo_capabilities") or []
        if not isinstance(caps, list):
            return []
        return [str(x).strip() for x in caps if str(x).strip()]

    def _phase3_gap(self, essence: str, audience: str,
                    capabilities: list[str],
                    competitors: list[CompetitorDetail]) -> list[Feature]:
        comp_block = "\n".join(
            f"- {c.name}:\n" + "\n".join(f"    • {kf}" for kf in c.key_features)
            for c in competitors
        ) or "(no competitors found)"
        cap_block = "\n".join(f"- {c}" for c in capabilities) or "(none)"
        user = GAP_USER_TMPL.format(
            essence=essence or "(unknown)",
            audience=audience or "(unknown)",
            capabilities=cap_block,
            competitors=comp_block,
        )
        data = self.llm.chat_json(GAP_SYS, user, web_search=False,
                                  temperature=0.3, max_tokens=4000)
        return [Feature(**_normalize_feature(f))
                for f in data.get("features", [])]

    # ---- public API ---------------------------------------------------------

    def run(self) -> AnalysisReport:
        listing, docs = self._build_input()
        capability_input = self._build_capability_input()

        # Phase 1: competitor discovery (with web search)
        d1 = self._phase1_competitors(listing, docs)
        essence = str(d1.get("essence", ""))
        audience = str(d1.get("audience", ""))
        competitors_detail = [_normalize_competitor(c)
                              for c in d1.get("competitors_detail", [])
                              if isinstance(c, dict)]
        log.info("Analyzer phase-1: competitors=%d", len(competitors_detail))

        # Phase 2: capability extraction (no web search, reads actual code)
        capabilities = self._phase2_capabilities(capability_input)
        log.info("Analyzer phase-2: demo_capabilities=%d", len(capabilities))

        # Phase 3: gap analysis (no web search)
        features = self._phase3_gap(essence, audience, capabilities,
                                    competitors_detail)
        log.info("Analyzer phase-3: features=%d (high=%d, partial=%d)",
                 len(features),
                 sum(1 for f in features if f.gap_severity == "high"),
                 sum(1 for f in features if f.in_demo == "partial"))

        return AnalysisReport(
            domain=str(d1.get("domain", "")),
            essence=essence,
            audience=audience,
            competitors=list(d1.get("competitors", [])),
            competitors_detail=competitors_detail,
            demo_capabilities=capabilities,
            features=features,
            ui_elements=list(d1.get("ui_elements", [])),
            raw_notes=str(d1.get("raw_notes", "")),
        )

    def run_cached(self, cache_path: Path,
                   *, use_cache: bool = True) -> tuple[AnalysisReport, bool]:
        """Wraps run() with on-disk caching keyed by fingerprint().

        Returns (report, was_cache_hit). Cache holds {fingerprint -> dict}.
        New fingerprints append; old ones aren't evicted (useful for repeated
        runs against the same demo with different providers)."""
        fp = self.fingerprint()
        cache_data: dict[str, Any] = {}
        if cache_path.is_file():
            try:
                cache_data = _json.loads(cache_path.read_text())
            except _json.JSONDecodeError:
                cache_data = {}
        if use_cache and fp in cache_data:
            d = cache_data[fp]
            report = AnalysisReport(
                domain=d.get("domain", ""),
                essence=d.get("essence", ""),
                audience=d.get("audience", ""),
                competitors=list(d.get("competitors", [])),
                competitors_detail=[_normalize_competitor(c)
                                    for c in d.get("competitors_detail", [])
                                    if isinstance(c, dict)],
                demo_capabilities=[str(x) for x in d.get("demo_capabilities", [])
                                   if str(x).strip()],
                features=[Feature(**_normalize_feature(f))
                          for f in d.get("features", [])],
                ui_elements=list(d.get("ui_elements", [])),
                raw_notes=d.get("raw_notes", ""),
            )
            return report, True
        report = self.run()
        cache_data[fp] = report.to_dict()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                _json.dumps(cache_data, ensure_ascii=False, indent=2)
            )
        except OSError as e:
            log.warning("analyzer cache write failed: %s", e)
        return report, False
