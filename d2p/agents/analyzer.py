"""Analyzer agent — reads the demo, extracts essence/audience/features.

The Analyzer does NO domain detection in code. Its prompt asks the LLM to:
- decide what the project IS (domain + essence + audience),
- search the web for mature competitors in that domain,
- distil concrete features that would improve the demo WITHOUT changing
  its essence.

Adding a new demo type (audio, blockchain, robotics) requires zero
code changes here.

Caching: results are persisted under <target>/.d2p/analysis_cache.json
keyed by a fingerprint of (listing + key docs + system prompt + model
identity). Second run against an unchanged codebase is an instant cache
hit. Pass `--no-cache-analysis` (or `use_cache=False` on `run_cached`)
to force a fresh run.
"""
from __future__ import annotations

import hashlib
import json as _json
import logging
from pathlib import Path
from typing import Any

from ..fs import Sandbox
from ..providers.base import LLMProvider
from ..models import AnalysisReport, Feature

log = logging.getLogger("d2p.agents.analyzer")


ANALYZER_SYS = """You are the Analyzer agent in a Demo-to-Product (d2p) pipeline.
Your job:

1. Read the demo (listing + key files) and identify TWO things separately:
   - "domain": the problem area (e.g. "social deduction game", "speech-to-text")
   - "essence": the demo's CORE NATURE that must be preserved across iterations.
     The essence captures what kind of artifact this demo IS — who its real
     audience is, and what makes it distinct from typical products in the
     same domain.
   - "audience": one short phrase, e.g. "LLM agents", "developers via API",
     "research notebook users", "humans on a web UI", "CLI power-users".

   Examples:
     * werewolf demo where 6 LLM-driven players debate each other
         -> domain: "Werewolf / social deduction"
         -> essence: "an Agent-vs-Agent simulation harness where LLM players
            debate, vote and reason; humans are spectators, not players"
         -> audience: "LLM agents (humans only observe / analyze)"
     * a Whisper-based offline transcriber CLI
         -> essence: "an offline batch-processing CLI; not a real-time web app"
         -> audience: "CLI users / scripted pipelines"

2. Search the web for 3-5 MATURE COMPETITOR PRODUCTS in the same DOMAIN.

3. From competitors, extract concrete features and UI elements that would
   improve a PRODUCT BUILT FROM THIS DEMO **without changing its essence**.
   - If the demo is agent-facing, do NOT propose human-multiplayer features
     like lobby codes, voice chat, ranked ladders — those would change the
     essence into a different product.
   - DO propose agent-facing analogues: e.g. instead of "voice chat", suggest
     "structured wolf-private channel for inter-agent reasoning"; instead of
     "leaderboard", suggest "agent performance benchmark dashboard".
   - Be concrete — "login with Google" not "auth".

4. Output STRICT JSON only — no markdown, no commentary.
"""

ANALYZER_USER_TMPL = """Demo project files:
{listing}

Top-level documents (truncated):
{docs}

Return a JSON object with this exact shape:
{{
  "domain": "<one sentence>",
  "essence": "<one or two sentences — the demo's core nature that must NOT change>",
  "audience": "<one short phrase>",
  "competitors": ["<product name + 1-line desc>", ...],
  "features": [
    {{"name": "...", "category": "backend|frontend|ux|ops|docs", "description": "...", "source": "<competitor name>"}}
  ],
  "ui_elements": ["...", ...],
  "raw_notes": "<short freeform notes, max 500 chars>"
}}
Provide 8-15 features. Skip features that would change the audience or essence.
Skip anything the demo clearly already has.
"""


def _normalize_feature(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(f.get("name", "")).strip() or "unnamed",
        "category": str(f.get("category", "other")).strip().lower(),
        "description": str(f.get("description", "")).strip(),
        "source": str(f.get("source", "")).strip(),
    }


class Analyzer:
    def __init__(self, llm: LLMProvider, sandbox: Sandbox) -> None:
        self.llm = llm
        self.sandbox = sandbox

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
        """Construct (listing, docs) — the exact context fed to the LLM.
        Extracted so cache fingerprinting can hash the same bytes the model
        actually sees, instead of approximating with mtime/size."""
        listing = "\n".join(self.sandbox.listing(max_entries=120))
        docs = self._gather_docs()
        return listing, docs

    def fingerprint(self) -> str:
        """Stable hash of the Analyzer's input. Same bytes → same fingerprint,
        which means we can safely reuse a previous run's output.

        The fingerprint folds in (a) the project listing, (b) the doc files
        Analyzer reads, AND (c) the system prompt + provider+model — if the
        prompt or model changes, you want a fresh analysis. Web-search
        results from the LLM aren't deterministic so cache hits trade some
        freshness for big speed/cost wins; pass `--no-cache-analysis` to
        force a fresh run."""
        listing, docs = self._build_input()
        h = hashlib.sha256()
        h.update(b"d2p-analyzer-v1\n")
        h.update(ANALYZER_SYS.encode("utf-8"))
        h.update(b"\n--listing--\n")
        h.update(listing.encode("utf-8"))
        h.update(b"\n--docs--\n")
        h.update(docs.encode("utf-8"))
        h.update(b"\n--model--\n")
        h.update(getattr(self.llm, "name", "?").encode("utf-8"))
        return h.hexdigest()[:16]

    def run(self) -> AnalysisReport:
        listing, docs = self._build_input()
        user = ANALYZER_USER_TMPL.format(listing=listing, docs=docs)
        data = self.llm.chat_json(ANALYZER_SYS, user, web_search=True,
                                  temperature=0.3, max_tokens=6000)
        features = [Feature(**_normalize_feature(f)) for f in data.get("features", [])]
        return AnalysisReport(
            domain=data.get("domain", ""),
            essence=data.get("essence", ""),
            audience=data.get("audience", ""),
            competitors=list(data.get("competitors", [])),
            features=features,
            ui_elements=list(data.get("ui_elements", [])),
            raw_notes=data.get("raw_notes", ""),
        )

    def run_cached(self, cache_path: Path,
                   *, use_cache: bool = True) -> tuple[AnalysisReport, bool]:
        """Wraps run() with on-disk caching keyed by fingerprint().

        Returns (report, was_cache_hit). The cache file holds a single dict
        keyed by fingerprint -> serialized AnalysisReport. New fingerprints
        get appended without evicting older entries (useful for repeated
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
