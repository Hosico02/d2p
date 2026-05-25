"""Verifier agent — placeholder.

Full implementation deferred to the spec in:
  ../demo2project/docs/superpowers/specs/2026-05-22-d2p-verify-agent-design.md

This stub exists to wire HubClient → standards pull at construction time,
so the integration is ready when the real Verifier body lands.
"""
from __future__ import annotations
import typing as t
from dataclasses import dataclass

BAKED_STANDARDS = """- tests_run_and_pass
- error_envelope_present
- readme_cmd_matches_manifest
"""

SYSTEM_PROMPT_TEMPLATE = """You are an independent verifier.

Standards for this archetype:
{standards}

(Full prompt body deferred to the design spec.)
"""


@dataclass
class VerifyResult:
    verdict: str
    raw_response: str


class Verifier:
    def __init__(self, llm_client: t.Any = None, system_root: t.Any = None,
                 hub_client: t.Any = None):
        self.llm = llm_client
        self.system_root = system_root
        self.hub = hub_client

    def verify(self, project_path: t.Any, claim: t.Any = None,
               pre_evidence: t.Any = None, previous_results: t.Any = None,
               archetype: str = "unknown") -> VerifyResult:
        standards = (self.hub.pull_standards(archetype)
                     if self.hub else BAKED_STANDARDS)
        _system_prompt = SYSTEM_PROMPT_TEMPLATE.format(standards=standards)
        # Real implementation deferred. For now we raise so callers don't
        # silently get a fake verdict.
        raise NotImplementedError(
            "Verifier body not yet implemented. See "
            "docs/superpowers/specs/2026-05-22-d2p-verify-agent-design.md"
        )
