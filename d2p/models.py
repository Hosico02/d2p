from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Feature:
    name: str
    category: str          # backend | frontend | ux | ops | docs | other
    description: str
    source: str = ""       # competitor or doc link the feature was inferred from
    # Gap-matrix fields (populated by Analyzer phase-3 gap analysis):
    in_demo: str = ""              # "missing" | "partial" | "present" | ""
    evidence_in_demo: str = ""     # short citation (file:line or quote)
    gap_severity: str = ""         # "low" | "medium" | "high" | ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompetitorDetail:
    name: str
    key_features: list[str] = field(default_factory=list)
    source_url: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    domain: str
    essence: str = ""           # the demo's CORE NATURE that must not change
    audience: str = ""          # who/what consumes it (e.g. 'LLM agents', 'humans on web')
    competitors: list[str] = field(default_factory=list)
    competitors_detail: list[CompetitorDetail] = field(default_factory=list)
    demo_capabilities: list[str] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    ui_elements: list[str] = field(default_factory=list)
    raw_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "essence": self.essence,
            "audience": self.audience,
            "competitors": self.competitors,
            "competitors_detail": [c.to_dict() for c in self.competitors_detail],
            "demo_capabilities": self.demo_capabilities,
            "features": [f.to_dict() for f in self.features],
            "ui_elements": self.ui_elements,
            "raw_notes": self.raw_notes,
        }


@dataclass
class Task:
    id: str
    title: str
    rationale: str
    target_files: list[str]
    instructions: str
    priority: int = 5            # 1 = highest
    category: str = "feature"    # feature | bugfix | ux | docs | infra
    status: str = "pending"      # pending | in_progress | done | failed | skipped
    notes: str = ""
    forbidden_files: list[str] = field(default_factory=list)  # never write these
    # ^ used by QA-fix tasks to protect the test file from being edited
    # Cross-iter tier escalation. Tasks carried over from a prior iter
    # arrive with tier_idx > 0; the orchestrator picks the matching ladder
    # provider when dispatching. attempts records every prior pass so
    # diagnostics can show the trail.
    tier_idx: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanResult:
    iteration: int
    tasks: list[Task]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"iteration": self.iteration, "rationale": self.rationale,
                "tasks": [t.to_dict() for t in self.tasks]}


@dataclass
class ExecutionResult:
    task_id: str
    status: str           # done | failed | skipped
    summary: str
    files_changed: list[str] = field(default_factory=list)
    error: str = ""
    # If the task was rolled back due to a baseline regression, these are
    # the module/test names that went green→red. Stored on the task's
    # carry-over attempts so the next retry sees "your previous patch
    # broke X, Y — avoid those" in its prompt.
    regressed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
