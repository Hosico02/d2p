from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Feature:
    name: str
    category: str          # backend | frontend | ux | ops | docs | other
    description: str
    source: str = ""       # competitor or doc link the feature was inferred from

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    domain: str
    essence: str = ""           # the demo's CORE NATURE that must not change
    audience: str = ""          # who/what consumes it (e.g. 'LLM agents', 'humans on web')
    competitors: list[str] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    ui_elements: list[str] = field(default_factory=list)
    raw_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "essence": self.essence,
            "audience": self.audience,
            "competitors": self.competitors,
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
