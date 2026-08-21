from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vanna.core.tool import ToolContext, ToolSchema


MainWorkflowStatus = Literal["success", "failed", "skipped"]
MainWorkflowStage = Literal[
    "question_understanding_subworkflow",
    "data_discovery",
    "sql_processing",
    "final",
]


@dataclass
class FallbackState:
    fb1_count: int = 0
    fb2_count: int = 0
    active_feedback: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fb1_used(self) -> bool:
        return self.fb1_count >= 1

    @property
    def fb2_used(self) -> bool:
        return self.fb2_count >= 1

    def can_use_fb1(self) -> bool:
        return self.fb1_count < 1

    def can_use_fb2(self) -> bool:
        return self.fb2_count < 1

    def mark_fb1(self, feedback: dict[str, Any]) -> None:
        self.fb1_count += 1
        self.active_feedback = feedback
        self.history.append(feedback)

    def mark_fb2(self, feedback: dict[str, Any]) -> None:
        self.fb2_count += 1
        self.active_feedback = feedback
        self.history.append(feedback)

    def snapshot(self) -> dict[str, Any]:
        return {
            "fb1_used": self.fb1_used,
            "fb2_used": self.fb2_used,
            "fb1_count": self.fb1_count,
            "fb2_count": self.fb2_count,
            "active_feedback": self.active_feedback,
        }


@dataclass
class SubworkflowState:
    status: MainWorkflowStatus = "skipped"
    current_node: str | None = None
    visited_nodes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retry_counts: dict[str, int] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "current_node": self.current_node,
            "visited_nodes": list(self.visited_nodes),
            "errors": list(self.errors),
            "retry_counts": dict(self.retry_counts),
        }


@dataclass(frozen=True)
class MainWorkflowInput:
    user_id: str
    conversation_id: str
    request_id: str
    original_message: str
    system_prompt: str | None
    tool_schemas: list[ToolSchema]
    tool_context: ToolContext
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MainWorkflowTurnState:
    turn_id: str
    original_question: str
    stage: MainWorkflowStage = "question_understanding_subworkflow"
    operation: str | None = None
    fallback_state: FallbackState = field(default_factory=FallbackState)

    subworkflows: dict[str, SubworkflowState] = field(
        default_factory=lambda: {
            "question_understanding_subworkflow": SubworkflowState(),
            "data_discovery": SubworkflowState(),
            "sql_processing": SubworkflowState(),
        }
    )

    question_understanding_subworkflow: dict[str, Any] | None = None
    data_discovery: dict[str, Any] | None = None
    sql_processing: dict[str, Any] | None = None

    result: dict[str, Any] = field(
        default_factory=lambda: {
            "message": None,
            "sql": None,
            "csv_name": None,
            "json_name": None,
            "status": None,
        }
    )
    def subworkflow(self, name: str) -> SubworkflowState:
        if name not in self.subworkflows:
            self.subworkflows[name] = SubworkflowState()
        return self.subworkflows[name]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "original_question": self.original_question,
            "stage": self.stage,
            "operation": self.operation,
            "fallback_state": {
                **self.fallback_state.snapshot(),
                "history": list(self.fallback_state.history),
            },
            "subworkflows": {
                name: subworkflow.to_metadata()
                for name, subworkflow in self.subworkflows.items()
            },
            "question_understanding_subworkflow": self.question_understanding_subworkflow,
            "data_discovery": self.data_discovery,
            "sql_processing": self.sql_processing,
            "result": dict(self.result),
        }

