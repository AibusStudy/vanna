from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from vanna.core.components import UiComponent

NodeStatus = Literal["success", "failed", "retry", "finish", "skipped"]
WorkflowStatus = Literal["success", "failed", "skipped", "fallback"]


@dataclass(frozen=True)
class DataDiscover_Input:
    status: str
    intent: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class DataDiscover_RetryState:
    attempts_by_node: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> int:
        next_count = self.attempts_by_node.get(node_id, 0) + 1
        self.attempts_by_node[node_id] = next_count
        return next_count

    def get_attempts(self, node_id: str) -> int:
        return self.attempts_by_node.get(node_id, 0)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.attempts_by_node)


@dataclass
class DataDiscover_NodeResult:
    status: NodeStatus
    output: Any = None
    metadata_output: Optional[Dict[str, Any]] = None
    fewshot_output: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DataDiscover_State:
    input: DataDiscover_Input
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata_output: Optional[Dict[str, Any]] = None
    fewshot_output: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    retry: DataDiscover_RetryState = field(default_factory=DataDiscover_RetryState)


@dataclass(frozen=True)
class DataDiscover_FinalResult:
    status: WorkflowStatus
    metadata_output: Optional[Dict[str, Any]] = None
    fewshot_output: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "metadata_output": self.metadata_output,
            "fewshot_output": self.fewshot_output,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "retry_counts": dict(self.retry_counts),
        }


def apply_node_result(
    state: DataDiscover_State,
    node_id: str,
    result: DataDiscover_NodeResult,
) -> DataDiscover_State:
    state.last_node_result = result

    if result.output is not None:
        state.set_node_output(node_id, result.output)

    if result.routing_intent is not None:
        state.routing_intent = result.routing_intent

    if result.structured_question is not None:
        state.structured_question = result.structured_question

    state.debug_metadata.update(result.debug_metadata)

    if result.error:
        state.add_error(result.error)

    return state