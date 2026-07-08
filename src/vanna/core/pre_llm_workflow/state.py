from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


NodeStatus = Literal["success", "failed", "retry", "finish", "skipped"]
WorkflowStatus = Literal["success", "failed", "skipped"]


@dataclass(frozen=True)
class WorkflowInput:
    user_id: str
    conversation_id: str
    request_id: str
    original_message: str
    system_prompt: Optional[str]
    tool_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryState:
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
class NodeResult:
    status: NodeStatus
    output: Any = None
    routing_intent: Optional[str] = None
    structured_question: Optional[Dict[str, Any]] = None
    debug_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class WorkflowState:
    """Mutable internal state used only while executing the workflow."""

    input: WorkflowInput

    routing_intent: Optional[str] = None
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    visited_nodes: List[str] = field(default_factory=list)
    last_node_result: Optional[NodeResult] = None

    structured_question: Optional[Dict[str, Any]] = None

    debug_metadata: Dict[str, Any] = field(default_factory=dict)

    retry: RetryState = field(default_factory=RetryState)
    errors: List[str] = field(default_factory=list)

    @property
    def original_message(self) -> str:
        return self.input.original_message

    @property
    def retry_counts(self) -> Dict[str, int]:
        return self.retry.to_dict()

    def set_node_output(self, node_id: str, output: Any) -> None:
        self.node_outputs[node_id] = output

    def get_node_output(self, node_id: str) -> Any:
        return self.node_outputs.get(node_id)

    def add_error(self, error: str) -> None:
        self.errors.append(error)


@dataclass(frozen=True)
class WorkflowFinalResult:
    status: WorkflowStatus
    intent: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "intent": self.intent,
            "structured_output": self.structured_output,
            "errors": list(self.errors),
            "retry_counts": dict(self.retry_counts),
        }


def apply_node_result(
    state: WorkflowState,
    node_id: str,
    result: NodeResult,
) -> WorkflowState:
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
