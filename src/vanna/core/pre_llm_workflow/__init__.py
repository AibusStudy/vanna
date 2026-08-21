"""Pre-LLM workflow primitives for enriching an Agent request before the first LLM call.

This package does not handle slash-command workflows or post-LLM tool execution.
"""

from .edge import EdgeCondition, WorkflowEdge
from .executor import PreLlmWorkflowExecutor
from .graph import WorkflowGraph, WorkflowGraphError
from .node import WorkflowNode
from .protocols import (
    IntentClassifier,
    QuestionStructurer,
    StructuredQuestionValidator,
    TimeNormalizer,
)
from .state import (
    NodeResult,
    NodeStatus,
    RetryState,
    QuestionUnderstandSubWorkflowFinalResult,
    WorkflowInput,
    WorkflowState,
    WorkflowStatus,
    apply_node_result,
)

__all__ = [
    "EdgeCondition",
    "IntentClassifier",
    "NodeResult",
    "NodeStatus",
    "PreLlmWorkflowExecutor",
    "QuestionStructurer",
    "RetryState",
    "StructuredQuestionValidator",
    "TimeNormalizer",
    "WorkflowEdge",
    "WorkflowFinalResult",
    "WorkflowGraph",
    "WorkflowGraphError",
    "WorkflowInput",
    "WorkflowNode",
    "WorkflowState",
    "WorkflowStatus",
    "apply_node_result",
]