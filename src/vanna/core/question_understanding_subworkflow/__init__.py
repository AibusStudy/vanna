"""Question-Understanding workflow primitives for enriching an Agent request before the first LLM call.

This package does not handle slash-command workflows or post-LLM tool execution.
"""

from .edge import EdgeCondition, WorkflowEdge
from .executor import QuestionUnderstandSubWorkflowExecutor
from .graph import WorkflowGraph, WorkflowGraphError
from .node import WorkflowNode
from .protocols import (
    IntentClassifier,
    QuestionStructurer,
    StructuredQuestionValidator,
    TimeNormalizer,
)
from .state import (
    QuestUnderstand_NodeResult,
    NodeStatus,
    QuestUnderstand_RetryState,
    QuestUnderstand_FinalResult,
    QuestUnderstand_Input,
    QuestUnderstand_State,
    WorkflowStatus,
    apply_node_result,
)

__all__ = [
    "EdgeCondition",
    "IntentClassifier",
    "QuestUnderstand_NodeResult",
    "NodeStatus",
    "QuestionUnderstandSubWorkflowExecutor",
    "QuestionStructurer",
    "QuestUnderstand_RetryState",
    "StructuredQuestionValidator",
    "TimeNormalizer",
    "WorkflowEdge",
    "QuestUnderstand_FinalResult",
    "WorkflowGraph",
    "WorkflowGraphError",
    "QuestUnderstand_Input",
    "WorkflowNode",
    "QuestUnderstand_State",
    "WorkflowStatus",
    "apply_node_result",
]