"""Data-Discovering workflow primitives for enriching an Agent request before the first LLM call.

This package does not handle slash-command workflows or post-LLM tool execution.
"""

from .edge import EdgeCondition, DataDiscover_Edge
from .executor import DataDiscoverSubWorkflowExecutor
from .graph import WorkflowGraph, WorkflowGraphError
from .node import WorkflowNode

from .state import (
    DataDiscover_NodeResult,
    DataDiscover_FinalResult,
    DataDiscover_Input,
    DataDiscover_RetryState,
    DataDiscover_State,
    NodeStatus,
    WorkflowStatus,
    apply_node_result,
)

__all__ = [
    "EdgeCondition",
    "WorkflowEdge",
    "QuestUnderstand_FinalResult",
    "WorkflowGraph",
    "WorkflowGraphError",
    "WorkflowNode",
    "WorkflowStatus",
    "apply_node_result",
    "DataDiscoverSubWorkflowExecutor",
    "DataDiscover_NodeResult",
    "DataDiscover_FinalResult",
    "DataDiscover_Input",
    "DataDiscover_RetryState",
    "DataDiscover_State",
    "NodeStatus",
]