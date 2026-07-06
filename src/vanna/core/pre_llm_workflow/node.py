""""Node contract for pre-LLM worklfow execution."""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from .state import NodeResult, WorkflowState

@runtime_checkable
class WorkflowNode(Protocol):
    """
    Executable unit in a pre-LLM workflow graph.
    A node reads WorkflowState and returns a NodeResult containing only the
    changes it produced. The executor is responsible for merging NodeResult
    back into WorkflowState.
    """

    node_id: str

    async def run(self, state:WorkflowState) -> NodeResult:
        """Run the node and return its result."""