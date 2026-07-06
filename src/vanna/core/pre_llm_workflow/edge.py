"""Edge contracts for pre-LLM workflow graphs.""" 

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from .state import NodeResult, WorkflowState

@runtime_checkable
class EdgeCondition(Protocol):
    """Conditino used to decide whether a workflow edge should be followed."""

    async def evaluate(self, state:WorkflowState, last_node_result: NodeResult, ) -> bool:
        """ Return True When this dege should be selected."""


@dataclass(frozen=True)
class WorkflowEdge:
    """Directed edge between two workflow nodes."""

    source_node_id: str
    target_node_id: str
    condition: Optional[EdgeCondition] = None
    label: Optional[str] = None
    
    async def matches(self, state:WorkflowState, last_node_result = NodeResult, ) -> bool:
        if self.condition is None:
            return True
        
        return await self.condition.evaluate(state, last_node_result) 