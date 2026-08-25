"""Edge contracts for Data-Discovering workflow graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from .state import DataDiscover_NodeResult, DataDiscover_State


@runtime_checkable
class EdgeCondition(Protocol):
    """Condition used to decide whether a workflow edge should be followed."""

    async def evaluate(
        self,
        state: DataDiscover_State,
        last_node_result: DataDiscover_NodeResult,
    ) -> bool:
        """이 edge를 선택해야할 때 True를 반환한다."""
        ...


@dataclass(frozen=True)
class DataDiscover_Edge:
    """두 워크플로우 노드 사이에 지정된 엣지"""

    source_node_id: str
    target_node_id: str
    condition: Optional[EdgeCondition] = None
    label: Optional[str] = None

    async def matches(
        self,
        state: DataDiscover_State,
        last_node_result: DataDiscover_NodeResult,
    ) -> bool:
        if self.condition is None:
            return True

        return await self.condition.evaluate(state, last_node_result)
