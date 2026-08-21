"""Node contract for Data-Discovering workflow execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .state import DataDiscover_NodeResult, DataDiscover_State


@runtime_checkable
class DataDiscover_Node(Protocol):
    """Executable unit in a Data-Discovering workflow graph.

    A node reads DataDiscover_State and returns a DataDiscover_NodeResult containing only the
    changes it produced. The executor is responsible for merging DataDiscover_NodeResult
    back into DataDiscover_State.
    """

    node_id: str

    async def run(self, state: DataDiscover_State) -> DataDiscover_NodeResult:
        """Run the node and return its result."""
        ...
