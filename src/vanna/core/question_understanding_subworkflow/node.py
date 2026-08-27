"""Node contract for Question-Understanding workflow execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from .state import QuestUnderstand_NodeResult, QuestUnderstand_State


@runtime_checkable
class QuestionUnderstand_Node(Protocol):
    """Executable unit in a Question-Understanding workflow graph.

    A node reads QuestUnderstand_State and returns a QuestUnderstand_NodeResult containing only the
    changes it produced. The executor is responsible for merging QuestUnderstand_NodeResult
    back into QuestUnderstand_State.
    """

    node_id: str

    async def run(self, state: QuestUnderstand_State) -> QuestUnderstand_NodeResult:
        """Run the node and return its result."""
        ...
