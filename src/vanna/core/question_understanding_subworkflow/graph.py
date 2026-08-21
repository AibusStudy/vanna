"""Graph container for Question-Understanding workflow nodes and edges."""

from __future__ import annotations
from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Optional, Set

from .edge import EdgeCondition, QuestionUnderstand_Edge
from .node import QuestionUnderstand_Node


class WorkflowGraphError(ValueError):
    """Raised when a Question-Understanding workflow graph is invalid."""


class WorkflowGraph:
    """Node-edge graph executed by QuestionUnderstandSubWorkflowExecutor."""

    def __init__(self) -> None:
        self._nodes: Dict[str, QuestionUnderstand_Edge] = {}
        self._edges_by_source: DefaultDict[str, List[QuestionUnderstand_Edge]] = defaultdict(list)
        self.start_node_id: Optional[str] = None
        self.end_node_ids: Set[str] = set()

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes.keys())

    def add_node(
        self,
        node: QuestionUnderstand_Node,
        *,
        start: bool = False,
        end: bool = False,
    ) -> "WorkflowGraph":
        if node.node_id in self._nodes:
            raise WorkflowGraphError(f"Duplicate workflow node: {node.node_id}")

        self._nodes[node.node_id] = node

        if start:
            self.set_start(node.node_id)

        if end:
            self.add_end(node.node_id)

        return self

    def set_start(self, node_id: str) -> "WorkflowGraph":
        self._ensure_node_exists(node_id)
        self.start_node_id = node_id
        return self

    def add_end(self, node_id: str) -> "WorkflowGraph":
        self._ensure_node_exists(node_id)
        self.end_node_ids.add(node_id)
        return self

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        condition: Optional[EdgeCondition] = None,
        label: Optional[str] = None,
    ) -> "WorkflowGraph":
        self._ensure_node_exists(source_node_id)
        self._ensure_node_exists(target_node_id)

        self._edges_by_source[source_node_id].append(
            QuestionUnderstand_Edge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                condition=condition,
                label=label,
            )
        )
        return self

    def get_node(self, node_id: str) -> QuestionUnderstand_Node:
        self._ensure_node_exists(node_id)
        return self._nodes[node_id]

    def get_edges(self, source_node_id: str) -> List[QuestionUnderstand_Edge]:
        self._ensure_node_exists(source_node_id)
        return list(self._edges_by_source.get(source_node_id, []))

    def validate(self) -> None:
        if not self._nodes:
            raise WorkflowGraphError("Workflow graph has no nodes.")

        if self.start_node_id is None:
            raise WorkflowGraphError("Workflow graph has no start node.")

        self._ensure_node_exists(self.start_node_id)

        if not self.end_node_ids:
            raise WorkflowGraphError("Workflow graph has no end nodes.")

        for end_node_id in self.end_node_ids:
            self._ensure_node_exists(end_node_id)

        for source_node_id, edges in self._edges_by_source.items():
            self._ensure_node_exists(source_node_id)

            for edge in edges:
                self._ensure_node_exists(edge.target_node_id)

        for node_id in self._nodes:
            if node_id not in self.end_node_ids and not self.get_edges(node_id):
                raise WorkflowGraphError(
                    f"Non-end workflow node has no outgoing edges: {node_id}"
                )

        unreachable_node_ids = self._find_unreachable_nodes()
        if unreachable_node_ids:
            raise WorkflowGraphError(
                "Workflow graph has unreachable nodes: "
                + ", ".join(sorted(unreachable_node_ids))
            )

    def _find_unreachable_nodes(self) -> Set[str]:
        if self.start_node_id is None:
            return self.node_ids

        visited_node_ids: Set[str] = set()
        queue = deque([self.start_node_id])

        while queue:
            node_id = queue.popleft()

            if node_id in visited_node_ids:
                continue

            visited_node_ids.add(node_id)

            for edge in self.get_edges(node_id):
                if edge.target_node_id not in visited_node_ids:
                    queue.append(edge.target_node_id)

        return self.node_ids - visited_node_ids

    def _ensure_node_exists(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise WorkflowGraphError(f"Unknown workflow node: {node_id}")
