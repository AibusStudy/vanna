"""Executor for Data-Discovering workflow graphs."""

from __future__ import annotations

import logging
from typing import List, Optional

from .edge import DataDiscover_Edge
from .graph import WorkflowGraph
from .state import (
    DataDiscover_FinalResult,
    DataDiscover_Input,
    DataDiscover_NodeResult,
    DataDiscover_State,
    WorkflowStatus,
    apply_node_result,
)

logger = logging.getLogger(__name__)


class DataDiscoverSubWorkflowExecutor:
    """Runs metadata and few-shot discovery nodes before the Agent LLM loop."""

    def __init__(
        self,
        graph: WorkflowGraph,
        *,
        max_steps: int = 5,
        retry_limit: int = 1,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")
        if retry_limit < 0:
            raise ValueError("retry_limit must be greater than or equal to 0.")

        self.graph = graph
        self.max_steps = max_steps
        self.retry_limit = retry_limit

    async def run(self, workflow_input: DataDiscover_Input) -> DataDiscover_FinalResult:
        self.graph.validate()

        state = DataDiscover_State(input=workflow_input)
        current_node_id = self.graph.start_node_id

        if current_node_id is None:
            return self._finalize(
                state,
                "failed",
                ["Workflow start node is missing."],
            )

        for _ in range(self.max_steps):
            state.set_node_output("current_node", current_node_id)

            node = self.graph.get_node(current_node_id)
            result = await node.run(state)
            apply_node_result(state, current_node_id, result)

            if result.status == "skipped":
                return self._finalize(state, "skipped")

            if result.status == "finish":
                return self._finalize(state, "success")

            if result.status == "retry":
                attempts = state.retry.increment(current_node_id)
                if attempts <= self.retry_limit:
                    retry_edge = await self._select_next_edge(
                        state,
                        current_node_id,
                        result,
                        require_condition=True,
                    )
                    if retry_edge is not None:
                        current_node_id = retry_edge.target_node_id
                    continue

                return self._finalize(
                    state,
                    "failed",
                    [f"Retry limit exceeded for node: {current_node_id}"],
                )

            next_edge = await self._select_next_edge(
                state,
                current_node_id,
                result,
            )

            if next_edge is not None:
                current_node_id = next_edge.target_node_id
                continue

            if current_node_id in self.graph.end_node_ids:
                if result.status == "failed":
                    return self._finalize(state, "failed")
                return self._finalize(state, "success")

            if result.status == "failed":
                return self._finalize(state, "failed")

            return self._finalize(
                state,
                "failed",
                [f"No matching edge from node: {current_node_id}"],
            )

        return self._finalize(
            state,
            "failed",
            [f"Max workflow steps exceeded: {self.max_steps}"],
        )

    async def _select_next_edge(
        self,
        state: DataDiscover_State,
        source_node_id: str,
        last_result: DataDiscover_NodeResult,
        *,
        require_condition: bool = False,
    ) -> Optional[DataDiscover_Edge]:
        for edge in self.graph.get_edges(source_node_id):
            if require_condition and edge.condition is None:
                continue
            if await edge.matches(state, last_result):
                return edge
        return None

    def _finalize(
        self,
        state: DataDiscover_State,
        status: WorkflowStatus,
        extra_errors: Optional[List[str]] = None,
    ) -> DataDiscover_FinalResult:
        errors = list(state.errors)
        if extra_errors:
            errors.extend(extra_errors)
        final_status = "failed" if status == "success" and errors and state.metadata_output is None else status

        logger.debug(
            "Data-Discovering workflow finalized",
            extra={
                "status": final_status,
                "warnings": state.warnings,
                "errors": errors,
                "retry_counts": state.retry_counts,
            },
        )

        return DataDiscover_FinalResult(
            status=final_status,
            metadata_output=state.metadata_output,
            fewshot_output=state.fewshot_output,
            warnings=list(state.warnings),
            errors=errors,
            retry_counts=state.retry_counts,
            failed_node_id=state.failed_node_id,
            failure_type=state.failure_type,
            failure_detail=state.failure_detail,
        )




