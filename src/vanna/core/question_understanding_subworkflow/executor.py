"""Executor for Question-Understanding workflow graphs."""

from __future__ import annotations

import logging
from typing import List, Optional

from .edge import QuestionUnderstand_Edge
from .graph import WorkflowGraph
from .state import (
    QuestUnderstand_NodeResult,
    QuestUnderstand_FinalResult,
    QuestUnderstand_Input,
    QuestUnderstand_State,
    WorkflowStatus,
    apply_node_result,
)

logger = logging.getLogger(__name__)


class QuestionUnderstandSubWorkflowExecutor:
    """Runs a Question-Understanding workflow graph from the start node to a final result."""

    def __init__(
        self,
        graph: WorkflowGraph,
        *,
        max_steps: int = 9,
        retry_limit: int = 1,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")

        if retry_limit < 0:
            raise ValueError("retry_limit must be greater than or equal to 0.")

        self.graph = graph
        self.max_steps = max_steps
        self.retry_limit = retry_limit

    async def run(self, workflow_input: QuestUnderstand_Input) -> QuestUnderstand_FinalResult:
        self.graph.validate()

        state = QuestUnderstand_State(input=workflow_input)
        current_node_id = self.graph.start_node_id

        if current_node_id is None:
            return self._finalize(
                state,
                "failed",
                ["Workflow start node is missing."],
            )

        for _ in range(self.max_steps):
            state.visited_nodes.append(current_node_id)

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
                    # conditional edge
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
        state: QuestUnderstand_State,
        source_node_id: str,
        last_result: QuestUnderstand_NodeResult,
        *,
        require_condition: bool = False,
    ) -> Optional[QuestionUnderstand_Edge]:
        for edge in self.graph.get_edges(source_node_id):
            if require_condition and edge.condition is None:
                continue
            if await edge.matches(state, last_result):
                return edge

        return None

    def _finalize(
        self,
        state: QuestUnderstand_State,
        status: WorkflowStatus,
        extra_errors: Optional[List[str]] = None,
    ) -> QuestUnderstand_FinalResult:
        errors = list(state.errors)

        if extra_errors:
            errors.extend(extra_errors)

        if state.debug_metadata:
            logger.debug(
                "Question-Understanding workflow debug metadata",
                extra={"debug_metadata": state.debug_metadata},
            )

        return QuestUnderstand_FinalResult(
            status=status,
            intent=self._extract_intent(state),
            structured_output=state.structured_question,
            errors=errors,
            retry_counts=state.retry_counts,
            failed_node_id=state.failed_node_id,
            failure_type=state.failure_type or self._infer_failure_type(errors),
            failure_detail=state.failure_detail,
        )

    def _extract_intent(self, state: QuestUnderstand_State) -> Optional[str]:
        if state.routing_intent:
            return state.routing_intent

        if state.structured_question:
            intent = state.structured_question.get("intent")
            if isinstance(intent, str):
                return intent

        return None

    @staticmethod
    def _infer_failure_type(errors: List[str]) -> Optional[str]:
        known_failure_types = {
            "intent_classification_failed",
            "time_normalization_failed",
            "time_metadata_validation_failed",
            "question_structuring_failed",
            "search_queries_generation_failed",
            "json_validation_failed",
            "json_validation_retry_exceeded",
        }
        joined_errors = "\n".join(str(error) for error in errors)
        for failure_type in known_failure_types:
            if failure_type in joined_errors:
                return failure_type
        return None

