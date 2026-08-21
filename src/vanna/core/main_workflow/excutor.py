"""MainWorkflow executor for turn state orchestration."""

from __future__ import annotations

import json
import logging
from typing import Any

from vanna.core.question_understanding_subworkflow import QuestUnderstand_Input
from vanna.core.data_discovering_subworkflow import DataDiscover_Input

from .state import MainWorkflowInput, MainWorkflowTurnState

logger = logging.getLogger(__name__)


def _log_turn_state(event: str, state: MainWorkflowTurnState) -> None:
    logger.info(
        "[main_workflow.turn_state] %s\n%s",
        event,
        json.dumps(state.to_metadata(), ensure_ascii=False, indent=2),
    )


class MainWorkflowExecutor:
    """Runs Question-Understanding/data-discovery subflows and stores their turn state.

    The Agent still owns the final LLM/tool-calling loop. This executor prepares
    and records the state that the enhancer and the LLM request can consume.
    """

    def __init__(
        self,
        question_understanding_executor=None,
        data_discovery_executor=None,
        #sql_processing_executor=None,
        router=None,
        question_understanding_subworkflow_executor=None,
    ):
        self.question_understanding_executor = question_understanding_executor
        self.data_discovery_executor = data_discovery_executor
        #self.sql_processing_executor = sql_processing_executor
        self.router = router

    async def run(self, input: MainWorkflowInput) -> MainWorkflowTurnState:
        state = MainWorkflowTurnState(
            turn_id=input.request_id,
            original_question=input.original_message,
        )
        _log_turn_state("initialized", state)

        await self._run_question_understanding_subworkflow(input, state)
        _log_turn_state("question_understanding_subworkflow_saved", state)
        await self._run_data_discovery(input, state)
        _log_turn_state("data_discovery_saved", state)

        # sql_processing is handled by Agent's existing LLM/tool-calling loop.
        state.stage = "sql_processing"
        state.operation = "agent_llm_tool_loop_ready"
        _log_turn_state("sql_processing_ready", state)
        return state

    async def _run_question_understanding_subworkflow(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> None:
        subflow_state = state.subworkflow("question_understanding_subworkflow")
        state.stage = "question_understanding_subworkflow"
        state.operation = "run_question_understanding_subworkflow"

        if self.question_understanding_executor is None:
            subflow_state.status = "skipped"
            return

        try:
            workflow_input = QuestUnderstand_Input(
                user_id=input.user_id,
                conversation_id=input.conversation_id,
                request_id=input.request_id,
                original_message=input.original_message,
                system_prompt=input.system_prompt,
                tool_names=[tool.name for tool in input.tool_schemas],
                metadata={
                    **input.metadata,
                    "fallback_state": state.fallback_state.snapshot(),
                },
            )
            result = await self.question_understanding_executor.run(workflow_input)
            state.question_understanding_subworkflow = result.to_metadata()
            _log_turn_state("question_understanding_subworkflow_result_assigned", state)
            subflow_state.status = result.status
            subflow_state.errors = list(result.errors)
            subflow_state.retry_counts = dict(result.retry_counts)
        except Exception as exc:
            subflow_state.status = "failed"
            subflow_state.errors.append(str(exc))
            state.question_understanding_subworkflow = {
                "status": "failed",
                "errors": [f"question_understanding_subworkflow failed: {str(exc)}"],
            }
            _log_turn_state("question_understanding_subworkflow_error_assigned", state)

    async def _run_data_discovery(
        self,
        question_result: DataDiscover_Input,
        state: MainWorkflowTurnState,
    ) -> None:
        question_result = state.question_understanding_subworkflow
        subflow_state = state.subworkflow("data_discovery")
        state.stage = "data_discovery"
        state.operation = "run_data_discovery"

        if self.data_discovery_executor is None:
            subflow_state.status = "skipped"
            return

        try:
            workflow_input = DataDiscover_Input(
                status=question_result.get("status"),
                intent=question_result.get("intent"),
                structured_output=question_result.get("structured_output"),
                errors=question_result.get("errors", []),
                retry_counts=question_result.get("retry_counts", {}),
            )
            result = await self.data_discovery_executor.run(workflow_input)
            state.data_discovery = self._to_metadata(result)
            _log_turn_state("data_discovery_result_assigned", state)
            subflow_state.status = getattr(result, "status", "success")
            subflow_state.errors = list(getattr(result, "errors", []))
            subflow_state.retry_counts = dict(getattr(result, "retry_counts", {}))
        except Exception as exc:
            subflow_state.status = "failed"
            subflow_state.errors.append(str(exc))
            state.data_discovery = {
                "status": "failed",
                "errors": [f"data_discovery failed: {str(exc)}"],
            }
            _log_turn_state("data_discovery_error_assigned", state)

    @staticmethod
    def _to_metadata(result: Any) -> dict[str, Any]:
        if hasattr(result, "to_metadata"):
            return result.to_metadata()
        if isinstance(result, dict):
            return result
        return {"status": getattr(result, "status", "success")}
