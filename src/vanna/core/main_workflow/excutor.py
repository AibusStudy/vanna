"""MainWorkflow executor for turn state orchestration."""

from __future__ import annotations

import logging
from typing import Any

from vanna.core.pre_llm_workflow import WorkflowInput

from .state import MainWorkflowInput, MainWorkflowTurnState

logger = logging.getLogger(__name__)


class MainWorkflowExecutor:
    """Runs pre-LLM/data-discovery subflows and stores their turn state.

    The Agent still owns the final LLM/tool-calling loop. This executor prepares
    and records the state that the enhancer and the LLM request can consume.
    """

    def __init__(
        self,
        question_understanding_executor=None,
        #data_discovery_executor=None,
        #sql_processing_executor=None,
        router=None,
        pre_llm_workflow_executor=None,
    ):
        self.question_understanding_executor = (
            question_understanding_executor or pre_llm_workflow_executor
        )
        #self.data_discovery_executor = data_discovery_executor
        #self.sql_processing_executor = sql_processing_executor
        self.router = router

    async def run(self, input: MainWorkflowInput) -> MainWorkflowTurnState:
        state = MainWorkflowTurnState(
            turn_id=input.request_id,
            original_question=input.original_message,
        )

        await self._run_pre_llm_workflow(input, state)
        await self._run_data_discovery(input, state)

        # sql_processing is handled by Agent's existing LLM/tool-calling loop.
        state.stage = "sql_processing"
        state.operation = "agent_llm_tool_loop_ready"
        return state

    async def _run_pre_llm_workflow(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> None:
        subflow_state = state.subworkflow("pre_llm_workflow")
        state.stage = "pre_llm_workflow"
        state.operation = "run_pre_llm_workflow"

        if self.question_understanding_executor is None:
            subflow_state.status = "skipped"
            return

        try:
            workflow_input = WorkflowInput(
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
            state.pre_llm_workflow = result.to_metadata()
            subflow_state.status = result.status
            subflow_state.errors = list(result.errors)
            subflow_state.retry_counts = dict(result.retry_counts)
        except Exception as exc:
            subflow_state.status = "failed"
            subflow_state.errors.append(str(exc))
            state.pre_llm_workflow = {
                "status": "failed",
                "errors": [f"pre_llm_workflow failed: {str(exc)}"],
            }

    async def _run_data_discovery(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> None:
        subflow_state = state.subworkflow("data_discovery")
        state.stage = "data_discovery"
        state.operation = "run_data_discovery"

        if self.data_discovery_executor is None:
            subflow_state.status = "skipped"
            return

        try:
            result = await self.data_discovery_executor.run(input, state)
            state.data_discovery = self._to_metadata(result)
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

    @staticmethod
    def _to_metadata(result: Any) -> dict[str, Any]:
        if hasattr(result, "to_metadata"):
            return result.to_metadata()
        if isinstance(result, dict):
            return result
        return {"status": getattr(result, "status", "success")}
