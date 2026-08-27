from __future__ import annotations

from typing import Any

from .state import SqlProcessingFinalResult, SqlProcessingInput


class SqlProcessingSubworkflowExecutor:
    """Thin wrapper around the Agent's existing LLM/tool-calling loop.

    The loop still runs in Agent for now because it owns conversation, UI,
    lifecycle hooks, audit, observability, and tool execution. This wrapper
    provides a subworkflow boundary and records governance state in TurnState.
    """

    def start(self, turn_state: Any, input: SqlProcessingInput) -> None:
        turn_state.stage = "sql_generation"
        turn_state.operation = "sql_generation"
        subflow = turn_state.subflow("sql_processing")
        subflow.status = "success"
        subflow.current_node = "agent_llm_tool_loop"
        if "agent_llm_tool_loop" not in subflow.visited_nodes:
            subflow.visited_nodes.append("agent_llm_tool_loop")

    def finalize(
        self,
        turn_state: Any,
        *,
        status: str = "success",
        errors: list[str] | None = None,
    ) -> SqlProcessingFinalResult:
        subflow = turn_state.subflow("sql_processing")
        subflow.status = "failed" if status == "failed" else "success"
        subflow.current_node = None
        if errors:
            subflow.errors.extend(str(error) for error in errors)
        turn_state.stage = "final"
        turn_state.operation = "final_response_ready"
        return SqlProcessingFinalResult(
            status=subflow.status,
            attempts=[attempt.to_metadata() for attempt in turn_state.attempts],
            errors=list(subflow.errors),
        )
