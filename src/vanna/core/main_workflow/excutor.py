"""MainWorkflow executor for turn state orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from vanna.core.question_understanding_subworkflow import QuestUnderstand_Input

try:
    from vanna.core.data_discovering_subworkflow import DataDiscover_Input
except ImportError:
    DataDiscover_Input = None

from .state import MainWorkflowInput, MainWorkflowTurnState

logger = logging.getLogger(__name__)

FB1_FAILURE_TYPES = {
    "intent_classification_failed",
    "time_normalization_failed",
    "time_metadata_validation_failed",
    "question_structuring_failed",
    "search_queries_generation_failed",
}

JSON_RETRY_FAILURE_TYPES = {
    "json_validation_failed",
    "json_validation_retry_exceeded",
}

FB2_DATA_DISCOVERY_FAILURE_TYPES = {"metadata_execution_error"}
FB2_QUESTION_UNDERSTANDING_FAILURE_TYPES = {"metadata_semantic_mismatch"}
TURNSTATE_LOG_DIR = Path(r"C:\Users\dlatn\GenSQL\gensql\scripts\turnstate")


def _safe_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _write_turn_state_file(event: str, state: MainWorkflowTurnState) -> Path | None:
    try:
        TURNSTATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        event_name = _safe_filename_part(event)[:80]
        turn_id = _safe_filename_part(state.turn_id)[:80]
        path = TURNSTATE_LOG_DIR / f"{timestamp}_{event_name}_{turn_id}.json"
        payload = {
            "event": event,
            "timestamp": timestamp,
            "turn_state": state.to_metadata(),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception:
        logger.exception("[main_workflow.turn_state] failed_to_write_file event=%s", event)
        return None

# intent == sql
def _log_turn_state(event: str, state: MainWorkflowTurnState) -> None:
    file_path = _write_turn_state_file(event, state)
    logger.info(
        "[main_workflow.turn_state] %s turn_id=%s stage=%s operation=%s "
        "intent=%s question_understanding=%s data_discovery=%s sql_processing=%s file=%s",
        event,
        state.turn_id,
        state.stage,
        state.operation,
        state.structured_question.get("intent"),
        state.subflow("question_understanding").status,
        state.subflow("data_discovery").status,
        state.subflow("sql_processing").status,
        str(file_path) if file_path is not None else None,
    )

# intent != sql
def _log_turn_state_summary(event: str, state: MainWorkflowTurnState) -> None:
    file_path = _write_turn_state_file(event, state)
    logger.info(
        "[main_workflow.turn_state] %s turn_id=%s stage=%s operation=%s "
        "intent=%s question_understanding=%s data_discovery=%s file=%s",
        event,
        state.turn_id,
        state.stage,
        state.operation,
        state.structured_question.get("intent"),
        state.subflow("question_understanding").status,
        state.subflow("data_discovery").status,
        str(file_path) if file_path is not None else None,
    )


class MainWorkflowExecutor:
    """Runs pre-LLM subflows and records normalized turn state."""

    def __init__(
        self,
        question_understanding_executor=None,
        data_discovery_executor=None,
        router=None,
        question_understanding_subworkflow_executor=None,
        pre_llm_workflow_executor=None,
        **_: Any,
    ):
        self.question_understanding_executor = (
            question_understanding_executor
            or question_understanding_subworkflow_executor
            or pre_llm_workflow_executor
        )
        self.data_discovery_executor = data_discovery_executor
        self.router = router

    async def run(self, input: MainWorkflowInput) -> MainWorkflowTurnState:
        state = MainWorkflowTurnState(
            turn_id=input.request_id,
            original_question=input.original_message,
        )
        _log_turn_state("initialized", state)

        await self._run_question_understanding_with_fb1(input, state)
        _log_turn_state("question_understanding_saved", state)

        is_non_sql_intent = self._is_non_sql_intent(state)
        if self._should_run_data_discovery(state):
            await self._run_data_discovery_with_fb2(input, state)
            _log_turn_state("data_discovery_saved", state)
        else:
            state.subflow("data_discovery").status = "skipped"
            if is_non_sql_intent:
                _log_turn_state_summary("data_discovery_skipped", state)
            else:
                _log_turn_state("data_discovery_saved", state)

        state.stage = "context_enrichment"
        if state.operation not in {"clarification_required", "continue_with_warning"}:
            state.operation = "context_enrichment_ready"
        if is_non_sql_intent:
            _log_turn_state_summary("context_enrichment_ready", state)
        else:
            _log_turn_state("context_enrichment_ready", state)
        return state

    @staticmethod
    def _is_non_sql_intent(state: MainWorkflowTurnState) -> bool:
        question_subflow = state.subflow("question_understanding")
        intent = state.structured_question.get("intent")
        return (
            question_subflow.status == "success"
            and isinstance(intent, str)
            and intent != "sql"
        )

    async def _run_question_understanding_with_fb1(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> dict[str, Any]:
        result_metadata = await self._run_question_understanding_once(input, state)
        failure_type = self._failure_type(result_metadata)

        if (
            result_metadata.get("status") == "failed"
            and failure_type in FB1_FAILURE_TYPES
            and state.fallback_state.can_use_fb1()
        ):
            feedback = self._build_fb1_feedback(result_metadata)
            state.fallback_state.mark_fb1(feedback)
            _log_turn_state("fb1_marked", state)
            result_metadata = await self._run_question_understanding_once(input, state)

        if result_metadata.get("status") == "failed":
            self._assign_clarification_result(
                state,
                failed_subflow="question_understanding",
                result_metadata=result_metadata,
            )

        return result_metadata

    async def _run_question_understanding_once(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> dict[str, Any]:
        subflow_state = state.subflow("question_understanding")
        state.stage = "question_understanding"
        state.operation = "run_question_understanding"

        if self.question_understanding_executor is None:
            subflow_state.status = "skipped"
            return {"status": "skipped"}

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
                    "fallback_feedback": state.fallback_state.active_feedback,
                },
            )
            result = await self.question_understanding_executor.run(workflow_input)
            result_metadata = self._to_metadata(result)

            structured_output = result_metadata.get("structured_output")
            state.structured_question = structured_output if isinstance(structured_output, dict) else {}
            _log_turn_state("structured_question_assigned", state)

            subflow_state.status = result_metadata.get("status", "success")
            subflow_state.errors = list(result_metadata.get("errors", []))
            subflow_state.retry_counts = dict(result_metadata.get("retry_counts", {}))
            return result_metadata
        except Exception as exc:
            subflow_state.status = "failed"
            subflow_state.errors.append(str(exc))
            state.structured_question = {}
            result_metadata = {
                "status": "failed",
                "failed_node_id": "question_understanding_subworkflow",
                "failure_type": "question_structuring_failed",
                "errors": [f"question_understanding_failed: {type(exc).__name__}: {exc}"],
            }
            _log_turn_state("question_understanding_error_assigned", state)
            return result_metadata

    async def _run_data_discovery_with_fb2(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> dict[str, Any]:
        result_metadata = await self._run_data_discovery_once(input, state)
        failure_type = self._failure_type(result_metadata)

        if result_metadata.get("status") != "failed":
            return result_metadata

        if failure_type in FB2_DATA_DISCOVERY_FAILURE_TYPES:
            if state.fallback_state.can_use_fb2():
                feedback = self._build_fb2_feedback(
                    result_metadata,
                    next_action="retry_data_discovery",
                    target_nodes=["metadata_search"],
                )
                state.fallback_state.mark_fb2(feedback)
                _log_turn_state("fb2_marked_retry_data_discovery", state)
                retry_result = await self._run_data_discovery_once(
                    input,
                    state,
                    structured_output_override=self._original_question_structured_output(state),
                    continue_with_warning=True,
                )
                if (
                    retry_result.get("status") == "failed"
                    and self._failure_type(retry_result) == "metadata_execution_error"
                ):
                    self._convert_metadata_execution_error_to_warning(state, retry_result)
                return retry_result

            self._convert_metadata_execution_error_to_warning(state, result_metadata)
            return result_metadata

        if failure_type in FB2_QUESTION_UNDERSTANDING_FAILURE_TYPES:
            if state.fallback_state.can_use_fb2():
                feedback = self._build_fb2_feedback(
                    result_metadata,
                    next_action="retry_question_understanding",
                    target_nodes=["question_structuring", "search_queries"],
                )
                state.fallback_state.mark_fb2(feedback)
                _log_turn_state("fb2_marked_retry_question_understanding", state)
                question_result = await self._run_question_understanding_once(input, state)
                if self._should_run_data_discovery(state):
                    retry_result = await self._run_data_discovery_once(input, state)
                    if retry_result.get("status") == "failed":
                        retry_failure_type = self._failure_type(retry_result)
                        if retry_failure_type == "metadata_execution_error":
                            self._convert_metadata_execution_error_to_warning(state, retry_result)
                        else:
                            self._assign_clarification_result(
                                state,
                                failed_subflow="data_discovery",
                                result_metadata=retry_result,
                            )
                    return retry_result
                self._assign_clarification_result(
                    state,
                    failed_subflow="question_understanding",
                    result_metadata=question_result,
                )
                return question_result

            self._assign_clarification_result(
                state,
                failed_subflow="data_discovery",
                result_metadata=result_metadata,
            )
            return result_metadata

        self._assign_clarification_result(
            state,
            failed_subflow="data_discovery",
            result_metadata=result_metadata,
        )
        return result_metadata

    async def _run_data_discovery_once(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
        *,
        structured_output_override: dict[str, Any] | None = None,
        continue_with_warning: bool = False,
    ) -> dict[str, Any]:
        subflow_state = state.subflow("data_discovery")
        state.stage = "data_discovery"
        state.operation = "run_data_discovery"

        if self.data_discovery_executor is None:
            subflow_state.status = "skipped"
            return {"status": "skipped"}

        try:
            workflow_input = self._build_data_discovery_input(
                state,
                structured_output_override=structured_output_override,
            )
            result = await self._run_data_discovery_executor(workflow_input, input, state)
            result_metadata = self._to_metadata(result)

            self._apply_data_discovery_result(state, result_metadata)
            _log_turn_state("data_discovery_result_assigned", state)

            subflow_state.status = result_metadata.get("status", "success")
            if continue_with_warning and self._failure_type(result_metadata) == "metadata_execution_error":
                subflow_state.status = "success"
            subflow_state.errors = list(result_metadata.get("errors", []))
            subflow_state.retry_counts = dict(result_metadata.get("retry_counts", {}))
            return result_metadata
        except Exception as exc:
            subflow_state.status = "failed"
            subflow_state.errors.append(str(exc))
            state.metadata = {"searches": [], "candidates": [], "selected": []}
            state.fewshot = []
            result_metadata = {
                "status": "failed",
                "failed_node_id": "data_discovery_subworkflow",
                "failure_type": "metadata_execution_error",
                "errors": [f"data_discovery_failed: {type(exc).__name__}: {exc}"],
            }
            _log_turn_state("data_discovery_error_assigned", state)
            return result_metadata

    def _build_data_discovery_input(
        self,
        state: MainWorkflowTurnState,
        *,
        structured_output_override: dict[str, Any] | None = None,
    ) -> Any:
        if DataDiscover_Input is None:
            return None

        question_subflow = state.subflow("question_understanding")
        structured_output = structured_output_override or state.structured_question
        return DataDiscover_Input(
            status=question_subflow.status,
            intent=structured_output.get("intent"),
            structured_output=structured_output,
            errors=question_subflow.errors,
            retry_counts=question_subflow.retry_counts,
        )

    async def _run_data_discovery_executor(
        self,
        workflow_input: Any,
        main_input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> Any:
        if workflow_input is not None:
            try:
                return await self.data_discovery_executor.run(workflow_input)
            except TypeError:
                logger.debug(
                    "data_discovery_executor rejected DataDiscover_Input; retrying with MainWorkflowInput and TurnState",
                    exc_info=True,
                )
        return await self.data_discovery_executor.run(main_input, state)

    def _should_run_data_discovery(self, state: MainWorkflowTurnState) -> bool:
        question_subflow = state.subflow("question_understanding")
        return (
            question_subflow.status == "success"
            and state.structured_question.get("intent") == "sql"
            and bool(state.structured_question)
        )

    def _apply_data_discovery_result(
        self,
        state: MainWorkflowTurnState,
        result_metadata: dict[str, Any],
    ) -> None:
        metadata_output = result_metadata.get("metadata_output")
        fewshot_output = result_metadata.get("fewshot_output")

        state.metadata = {
            "searches": self._extract_searches(state.structured_question),
            "candidates": self._extract_metadata_candidates(metadata_output),
            "selected": [],
        }
        state.fewshot = self._extract_fewshot_examples(fewshot_output)

    def _original_question_structured_output(self, state: MainWorkflowTurnState) -> dict[str, Any]:
        return {
            "intent": "sql",
            "question": state.original_question,
            "original_question": state.original_question,
        }

    def _convert_metadata_execution_error_to_warning(
        self,
        state: MainWorkflowTurnState,
        result_metadata: dict[str, Any],
    ) -> None:
        errors = list(result_metadata.get("errors", []))
        warning = (
            "\n".join(str(error) for error in errors)
            if errors
            else "metadata_execution_error"
        )
        state.result["message"] = (
            f"{warning}\n"
            "위 에러로 인해 metadata 검색을 완료하지 못했습니다.\n"
            "SQL을 생성하기 전에 search_business_metadata tool을 사용해 "
            "전체 table metadata와 column metadata를 다시 검색하세요."
        )
        state.operation = "continue_with_warning"
        state.subflow("data_discovery").status = "success"

    def _assign_clarification_result(
        self,
        state: MainWorkflowTurnState,
        *,
        failed_subflow: str,
        result_metadata: dict[str, Any],
    ) -> None:
        failure_type = self._failure_type(result_metadata) or "unknown_failure"
        failed_node_id = result_metadata.get("failed_node_id") or failed_subflow
        errors = result_metadata.get("errors") or []
        error_text = "\n".join(str(error) for error in errors) or failure_type
        state.operation = "clarification_required"
        state.result["message"] = (
            f"{failed_subflow}의 {failed_node_id}에서 {failure_type} 오류가 발생했습니다.\n"
            f"원인: {error_text}\n"
            "사용자에게 어떤 조건, 지표, 테이블/컬럼 또는 시간 범위가 필요한지 "
            "명확히 재질문하세요."
        )

    @staticmethod
    def _build_fb1_feedback(result_metadata: dict[str, Any]) -> dict[str, Any]:
        failure_type = MainWorkflowExecutor._failure_type(result_metadata) or "question_understanding_failed"
        return {
            "fallback_id": "FB1",
            "failed_subflow": "question_understanding",
            "failed_node_id": result_metadata.get("failed_node_id"),
            "reason": failure_type,
            "next_action": "retry_question_understanding",
            "feedback": {
                "reason": failure_type,
                "errors": list(result_metadata.get("errors", [])),
                "failure_detail": result_metadata.get("failure_detail"),
            },
        }

    @staticmethod
    def _build_fb2_feedback(
        result_metadata: dict[str, Any],
        *,
        next_action: str,
        target_nodes: list[str],
    ) -> dict[str, Any]:
        failure_type = MainWorkflowExecutor._failure_type(result_metadata) or "data_discovery_failed"
        return {
            "fallback_id": "FB2",
            "failed_subflow": "data_discovery",
            "failed_node_id": result_metadata.get("failed_node_id"),
            "reason": failure_type,
            "next_action": next_action,
            "feedback": {
                "reason": failure_type,
                "target_nodes": target_nodes,
                "errors": list(result_metadata.get("errors", [])),
                "failure_detail": result_metadata.get("failure_detail"),
            },
        }

    @staticmethod
    def _failure_type(result_metadata: dict[str, Any]) -> str | None:
        failure_type = result_metadata.get("failure_type")
        if isinstance(failure_type, str) and failure_type:
            return failure_type
        joined_errors = "\n".join(str(error) for error in result_metadata.get("errors", []))
        for candidate in (
            "metadata_semantic_mismatch",
            "metadata_execution_error",
            "intent_classification_failed",
            "time_normalization_failed",
            "time_metadata_validation_failed",
            "question_structuring_failed",
            "search_queries_generation_failed",
            "json_validation_failed",
            "json_validation_retry_exceeded",
        ):
            if candidate in joined_errors:
                return candidate
        return None

    @staticmethod
    def _extract_searches(structured_question: dict[str, Any]) -> list[dict[str, Any]]:
        search_plan = structured_question.get("search_plan")
        if isinstance(search_plan, dict):
            for key in ("queries", "search_queries"):
                queries = search_plan.get(key)
                if isinstance(queries, list):
                    return [query for query in queries if isinstance(query, dict)]

        queries = structured_question.get("search_queries")
        if isinstance(queries, list):
            return [query for query in queries if isinstance(query, dict)]
        return []

    @staticmethod
    def _extract_metadata_candidates(metadata_output: Any) -> list[dict[str, Any]]:
        if not isinstance(metadata_output, dict):
            return []

        candidates: list[dict[str, Any]] = []
        for table in metadata_output.get("tables", []) or []:
            if isinstance(table, dict):
                candidates.append({"type": "table", **table})
        for column in metadata_output.get("columns", []) or []:
            if isinstance(column, dict):
                candidates.append({"type": "column", **column})
        return candidates

    @staticmethod
    def _extract_fewshot_examples(fewshot_output: Any) -> list[dict[str, Any]]:
        if not isinstance(fewshot_output, dict):
            return []
        examples = fewshot_output.get("examples")
        if not isinstance(examples, list):
            return []
        return [example for example in examples if isinstance(example, dict)]

    @staticmethod
    def _to_metadata(result: Any) -> dict[str, Any]:
        if hasattr(result, "to_metadata"):
            return result.to_metadata()
        if isinstance(result, dict):
            return result
        return {"status": getattr(result, "status", "success")}



