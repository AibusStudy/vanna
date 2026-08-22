from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NextAction = Literal[
    "retry_question_understanding",
    "retry_data_discovery",
    "continue_with_warning",
    "ask_clarification",
    "fail",
]


@dataclass(frozen=True)
class FallbackDecision:
    fallback_id: str
    failed_subflow: str
    failed_node_id: str
    reason: str
    next_action: NextAction
    feedback: dict[str, object]


class FallbackRouter:
    def decide_fb2(
        self,
        *,
        failed_node_id: str,
        failure_type: str,
        retry_count: int,
        max_retry: int,
        errors: list[str] | None = None,
    ) -> FallbackDecision:
        if failure_type == "metadata_semantic_mismatch":
            if retry_count < max_retry:
                return FallbackDecision(
                    fallback_id="FB2",
                    failed_subflow="data_discovery",
                    failed_node_id=failed_node_id,
                    reason=failure_type,
                    next_action="retry_question_understanding",
                    feedback={
                        "reason": failure_type,
                        "target_nodes": ["question_structuring", "search_queries"],
                        "retry_constraints": [
                            "Regenerate structured question and search_plan with mismatch feedback.",
                            "Search queries must preserve the user's business intent.",
                        ],
                    },
                )
            return self._ask_or_fail(
                failed_node_id=failed_node_id,
                failure_type=failure_type,
                errors=errors,
            )

        if failure_type == "metadata_execution_error":
            if retry_count < max_retry:
                return FallbackDecision(
                    fallback_id="FB2",
                    failed_subflow="data_discovery",
                    failed_node_id=failed_node_id,
                    reason=failure_type,
                    next_action="retry_data_discovery",
                    feedback={
                        "reason": failure_type,
                        "target_nodes": ["metadata_search"],
                        "retry_constraints": [
                            "Fallback to original question based metadata search."
                        ],
                    },
                )
            return self._continue_with_warning(
                failed_node_id=failed_node_id,
                failure_type=failure_type,
                errors=errors,
            )

        return self._ask_or_fail(
            failed_node_id=failed_node_id,
            failure_type=failure_type,
            errors=errors,
        )

    def _continue_with_warning(
        self,
        *,
        failed_node_id: str,
        failure_type: str,
        errors: list[str] | None,
    ) -> FallbackDecision:
        error_text = self._error_text(failed_node_id, failure_type, errors)
        return FallbackDecision(
            fallback_id="FB2",
            failed_subflow="data_discovery",
            failed_node_id=failed_node_id,
            reason=failure_type,
            next_action="continue_with_warning",
            feedback={
                "reason": failure_type,
                "warning": (
                    f"{error_text}\n"
                    "위 에러로 인해 metadata 검색을 완료하지 못했다. "
                    "SQL을 생성하기 전에 search_business_metadata tool을 사용해 "
                    "전체 table metadata와 column metadata를 다시 검색하라."
                ),
            },
        )

    def _ask_or_fail(
        self,
        *,
        failed_node_id: str,
        failure_type: str,
        errors: list[str] | None,
    ) -> FallbackDecision:
        error_text = self._error_text(failed_node_id, failure_type, errors)
        return FallbackDecision(
            fallback_id="FB2",
            failed_subflow="data_discovery",
            failed_node_id=failed_node_id,
            reason=failure_type,
            next_action="ask_clarification",
            feedback={
                "reason": failure_type,
                "error": error_text,
            },
        )

    @staticmethod
    def _error_text(
        failed_node_id: str,
        failure_type: str,
        errors: list[str] | None,
    ) -> str:
        return (
            "\n".join(str(error) for error in errors)
            if errors
            else f"{failed_node_id} failed: {failure_type}"
        )
