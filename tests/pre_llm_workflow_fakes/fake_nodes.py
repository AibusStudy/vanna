"""Fake nodes and edge conditions for exercising the pre-LLM workflow engine."""

from __future__ import annotations

from typing import Any, Dict, Optional

from vanna.core.pre_llm_workflow import NodeResult, WorkflowState


class IntentIs:
    """Edge condition that matches the current routing intent."""

    def __init__(self, intent: str) -> None:
        self.intent = intent

    async def evaluate(
        self,
        state: WorkflowState,
        last_node_result: NodeResult,
    ) -> bool:
        return state.routing_intent == self.intent


class FakeIntentNode:
    """Fake intent classifier node.

    Set intent to "sql" to follow the structured-question path, or "general" to
    finish as skipped.
    """

    node_id = "intent_classification"

    def __init__(self, intent: str = "sql") -> None:
        self.intent = intent

    async def run(self, state: WorkflowState) -> NodeResult:
        return NodeResult(
            status="success",
            output={"intent": self.intent},
            routing_intent=self.intent,
        )


class FakeTimeNormalizerNode:
    """Fake time normalizer node."""

    node_id = "time_normalization"

    async def run(self, state: WorkflowState) -> NodeResult:
        normalized_time = {
            "time_conditions": [
                {
                    "field": "order_date",
                    "operator": "between",
                    "start": "20260701",
                    "end": "20260707",
                    "grain": "day",
                }
            ]
        }

        return NodeResult(
            status="success",
            output=normalized_time,
            debug_metadata={"normalized_time_source": "fake"},
        )


class FakeQuestionStructurerNode:
    """Fake question structurer node."""

    node_id = "question_structuring"

    async def run(self, state: WorkflowState) -> NodeResult:
        normalized_time = state.get_node_output("time_normalization") or {}
        structured_question: Dict[str, Any] = {
            "original_question": state.original_message,
            "intent": state.routing_intent,
            "parse_status": "success",
            "target_entity": "sales_order",
            "business_terms": ["sales", "orders"],
            "time_conditions": normalized_time.get("time_conditions", []),
            "value_conditions": [],
            "aggregation": {"function": "sum", "field": "amount"},
            "group_by": [],
            "sort_conditions": [],
            "limit": 10,
            "output_fields": ["order_date", "amount"],
            "search_queries": [state.original_message],
            "ambiguous_terms": [],
        }

        return NodeResult(
            status="success",
            output={"structured": True},
            structured_question=structured_question,
        )


class FakeStructuredQuestionValidatorNode:
    """Fake validator node."""

    node_id = "structured_json_validation"

    def __init__(self, *, force_error: Optional[str] = None) -> None:
        self.force_error = force_error

    async def run(self, state: WorkflowState) -> NodeResult:
        if self.force_error:
            return NodeResult(
                status="failed",
                output={"valid": False},
                error=self.force_error,
            )

        if not state.structured_question:
            return NodeResult(
                status="failed",
                output={"valid": False},
                error="Structured question is missing.",
            )

        return NodeResult(
            status="finish",
            output={"valid": True},
            request_metadata={"validated_by": "fake"},
        )


class FakeGeneralFinishNode:
    """Fake terminal node for non-SQL questions."""

    node_id = "general_finish"

    async def run(self, state: WorkflowState) -> NodeResult:
        return NodeResult(
            status="skipped",
            output={"reason": "general intent does not need structuring"},
        )

