"""Tests for the Question-Understanding workflow executor with fake nodes.

test 항목
- workflow fake SQL 경로 성공
- general intent skip
- QuestUnderstand_FinalResult metadata 전달
- retry limit 초과
- max steps 초과
- start node 없음 검증
- end node 없음 검증
- unreachable node 검증
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_PATH = Path(__file__).resolve().parent
SRC_PATH = TESTS_PATH.parent / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(TESTS_PATH))

from vanna.core.question_understanding_subworkflow import (
    QuestUnderstand_NodeResult,
    QuestionUnderstandSubWorkflowExecutor,
    WorkflowGraph,
    WorkflowGraphError,
    QuestUnderstand_Input,
    QuestUnderstand_State,
)
from question_understanding_subworkflow_fakes.fake_graph import build_fake_question_understanding_subworkflow_graph


class AlwaysRetryNode:
    node_id = "retry_node"

    async def run(self, state: QuestUnderstand_State) -> QuestUnderstand_NodeResult:
        return QuestUnderstand_NodeResult(status="retry", error="retry requested")


class RetryStatusIs:
    def __init__(self, status: str) -> None:
        self.status = status

    async def evaluate(
        self,
        state: QuestUnderstand_State,
        last_node_result: QuestUnderstand_NodeResult,
    ) -> bool:
        return last_node_result.status == self.status


class RegeneratingNode:
    node_id = "structuring_node"

    async def run(self, state: QuestUnderstand_State) -> QuestUnderstand_NodeResult:
        generation = state.visited_nodes.count(self.node_id)
        return QuestUnderstand_NodeResult(
            status="success",
            structured_question={"generation": generation},
        )


class RetryOnceThenFinishNode:
    node_id = "validation_node"

    async def run(self, state: QuestUnderstand_State) -> QuestUnderstand_NodeResult:
        if state.retry.get_attempts(self.node_id) == 0:
            return QuestUnderstand_NodeResult(
                status="retry",
                output={"reason": "regeneration required"},
            )

        return QuestUnderstand_NodeResult(status="finish")


class RetryOnceThenSucceedNode:
    node_id = "retry_then_succeed"

    async def run(self, state: QuestUnderstand_State) -> QuestUnderstand_NodeResult:
        attempts = state.retry.get_attempts(self.node_id)
        if attempts == 0:
            return QuestUnderstand_NodeResult(status="retry")

        return QuestUnderstand_NodeResult(
            status="success",
            structured_question={"attempts": attempts + 1},
        )


class SuccessfulNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    async def run(self, state: QuestUnderstand_State) -> QuestUnderstand_NodeResult:
        return QuestUnderstand_NodeResult(status="success")


class MetadataFinishNode:
    node_id = "metadata_finish"

    async def run(self, state: QuestUnderstand_State) -> QuestUnderstand_NodeResult:
        return QuestUnderstand_NodeResult(
            status="finish",
            structured_question={"intent": "sql", "target_entity": "invoice"},
            debug_metadata={"node_version": "test"},
        )


def build_input(message: str = "Show me sales by day for this week") -> QuestUnderstand_Input:
    return QuestUnderstand_Input(
        user_id="test_user",
        conversation_id="test_conversation",
        request_id="test_request",
        original_message=message,
        system_prompt="You are a SQL assistant.",
        tool_names=["run_sql"],
        metadata={"source": "test"},
    )


@pytest.mark.asyncio
async def test_fake_sql_workflow_reaches_structured_result() -> None:
    graph = build_fake_question_understanding_subworkflow_graph(intent="sql")
    executor = QuestionUnderstandSubWorkflowExecutor(graph)

    result = await executor.run(build_input())

    assert result.status == "success"
    assert result.intent == "sql"
    assert result.structured_output is not None
    assert result.structured_output["target_entity"] == "sales_order"
    assert result.retry_counts == {}


@pytest.mark.asyncio
async def test_fake_general_workflow_is_skipped() -> None:
    graph = build_fake_question_understanding_subworkflow_graph(intent="general")
    executor = QuestionUnderstandSubWorkflowExecutor(graph)

    result = await executor.run(build_input("hello"))

    assert result.status == "skipped"
    assert result.intent == "general"
    assert result.structured_output is None


@pytest.mark.asyncio
async def test_workflow_final_result_exposes_minimal_metadata() -> None:
    graph = WorkflowGraph()
    graph.add_node(MetadataFinishNode(), start=True, end=True)
    executor = QuestionUnderstandSubWorkflowExecutor(graph)

    result = await executor.run(build_input())

    assert result.status == "success"
    assert result.intent == "sql"
    assert result.structured_output == {"intent": "sql", "target_entity": "invoice"}
    assert result.to_metadata() == {
        "status": "success",
        "intent": "sql",
        "structured_output": {"intent": "sql", "target_entity": "invoice"},
        "errors": [],
        "retry_counts": {},
    }


@pytest.mark.asyncio
async def test_retry_limit_exceeded_returns_failed_result() -> None:
    graph = WorkflowGraph()
    graph.add_node(AlwaysRetryNode(), start=True, end=True)
    executor = QuestionUnderstandSubWorkflowExecutor(graph, retry_limit=1)

    result = await executor.run(build_input())

    assert result.status == "failed"
    assert result.retry_counts == {"retry_node": 2}
    assert "Retry limit exceeded for node: retry_node" in result.errors


@pytest.mark.asyncio
async def test_retry_follows_matching_edge_and_preserves_retry_count() -> None:
    graph = WorkflowGraph()
    graph.add_node(RegeneratingNode(), start=True)
    graph.add_node(RetryOnceThenFinishNode(), end=True)
    graph.add_edge("structuring_node", "validation_node")
    graph.add_edge(
        "validation_node",
        "structuring_node",
        condition=RetryStatusIs("retry"),
        label="regenerate",
    )
    executor = QuestionUnderstandSubWorkflowExecutor(graph, retry_limit=1)

    result = await executor.run(build_input())

    assert result.status == "success"
    assert result.structured_output == {"generation": 2}
    assert result.retry_counts == {"validation_node": 1}


@pytest.mark.asyncio
async def test_retry_ignores_unconditional_edge_and_retries_current_node() -> None:
    graph = WorkflowGraph()
    graph.add_node(RetryOnceThenSucceedNode(), start=True)
    graph.add_node(SuccessfulNode("end"), end=True)
    graph.add_edge("retry_then_succeed", "end")
    executor = QuestionUnderstandSubWorkflowExecutor(graph, retry_limit=1)

    result = await executor.run(build_input())

    assert result.status == "success"
    assert result.structured_output == {"attempts": 2}
    assert result.retry_counts == {"retry_then_succeed": 1}


@pytest.mark.asyncio
async def test_max_steps_exceeded_returns_failed_result() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("loop"), start=True)
    graph.add_node(SuccessfulNode("end"), end=True)
    graph.add_edge("loop", "loop", label="repeat")
    graph.add_edge("loop", "end", label="reachable_end")
    executor = QuestionUnderstandSubWorkflowExecutor(graph, max_steps=2)

    result = await executor.run(build_input())

    assert result.status == "failed"
    assert "Max workflow steps exceeded: 2" in result.errors


@pytest.mark.asyncio
async def test_missing_start_node_is_rejected() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("end"), end=True)
    executor = QuestionUnderstandSubWorkflowExecutor(graph)

    with pytest.raises(WorkflowGraphError, match="no start node"):
        await executor.run(build_input())


@pytest.mark.asyncio
async def test_missing_end_node_is_rejected() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("start"), start=True)
    executor = QuestionUnderstandSubWorkflowExecutor(graph)

    with pytest.raises(WorkflowGraphError, match="no end nodes"):
        await executor.run(build_input())


@pytest.mark.asyncio
async def test_unreachable_node_is_rejected() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("start"), start=True)
    graph.add_node(SuccessfulNode("end"), end=True)
    graph.add_node(SuccessfulNode("unreachable"), end=True)
    graph.add_edge("start", "end")
    executor = QuestionUnderstandSubWorkflowExecutor(graph)

    with pytest.raises(WorkflowGraphError, match="unreachable nodes"):
        await executor.run(build_input())
