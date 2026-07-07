"""Tests for the pre-LLM workflow executor with fake nodes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_PATH = Path(__file__).resolve().parent
SRC_PATH = TESTS_PATH.parent / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(TESTS_PATH))

from vanna.core.pre_llm_workflow import (
    NodeResult,
    PreLlmWorkflowExecutor,
    WorkflowGraph,
    WorkflowGraphError,
    WorkflowInput,
    WorkflowState,
)
from pre_llm_workflow_fakes.fake_graph import build_fake_pre_llm_graph


class AlwaysRetryNode:
    node_id = "retry_node"

    async def run(self, state: WorkflowState) -> NodeResult:
        return NodeResult(status="retry", error="retry requested")


class SuccessfulNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    async def run(self, state: WorkflowState) -> NodeResult:
        return NodeResult(status="success")


class MetadataFinishNode:
    node_id = "metadata_finish"

    async def run(self, state: WorkflowState) -> NodeResult:
        return NodeResult(
            status="finish",
            structured_question={"intent": "sql", "target_entity": "invoice"},
            prompt_metadata={"prompt_hint": "prefer_invoice_tables"},
            request_metadata={"request_hint": "include_workflow"},
            debug_metadata={"node_version": "test"},
        )


def build_input(message: str = "Show me sales by day for this week") -> WorkflowInput:
    return WorkflowInput(
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
    graph = build_fake_pre_llm_graph(intent="sql")
    executor = PreLlmWorkflowExecutor(graph)

    result = await executor.run(build_input())

    assert result.status == "success"
    assert result.intent == "sql"
    assert result.structured_output is not None
    assert result.structured_output["target_entity"] == "sales_order"
    assert result.retry_counts == {}


@pytest.mark.asyncio
async def test_fake_general_workflow_is_skipped() -> None:
    graph = build_fake_pre_llm_graph(intent="general")
    executor = PreLlmWorkflowExecutor(graph)

    result = await executor.run(build_input("hello"))

    assert result.status == "skipped"
    assert result.intent == "general"
    assert result.structured_output is None


@pytest.mark.asyncio
async def test_workflow_final_result_includes_metadata() -> None:
    graph = WorkflowGraph()
    graph.add_node(MetadataFinishNode(), start=True, end=True)
    executor = PreLlmWorkflowExecutor(graph)

    result = await executor.run(build_input())

    assert result.status == "success"
    assert result.structured_output == {"intent": "sql", "target_entity": "invoice"}
    assert result.prompt_metadata == {"prompt_hint": "prefer_invoice_tables"}
    assert result.request_metadata == {"request_hint": "include_workflow"}
    assert result.debug_metadata == {"node_version": "test"}
    assert result.to_metadata()["prompt_metadata"] == result.prompt_metadata
    assert result.to_metadata()["request_metadata"] == result.request_metadata
    assert result.to_metadata()["debug_metadata"] == result.debug_metadata


@pytest.mark.asyncio
async def test_retry_limit_exceeded_returns_failed_result() -> None:
    graph = WorkflowGraph()
    graph.add_node(AlwaysRetryNode(), start=True, end=True)
    executor = PreLlmWorkflowExecutor(graph, retry_limit=1)

    result = await executor.run(build_input())

    assert result.status == "failed"
    assert result.retry_counts == {"retry_node": 2}
    assert "Retry limit exceeded for node: retry_node" in result.errors


@pytest.mark.asyncio
async def test_max_steps_exceeded_returns_failed_result() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("loop"), start=True)
    graph.add_node(SuccessfulNode("end"), end=True)
    graph.add_edge("loop", "loop", label="repeat")
    graph.add_edge("loop", "end", label="reachable_end")
    executor = PreLlmWorkflowExecutor(graph, max_steps=2)

    result = await executor.run(build_input())

    assert result.status == "failed"
    assert "Max workflow steps exceeded: 2" in result.errors


@pytest.mark.asyncio
async def test_missing_start_node_is_rejected() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("end"), end=True)
    executor = PreLlmWorkflowExecutor(graph)

    with pytest.raises(WorkflowGraphError, match="no start node"):
        await executor.run(build_input())


@pytest.mark.asyncio
async def test_missing_end_node_is_rejected() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("start"), start=True)
    executor = PreLlmWorkflowExecutor(graph)

    with pytest.raises(WorkflowGraphError, match="no end nodes"):
        await executor.run(build_input())


@pytest.mark.asyncio
async def test_unreachable_node_is_rejected() -> None:
    graph = WorkflowGraph()
    graph.add_node(SuccessfulNode("start"), start=True)
    graph.add_node(SuccessfulNode("end"), end=True)
    graph.add_node(SuccessfulNode("unreachable"), end=True)
    graph.add_edge("start", "end")
    executor = PreLlmWorkflowExecutor(graph)

    with pytest.raises(WorkflowGraphError, match="unreachable nodes"):
        await executor.run(build_input())
