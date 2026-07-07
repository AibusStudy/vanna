"""Tests for the pre-LLM workflow executor with fake nodes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_PATH = Path(__file__).resolve().parent
SRC_PATH = TESTS_PATH.parent / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(TESTS_PATH))

from vanna.core.pre_llm_workflow import PreLlmWorkflowExecutor, WorkflowInput
from pre_llm_workflow_fakes.fake_graph import build_fake_pre_llm_graph


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
