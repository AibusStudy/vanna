"""Tests for Agent integration with the optional pre-LLM workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
from pydantic import BaseModel

TESTS_PATH = Path(__file__).resolve().parent
SRC_PATH = TESTS_PATH.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from vanna import Agent, AgentConfig
from vanna.core.llm import LlmRequest, LlmResponse, LlmService, LlmStreamChunk
from vanna.core.pre_llm_workflow import WorkflowFinalResult, WorkflowInput
from vanna.core.registry import ToolRegistry
from vanna.core.tool import Tool, ToolCall, ToolContext, ToolResult
from vanna.core.user import User
from vanna.core.user.request_context import RequestContext
from vanna.core.user.resolver import UserResolver
from vanna.integrations.local.agent_memory import DemoAgentMemory


class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        return User(
            id="test_user",
            email="test@example.com",
            group_memberships=["user"],
        )


class RecordingLlmService(LlmService):
    def __init__(self, responses: Optional[List[LlmResponse]] = None) -> None:
        self.requests: List[LlmRequest] = []
        self.responses = responses or [LlmResponse(content="done")]

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return LlmResponse(content="done")

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        self.requests.append(request)
        yield LlmStreamChunk(content="done", finish_reason="stop")

    async def validate_tools(self, tools: List[Any]) -> List[str]:
        return []


class StaticWorkflowExecutor:
    def __init__(self) -> None:
        self.inputs: List[WorkflowInput] = []

    async def run(self, workflow_input: WorkflowInput) -> WorkflowFinalResult:
        self.inputs.append(workflow_input)
        return WorkflowFinalResult(
            status="success",
            intent="sql",
            structured_output={
                "intent": "sql",
                "target_entity": "sales_order",
            },
            prompt_metadata={"prompt_hint": "use_structured_question"},
            request_metadata={"request_hint": "attach_to_llm_request"},
            debug_metadata={"debug_hint": "test_executor"},
        )


class FailingWorkflowExecutor:
    def __init__(self) -> None:
        self.inputs: List[WorkflowInput] = []

    async def run(self, workflow_input: WorkflowInput) -> WorkflowFinalResult:
        self.inputs.append(workflow_input)
        raise RuntimeError("workflow boom")


class EmptyArgs(BaseModel):
    pass


class CaptureContextTool(Tool[EmptyArgs]):
    def __init__(self) -> None:
        self.captured_metadata: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "capture_context"

    @property
    def description(self) -> str:
        return "Capture tool context metadata for tests."

    def get_args_schema(self) -> type[EmptyArgs]:
        return EmptyArgs

    async def execute(self, context: ToolContext, args: EmptyArgs) -> ToolResult:
        self.captured_metadata = dict(context.metadata)
        return ToolResult(success=True, result_for_llm="captured")


def create_agent(
    *,
    llm_service: LlmService,
    pre_llm_workflow_executor: Any = None,
    enable_pre_llm_workflow: bool = True,
    tool_registry: Optional[ToolRegistry] = None,
) -> Agent:
    return Agent(
        llm_service=llm_service,
        tool_registry=tool_registry or ToolRegistry(),
        user_resolver=SimpleUserResolver(),
        agent_memory=DemoAgentMemory(max_items=100),
        config=AgentConfig(
            enable_pre_llm_workflow=enable_pre_llm_workflow,
            stream_responses=False,
        ),
        pre_llm_workflow_executor=pre_llm_workflow_executor,
    )


async def drain_agent(agent: Agent, message: str = "Show sales by day") -> None:
    request_context = RequestContext(cookies={}, headers={})
    async for _component in agent.send_message(request_context, message):
        pass


@pytest.mark.asyncio
async def test_agent_attaches_pre_llm_workflow_metadata_to_llm_request() -> None:
    llm = RecordingLlmService()
    workflow_executor = StaticWorkflowExecutor()
    agent = create_agent(
        llm_service=llm,
        pre_llm_workflow_executor=workflow_executor,
    )

    await drain_agent(agent)

    assert len(workflow_executor.inputs) == 1
    assert workflow_executor.inputs[0].original_message == "Show sales by day"
    assert len(llm.requests) == 1

    metadata = llm.requests[0].metadata["pre_llm_workflow"]
    assert metadata["status"] == "success"
    assert metadata["intent"] == "sql"
    assert metadata["structured_output"]["target_entity"] == "sales_order"
    assert metadata["prompt_metadata"] == {"prompt_hint": "use_structured_question"}
    assert metadata["request_metadata"] == {
        "request_hint": "attach_to_llm_request"
    }
    assert metadata["debug_metadata"] == {"debug_hint": "test_executor"}


@pytest.mark.asyncio
async def test_agent_does_not_run_workflow_when_disabled() -> None:
    llm = RecordingLlmService()
    workflow_executor = StaticWorkflowExecutor()
    agent = create_agent(
        llm_service=llm,
        pre_llm_workflow_executor=workflow_executor,
        enable_pre_llm_workflow=False,
    )

    await drain_agent(agent)

    assert workflow_executor.inputs == []
    assert len(llm.requests) == 1
    assert llm.requests[0].metadata == {}


@pytest.mark.asyncio
async def test_agent_continues_llm_request_when_workflow_raises() -> None:
    llm = RecordingLlmService()
    workflow_executor = FailingWorkflowExecutor()
    agent = create_agent(
        llm_service=llm,
        pre_llm_workflow_executor=workflow_executor,
    )

    await drain_agent(agent)

    assert len(workflow_executor.inputs) == 1
    assert len(llm.requests) == 1
    metadata = llm.requests[0].metadata["pre_llm_workflow"]
    assert metadata["status"] == "failed"
    assert metadata["structured_output"] is None
    assert metadata["errors"] == ["Pre-LLM workflow failed: workflow boom"]


@pytest.mark.asyncio
async def test_agent_attaches_workflow_metadata_to_tool_context() -> None:
    tool = CaptureContextTool()
    tools = ToolRegistry()
    tools.register_local_tool(tool, access_groups=["user"])

    llm = RecordingLlmService(
        responses=[
            LlmResponse(
                content="calling tool",
                tool_calls=[
                    ToolCall(id="call_1", name="capture_context", arguments={})
                ],
            ),
            LlmResponse(content="done"),
        ]
    )
    agent = create_agent(
        llm_service=llm,
        pre_llm_workflow_executor=StaticWorkflowExecutor(),
        tool_registry=tools,
    )

    await drain_agent(agent)

    assert tool.captured_metadata is not None
    workflow_metadata = tool.captured_metadata["pre_llm_workflow"]
    assert workflow_metadata["status"] == "success"
    assert workflow_metadata["structured_output"]["target_entity"] == "sales_order"
    assert tool.captured_metadata["structured_question"] == {
        "intent": "sql",
        "target_entity": "sales_order",
    }
