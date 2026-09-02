from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncGenerator

import pytest

TESTS_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_PATH.parent / "src"))

from vanna import Agent, AgentConfig
from vanna.core.agent.agent import (
    _can_compact_workflow_tool_result,
    _workflow_context_has_tool_result,
)
from vanna.core.llm import LlmRequest, LlmResponse, LlmService, LlmStreamChunk
from vanna.core.registry import ToolRegistry
from vanna.core.storage import Conversation
from vanna.core.user import User
from vanna.core.user.request_context import RequestContext
from vanna.core.user.resolver import UserResolver
from vanna.integrations.local import MemoryConversationStore
from vanna.integrations.local.agent_memory import DemoAgentMemory


class StaticUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        return User(
            id="test_user",
            email="test@example.com",
            group_memberships=["user"],
        )


class RecordingLlmService(LlmService):
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(content="done")

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        self.requests.append(request)
        yield LlmStreamChunk(content="done", finish_reason="stop")

    async def validate_tools(self, tools: list[Any]) -> list[str]:
        return []


class RecordingConversationStore(MemoryConversationStore):
    def __init__(self, *, fail_on_user_snapshot: bool = False) -> None:
        super().__init__()
        self.fail_on_user_snapshot = fail_on_user_snapshot
        self.saved_roles: list[list[str]] = []

    async def update_conversation(self, conversation: Conversation) -> None:
        roles = [message.role for message in conversation.messages]
        self.saved_roles.append(roles)
        if self.fail_on_user_snapshot and "user" in roles:
            raise RuntimeError("user snapshot persistence failed")
        await super().update_conversation(conversation)


class StubTurnStateStore:
    async def get_next_turn_number(self, conversation_id: str) -> int:
        return 1

    async def load_history(
        self, *, conversation_id: str, current_turn_number: int
    ) -> dict[str, list[Any]]:
        return {"latest": [], "recent": [], "older": []}


class FailingMainWorkflowExecutor:
    def __init__(self, conversation_store: RecordingConversationStore) -> None:
        self.conversation_store = conversation_store
        self.calls = 0
        self.roles_seen_at_run: list[str] | None = None

    async def run(self, workflow_input: Any) -> Any:
        self.calls += 1
        self.roles_seen_at_run = list(self.conversation_store.saved_roles[-1])
        raise RuntimeError("stop after persistence check")


def create_agent(
    *,
    llm_service: RecordingLlmService,
    conversation_store: RecordingConversationStore,
    main_workflow_executor: FailingMainWorkflowExecutor,
) -> Agent:
    agent = Agent(
        llm_service=llm_service,
        tool_registry=ToolRegistry(),
        user_resolver=StaticUserResolver(),
        agent_memory=DemoAgentMemory(max_items=100),
        conversation_store=conversation_store,
        config=AgentConfig(
            auto_save_conversations=False,
            stream_responses=False,
        ),
        main_workflow_executor=main_workflow_executor,
    )
    agent.turn_state_store = StubTurnStateStore()
    return agent


async def drain_agent(agent: Agent) -> None:
    request_context = RequestContext(cookies={}, headers={})
    async for _component in agent.send_message(request_context, "current question"):
        pass


@pytest.mark.asyncio
async def test_persists_current_user_before_main_workflow_without_auto_save() -> None:
    llm = RecordingLlmService()
    conversation_store = RecordingConversationStore()
    main_workflow_executor = FailingMainWorkflowExecutor(conversation_store)
    agent = create_agent(
        llm_service=llm,
        conversation_store=conversation_store,
        main_workflow_executor=main_workflow_executor,
    )

    await drain_agent(agent)

    assert main_workflow_executor.roles_seen_at_run == ["user"]
    stored = next(iter(conversation_store._conversations.values()))
    assert [message.role for message in stored.messages].count("user") == 1


@pytest.mark.asyncio
async def test_persistence_failure_stops_before_main_workflow_and_llm() -> None:
    llm = RecordingLlmService()
    conversation_store = RecordingConversationStore(fail_on_user_snapshot=True)
    main_workflow_executor = FailingMainWorkflowExecutor(conversation_store)
    agent = create_agent(
        llm_service=llm,
        conversation_store=conversation_store,
        main_workflow_executor=main_workflow_executor,
    )

    await drain_agent(agent)

    assert main_workflow_executor.calls == 0
    assert llm.requests == []


def test_tool_result_compression_requires_refreshed_payload() -> None:
    skipped_with_results = {
        "subflows": {"data_discovery": {"status": "skipped"}},
        "metadata": {"candidates": [{"table_name": "HCRS_G1_A18"}]},
        "fewshot": [{"question": "q", "args": {"sql": "SELECT 1"}}],
    }

    assert _workflow_context_has_tool_result(
        skipped_with_results, "search_business_metadata"
    )
    assert _workflow_context_has_tool_result(
        skipped_with_results, "search_saved_correct_tool_uses"
    )
    assert not _workflow_context_has_tool_result(
        {"metadata": {"candidates": []}}, "search_business_metadata"
    )
    assert not _workflow_context_has_tool_result(
        {"fewshot": []}, "search_saved_correct_tool_uses"
    )
    assert not _workflow_context_has_tool_result(
        {"metadata": {"candidates": [{}]}}, "search_business_metadata"
    )
    assert not _workflow_context_has_tool_result(
        {"fewshot": [{}]}, "search_saved_correct_tool_uses"
    )
    assert _can_compact_workflow_tool_result(
        workflow_context_refresh_succeeded=True,
        workflow_metadata=skipped_with_results,
        tool_name="search_business_metadata",
        tool_succeeded=True,
    )
    assert not _can_compact_workflow_tool_result(
        workflow_context_refresh_succeeded=True,
        workflow_metadata={"metadata": {"candidates": []}},
        tool_name="search_business_metadata",
        tool_succeeded=True,
    )
    assert not _can_compact_workflow_tool_result(
        workflow_context_refresh_succeeded=False,
        workflow_metadata=skipped_with_results,
        tool_name="search_business_metadata",
        tool_succeeded=True,
    )
