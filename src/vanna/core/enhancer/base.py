"""
LLM context enhancer interface.

LLM context enhancers allow you to add additional context to the system prompt
and user messages before LLM calls.
"""

from abc import ABC
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..user.models import User
    from ..llm.models import LlmMessage


class LlmContextEnhancer(ABC):
    """Enhancer for adding context to LLM prompts and messages."""

    async def enhance_system_prompt(
        self, system_prompt: str, user_message: str, user: "User"
    ) -> str:
        return system_prompt

    async def enhance_user_messages(
        self, messages: list["LlmMessage"], user: "User"
    ) -> list["LlmMessage"]:
        return messages

    async def enhance_system_prompt_with_workflow(
        self,
        system_prompt: str,
        user_message: str,
        user: "User",
        workflow_result: Any = None,
        workflow_state: Any = None,
        workflow_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return await self.enhance_system_prompt(
            system_prompt,
            user_message,
            user,
        )
