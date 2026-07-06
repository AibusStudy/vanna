"""
Agent module.

This module contains the core Agent implementation and configuration.
"""

from .agent import Agent
from .config import AgentConfig
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from vanna.core.pre_llm_workflow import (
    PreLlmWorkflowExecutor,
    WorkflowFinalResult,
    WorkflowInput,
)

pre_llm_workflow_executor: Optional[PreLlmWorkflowExecutor] = None,

__all__ = ["Agent", "AgentConfig"]
self.pre_llm_workflow_executor = pre_llm_workflow_executor