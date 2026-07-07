"""Run a fake pre-LLM workflow and print node/edge logs.

Usage:
    python tests\\pre_llm_workflow_fakes\\run_fake_workflow.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
TESTS_PATH = REPO_ROOT / "tests"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(TESTS_PATH))

from vanna.core.pre_llm_workflow import PreLlmWorkflowExecutor, WorkflowInput

from pre_llm_workflow_fakes.fake_graph import build_fake_pre_llm_graph


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    graph = build_fake_pre_llm_graph(intent="sql")
    executor = PreLlmWorkflowExecutor(graph, max_steps=10, retry_limit=1)

    result = await executor.run(
        WorkflowInput(
            user_id="fake_user",
            conversation_id="fake_conversation",
            request_id="fake_request",
            original_message="Show me sales by day for this week",
            system_prompt="You are a SQL assistant.",
            tool_names=["run_sql"],
            metadata={"source": "fake_run"},
        )
    )

    print("\nFinal result:")
    print(result.to_metadata())


if __name__ == "__main__":
    asyncio.run(main())
