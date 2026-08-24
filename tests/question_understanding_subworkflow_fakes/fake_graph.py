"""Fake graph factory for local Question-Understanding workflow smoke tests."""

from __future__ import annotations

from vanna.core.question_understanding_subworkflow import WorkflowGraph

from .fake_nodes import (
    FakeGeneralFinishNode,
    FakeIntentNode,
    FakeQuestionStructurerNode,
    FakeStructuredQuestionValidatorNode,
    FakeTimeNormalizerNode,
    IntentIs,
)


def build_fake_question_understanding_subworkflow_graph(intent: str = "sql") -> WorkflowGraph:
    graph = WorkflowGraph()

    graph.add_node(FakeIntentNode(intent=intent), start=True)
    graph.add_node(FakeGeneralFinishNode(), end=True)
    graph.add_node(FakeTimeNormalizerNode())
    graph.add_node(FakeQuestionStructurerNode())
    graph.add_node(FakeStructuredQuestionValidatorNode(), end=True)

    graph.add_edge(
        "intent_classification",
        "general_finish",
        condition=IntentIs("general"),
        label="general_intent",
    )
    graph.add_edge(
        "intent_classification",
        "time_normalization",
        condition=IntentIs("sql"),
        label="sql_intent",
    )
    graph.add_edge(
        "time_normalization",
        "question_structuring",
        label="time_normalized",
    )
    graph.add_edge(
        "question_structuring",
        "structured_json_validation",
        label="structured",
    )

    return graph

