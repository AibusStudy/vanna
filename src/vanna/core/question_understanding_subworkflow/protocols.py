"""Integration protocols implemented outside vanna core."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class IntentClassifier(Protocol):
    """Classifies whether a user question should use general or SQL intent."""

    async def classify_intent(
        self,
        question: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        ...


@runtime_checkable
class TimeNormalizer(Protocol):
    """Normalizes relative and absolute time expressions in a user question."""

    async def normalize_time(
        self,
        question: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...


@runtime_checkable
class QuestionStructurer(Protocol):
    """Builds schema-compliant structured question output."""

    async def structure_question(
        self,
        question: str,
        *,
        intent: Optional[str] = None,
        normalized_time: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...


@runtime_checkable
class StructuredQuestionValidator(Protocol):
    """Validates structured question output before downstream use."""

    async def validate_structured_question(
        self,
        structured_question: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...