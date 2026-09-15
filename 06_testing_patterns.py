"""Topic 6: Testing production and AI code.

Production behaviour becomes testable when dependencies are controllable and
scenarios are deterministic. Ordinary tests verify software behaviour, contract
tests protect boundary compatibility, and AI evaluations measure quality against
acceptance criteria.

Run this file in either mode:

    python 06_testing_patterns.py
    pytest -q 06_testing_patterns.py

No test makes a network call, uses credentials, or depends on model randomness.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Literal, Protocol
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field


ConfidenceLevel = Literal["low", "medium", "high"]


def classify_confidence(score: float) -> ConfidenceLevel:
    """Classify a validated confidence score at meaningful boundaries."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


class LLMClient(Protocol):
    """The external behaviour required by the application."""

    async def generate(self, prompt: str) -> str:
        """Generate an answer for one prompt."""
        ...


class FakeLLM:
    """A small working dependency with deterministic behaviour and call history."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


async def answer_question(question: str, client: LLMClient) -> str:
    """Tiny application workflow whose external dependency is injected."""
    if not question.strip():
        raise ValueError("question must not be empty")
    prompt = f"Answer briefly: {question}"
    return (await client.generate(prompt)).strip()


def current_request_timestamp() -> float:
    """Narrow clock boundary that production code can replace in a test."""
    return time.time()


def build_request_metadata(request_id: str) -> dict[str, str | float]:
    return {
        "request_id": request_id,
        "received_at": current_request_timestamp(),
    }


class AnswerPayload(BaseModel):
    """The response shape our application expects from a provider."""

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


def keyword_recall(answer: str, required_terms: set[str]) -> float:
    """Toy quality metric for teaching—not a production LLM evaluator."""
    if not required_terms:
        return 1.0
    answer_words = {
        word.strip(".,:;!?").casefold() for word in answer.split() if word.strip()
    }
    expected = {term.casefold() for term in required_terms}
    return len(answer_words & expected) / len(expected)


# ---------------------------------------------------------------------------
# Tests: pytest discovers these functions when this file is passed explicitly.
# ---------------------------------------------------------------------------


def test_basic_confidence_classification() -> None:
    # Arrange
    score = 0.92

    # Act
    result = classify_confidence(score)

    # Assert externally meaningful behaviour
    assert result == "high"


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.95, "high"),
        (0.80, "high"),
        (0.65, "medium"),
        (0.50, "medium"),
        (0.20, "low"),
    ],
)
def test_confidence_boundaries(
    score: float,
    expected: ConfidenceLevel,
) -> None:
    """One parametrized test replaces five nearly identical test functions."""
    assert classify_confidence(score) == expected


@pytest.mark.parametrize("invalid_score", [-0.1, 1.1])
def test_invalid_confidence_raises(invalid_score: float) -> None:
    """An expected exception is correct behaviour, not a broken test."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        classify_confidence(invalid_score)


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A fixture provides reusable setup with a fresh fake for each test."""
    return FakeLLM("Production is on schedule.")


def test_async_question_workflow_with_fake(fake_llm: FakeLLM) -> None:
    result = asyncio.run(answer_question("Is Line 1 available?", fake_llm))

    # Prefer the returned behaviour over brittle assertions about private helpers.
    assert result == "Production is on schedule."
    # This interaction matters: the service must actually send the user's question.
    assert len(fake_llm.calls) == 1
    assert "Is Line 1 available?" in fake_llm.calls[0]


def test_timestamp_is_patched_at_narrow_boundary() -> None:
    """Patch the clock boundary rather than mocking the application call chain."""
    module = sys.modules[__name__]
    with patch.object(
        module, "current_request_timestamp", return_value=1_700_000_000.0
    ):
        metadata = build_request_metadata("req-test-001")

    assert metadata == {
        "request_id": "req-test-001",
        "received_at": 1_700_000_000.0,
    }


def test_representative_provider_payload_contract() -> None:
    """Check boundary shape without making a real provider request."""
    representative_payload: dict[str, object] = {
        "answer": "Line 2 is delayed.",
        "confidence": 0.91,
    }

    validated = AnswerPayload.model_validate(representative_payload)

    assert validated.answer == "Line 2 is delayed."
    assert validated.confidence == pytest.approx(0.91)


def test_toy_ai_evaluation_meets_acceptance_criterion() -> None:
    """Evaluate answer quality without requiring one exact valid sentence."""
    candidate = "The delay is caused by a material shortage."
    required_terms = {"material", "delay"}

    score = keyword_recall(candidate, required_terms)

    assert score >= 0.5


def run_tutorial_summary() -> None:
    """Show the testing portfolio without manually invoking pytest tests."""
    fake = FakeLLM("Production is on schedule.")
    fake_result = asyncio.run(answer_question("Is Line 1 available?", fake))
    evaluation_score = keyword_recall(
        "The delay is caused by a material shortage.",
        {"material", "delay"},
    )
    contract = AnswerPayload.model_validate(
        {"answer": "Line 2 is delayed.", "confidence": 0.91}
    )

    print("Testing mental model")
    print("Unit test: deterministic software behaviour")
    print("Contract test: boundary/interface compatibility")
    print("AI evaluation: quality against an acceptance criterion")
    print(f"Fake LLM result: {fake_result}")
    print(f"Representative contract confidence: {contract.confidence:.2f}")
    print(f"Toy evaluation score: {evaluation_score:.2f}")
    print("Fake = working deterministic dependency; mock = narrow substitution.")
    print(
        "Portfolio: many unit tests, fewer contract/integration tests, few end-to-end."
    )
    print("AI evaluations complement tests; they do not replace them.")
    print("Ordinary unit tests avoid real APIs, credentials, cost, and flaky outputs.")
    print("\nRun the full tests with:\npytest -q 06_testing_patterns.py")


if __name__ == "__main__":
    run_tutorial_summary()
