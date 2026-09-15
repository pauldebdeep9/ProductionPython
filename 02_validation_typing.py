"""Topic 2: Runtime validation and practical typing.

Type annotations help developers and static analysis tools understand code, but
Python does not automatically enforce them at runtime. Runtime validation has a
different job: protecting application boundaries from untrusted data.

    user / API / database / LLM data
                    |
                    v
           Pydantic validation
                    |
                    v
        typed application objects -> business logic

This deterministic tutorial simulates an AI question-answering service. It does
not call an LLM or require credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, assert_never

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


Priority = Literal["low", "normal", "high"]


class Citation(BaseModel):
    """A nested piece of evidence returned by an external system."""

    document_id: str = Field(min_length=1)
    # None means that the source legitimately has no page number.
    page: int | None = Field(default=None, ge=1)


class Answer(BaseModel):
    """Validated structured output at the AI-service boundary."""

    answer: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(min_length=1)


class QuestionRequest(BaseModel):
    """Constrained input prevents ambiguous priority values."""

    question: str = Field(min_length=1)
    priority: Priority = "normal"
    requester_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalStats:
    """A dataclass is enough for trusted state created inside the application."""

    documents_found: int
    elapsed_ms: float


class LLMClient(Protocol):
    """The small behaviour that question-answering code depends on."""

    def generate(self, prompt: str) -> str:
        """Return generated text for a prompt."""
        ...


class FakeLocalClient:
    """A structural match for LLMClient; inheritance is unnecessary."""

    def generate(self, prompt: str) -> str:
        return f"Fake answer for: {prompt}"


def ask_question(client: LLMClient, question: str) -> str:
    """Depend on required behaviour rather than a concrete provider class."""
    return client.generate(question)


class AnswerResult(BaseModel):
    kind: Literal["answer"]
    answer: str


class ClarificationRequired(BaseModel):
    kind: Literal["clarification"]
    question: str


class RejectedRequest(BaseModel):
    kind: Literal["rejected"]
    reason: str


Result = Annotated[
    AnswerResult | ClarificationRequired | RejectedRequest,
    Field(discriminator="kind"),
]
RESULT_ADAPTER: TypeAdapter[Result] = TypeAdapter(Result)


def calculate_cost(tokens: int, price_per_token: float) -> float:
    """Annotations describe the intended contract; they do not enforce it."""
    return tokens * price_per_token


def print_validation_errors(label: str, error: ValidationError) -> None:
    """Show concise boundary failures without echoing raw external input."""
    print(f"{label}: rejected")
    for detail in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in detail["loc"]) or "object"
        print(f"  {location}: {detail['msg']}")


def demo_annotations_are_not_validation() -> None:
    print("\nExample A - annotation is not runtime validation")
    unvalidated_tokens = 1_000.0
    # A static checker flags this float, but Python still calls the function.
    cost = calculate_cost(unvalidated_tokens, 0.002)  # type: ignore[arg-type]
    print(f"Runtime accepted {type(unvalidated_tokens).__name__}; cost = ${cost:.2f}")


def demo_valid_external_data() -> None:
    print("\nExample B - untrusted dictionary becomes a validated Answer")
    raw: dict[str, object] = {
        "answer": "Line 2 is delayed because material X is unavailable.",
        "confidence": 0.91,
        "citations": [{"document_id": "production-log-17", "page": 3}],
    }

    answer = Answer.model_validate(raw)
    citation = answer.citations[0]
    print("dict -> Pydantic validation -> Answer")
    print(f"Accepted: confidence={answer.confidence:.0%}")
    print(f"Evidence: {citation.document_id}, page={citation.page}")

    stats = RetrievalStats(documents_found=len(answer.citations), elapsed_ms=8.4)
    print(f"Trusted internal stats: {stats}")


def demo_invalid_external_data() -> None:
    print("\nExample C - malformed external data")
    malformed = {
        "answer": "This looks plausible but violates the schema.",
        "confidence": 1.7,
        "citations": [{"document_id": "production-log-17", "page": -3}],
    }

    try:
        Answer.model_validate(malformed)
    except ValidationError as error:
        print_validation_errors("Malformed LLM object", error)
    else:
        raise AssertionError("invalid confidence and page should be rejected")


def demo_llm_json_validation() -> None:
    print("\nExample D - JSON text returned by a fake LLM")
    valid_json = """
    {
      "answer": "Production can continue on Line 1.",
      "confidence": 0.87,
      "citations": [{"document_id": "shift-report-8", "page": null}]
    }
    """
    answer = Answer.model_validate_json(valid_json)
    print(f"Valid JSON + valid schema: accepted ({answer.confidence:.0%})")

    invalid_schema_json = """
    {
      "answer": "Production can continue on Line 1.",
      "confidence": 4.2,
      "citations": [{"document_id": "shift-report-8", "page": 2}]
    }
    """
    try:
        Answer.model_validate_json(invalid_schema_json)
    except ValidationError as error:
        print_validation_errors("Valid JSON + invalid schema", error)
    else:
        raise AssertionError("out-of-range LLM confidence should be rejected")


def demo_constrained_values() -> None:
    print("\nExample E - Literal constrains request priority")
    request = QuestionRequest(
        question="Which production line is available?",
        priority="high",
        requester_id=None,
    )
    print(f"Accepted priority: {request.priority}; requester: {request.requester_id}")

    try:
        QuestionRequest(question="Status?", priority="urgent")  # type: ignore[arg-type]
    except ValidationError as error:
        print_validation_errors("Unknown priority", error)
    else:
        raise AssertionError("an unknown priority should be rejected")


def demo_protocol() -> None:
    print("\nExample F - Protocol describes a small provider contract")
    client = FakeLocalClient()
    print(ask_question(client, "Summarize the production status."))


def render_result(result: Result) -> str:
    """Handle every known result variant explicitly."""
    if isinstance(result, AnswerResult):
        return f"Answer: {result.answer}"
    if isinstance(result, ClarificationRequired):
        return f"Clarify: {result.question}"
    if isinstance(result, RejectedRequest):
        return f"Rejected: {result.reason}"
    assert_never(result)


def demo_typed_variants() -> None:
    print("\nExample G - discriminated results and exhaustive handling")
    raw_results = (
        {"kind": "answer", "answer": "Line 1 is available."},
        {"kind": "clarification", "question": "Which site do you mean?"},
        {"kind": "rejected", "reason": "The request is outside policy."},
    )

    for raw in raw_results:
        result = RESULT_ADAPTER.validate_python(raw)
        print(render_result(result))


def main() -> None:
    """Run deterministic, offline typing and validation examples."""
    print("Type annotations guide tools; runtime validation protects boundaries.")
    demo_annotations_are_not_validation()
    demo_valid_external_data()
    demo_invalid_external_data()
    demo_llm_json_validation()
    demo_constrained_values()
    demo_protocol()
    demo_typed_variants()


if __name__ == "__main__":
    main()
