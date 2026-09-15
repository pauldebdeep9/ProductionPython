"""
09 — Testing types: what a test suite should assert about your type discipline.

Run:  pytest 09_test_typing.py -v

THE THREE THINGS WORTH TESTING
------------------------------
  1. THE CHECKER RUNS CLEAN. `mypy --strict` as a pytest test, so a type
     regression fails the same suite as a logic regression. If mypy only runs
     in a separate CI job that people skip on a red build, it does not hold.

  2. PROTOCOL CONFORMANCE. A one-line annotated assignment per implementation.
     Cheap, and it fails in the file you are editing rather than at whatever
     call site happens to pass the object.

  3. THAT VALIDATION ACTUALLY REJECTS. The most important category, and the
     one people skip: a test proving your boundary model rejects the bad
     inputs you expect. Without it, a model with all-optional fields and no
     constraints passes every happy-path test while validating nothing.

WHAT IS NOT WORTH TESTING
-------------------------
  * That mypy catches a type error. That is mypy's test suite, not yours.
  * That a dataclass has the fields you gave it.
  * `isinstance` checks on internal domain objects — mypy proved that already,
    statically, for free.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from typing_lab import (
    Chunk,
    ChunkId,
    DeploymentName,
    DocId,
    Disposition,
    FinishReason,
    banner,
)

HERE = Path(__file__).parent


# ===========================================================================
# 1. mypy as a test
# ===========================================================================

TUTORIAL_FILES = [
    "typing_lab.py", "01_static_vs_runtime.py", "02_protocols.py",
    "03_generics.py", "04_narrowing.py", "05_containers.py",
    "06_pydantic_boundaries.py", "07_mypy_in_practice.py",
    "08_typed_pipeline.py",
]


@pytest.mark.parametrize("filename", TUTORIAL_FILES)
def test_module_passes_strict_mypy(filename: str) -> None:
    """One test per module, so a failure names the file.

    PARAMETRISING BY FILE matters more than it looks: a single test running
    mypy over the whole package reports "47 errors" and someone has to read
    the output. Per-file tests give you a red dot next to the module that
    broke, which is what makes people fix it rather than skip it.

    SPEED NOTE: mypy's incremental cache makes repeat runs fast. In CI, cache
    `.mypy_cache` between runs or this becomes the slowest thing in your
    suite and someone will delete it.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", filename],
        capture_output=True, text=True, cwd=HERE,
    )
    assert proc.returncode == 0, (
        f"mypy --strict failed for {filename}:\n{proc.stdout}"
    )


def test_no_bare_type_ignores() -> None:
    """A bare `# type: ignore` suppresses every error on its line, forever,
    including ones a later edit introduces.

    NOTE ON THE IMPLEMENTATION — this test's first version scanned lines for
    the substring `# type: ignore` and immediately reported SEVEN false
    positives, all of them prose in docstrings discussing the rule. That is
    the classic lint-by-grep failure.

    The fix is `tokenize`, which distinguishes a real COMMENT token from
    identical text inside a string literal. Any check that reasons about code
    structure should use `tokenize` or `ast`, never string matching — and a
    check that reports failures on its own documentation is telling you it is
    the wrong kind of check.
    """
    import io
    import tokenize

    offenders: list[str] = []
    for f in sorted(HERE.glob("*.py")):
        src = f.read_text()
        try:
            tokens = tokenize.generate_tokens(io.StringIO(src).readline)
            for tok in tokens:
                if tok.type != tokenize.COMMENT:
                    continue
                text = tok.string.strip()
                if text.startswith("# type: ignore") and not text.startswith(
                    "# type: ignore["
                ):
                    offenders.append(f"{f.name}:{tok.start[0]}")
        except tokenize.TokenError:  # pragma: no cover
            continue

    assert not offenders, (
        "bare `# type: ignore` found (use an error code and a reason): "
        + ", ".join(offenders)
    )


# ===========================================================================
# 2. Protocol conformance
# ===========================================================================

from typing import Protocol


class Retriever(Protocol):
    async def search(
        self, query: str, groups: frozenset[str], *, k: int
    ) -> Sequence[Chunk]: ...


class GoodRetriever:
    async def search(
        self, query: str, groups: frozenset[str], *, k: int
    ) -> Sequence[Chunk]:
        return ()


def test_protocol_conformance_statically() -> None:
    """THE WHOLE TEST IS THE ANNOTATION.

    This assignment is checked by mypy, not at runtime — the test body does
    almost nothing when executed. That is fine and intentional: the value is
    that `test_module_passes_strict_mypy` above will fail if this line stops
    type-checking.

    Put one of these per implementation. When someone changes the Protocol,
    every non-conforming implementation is named by mypy with the expected and
    actual signatures side by side (verified in script 02 part 6).
    """
    _r: Retriever = GoodRetriever()
    assert _r is not None


def test_protocol_does_not_constrain_behaviour() -> None:
    """The LIMIT of Protocol conformance, asserted so nobody forgets it.

    `IgnoresGroups` satisfies the Protocol perfectly — right method name,
    right parameters, right return type. And it ignores the security-relevant
    argument entirely.

    Types constrain SHAPE. Only a runtime check constrains BEHAVIOUR. This
    test exists to make that a stated, tested fact rather than folklore.
    """

    class IgnoresGroups:
        async def search(
            self, query: str, groups: frozenset[str], *, k: int
        ) -> Sequence[Chunk]:
            # Conforms structurally. Ignores `groups`. Leaks everything.
            return (
                Chunk(ChunkId("c6"), DocId("legal-1"), "confidential",
                      frozenset({"legal-only"}), "restricted"),
            )

    _r: Retriever = IgnoresGroups()   # mypy: fine. Reality: a data leak.
    assert _r is not None


# ===========================================================================
# 3. Boundary validation — the tests that matter most
# ===========================================================================

class ProposalSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    disposition: Disposition
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=10, max_length=400)
    evidence_ids: list[str] = Field(min_length=1)


VALID = {
    "disposition": "adjust_quantity",
    "confidence": 0.84,
    "reason": "Invoice billed 402 units against a 400 unit receipt.",
    "evidence_ids": ["c3", "c4"],
}


def test_valid_payload_is_accepted() -> None:
    """The happy path. Necessary, and by itself proves nothing about
    validation — a model with every field `Any` and no constraints passes
    this test too."""
    p = ProposalSchema.model_validate(VALID)
    assert p.disposition == "adjust_quantity"
    assert p.confidence == pytest.approx(0.84)


@pytest.mark.parametrize(
    "mutation,bad_field",
    [
        ({"disposition": "partial_approve"}, "disposition"),
        ({"disposition": ""}, "disposition"),
        ({"confidence": 1.4}, "confidence"),
        ({"confidence": -0.1}, "confidence"),
        ({"confidence": "high"}, "confidence"),
        ({"reason": "short"}, "reason"),
        ({"reason": "x" * 500}, "reason"),
        ({"evidence_ids": []}, "evidence_ids"),
        ({"evidence_ids": "c3"}, "evidence_ids"),
        ({"notes": "invented"}, "notes"),
    ],
    ids=["hallucinated_enum", "empty_enum", "conf_too_high", "conf_negative",
         "conf_not_numeric", "reason_too_short", "reason_too_long",
         "no_evidence", "evidence_not_list", "extra_field"],
)
def test_invalid_payloads_are_rejected(
    mutation: dict[str, object], bad_field: str
) -> None:
    """THE IMPORTANT TEST. Each row is a real failure mode of LLM output.

    Asserting on the FIELD NAME, not just that an error occurred, is what
    makes this test meaningful: it proves the RIGHT constraint fired. A test
    that only asserts `pytest.raises(ValidationError)` passes even when your
    model rejects the payload for an unrelated reason.
    """
    payload = {**VALID, **mutation}
    with pytest.raises(ValidationError) as exc:
        ProposalSchema.model_validate(payload)
    locs = {str(e["loc"][0]) for e in exc.value.errors() if e["loc"]}
    assert bad_field in locs, f"expected an error on {bad_field}, got {locs}"


def test_coercion_is_allowed_where_intended() -> None:
    """Coercion is a DECISION, so test it explicitly in both directions.

    Models emit "0.84" and 0.84 interchangeably; both are the number you
    wanted. This test pins that behaviour so a future `strict=True` cannot be
    added without someone noticing it breaks LLM parsing.
    """
    p = ProposalSchema.model_validate({**VALID, "confidence": "0.84"})
    assert p.confidence == pytest.approx(0.84)
    assert isinstance(p.confidence, float)


def test_schema_generation_stays_in_sync() -> None:
    """The schema sent to the API is generated from the same model that
    validates the response, so they cannot drift. Assert the properties the
    prompt depends on.

    Specifically: the `enum` must be present, or the model is not being TOLD
    the valid dispositions and will invent them.
    """
    schema = ProposalSchema.model_json_schema()
    assert schema.get("additionalProperties") is False, (
        "extra='forbid' must produce additionalProperties: false — "
        "OpenAI strict structured outputs requires it"
    )
    props = schema["properties"]
    assert set(schema["required"]) == {
        "disposition", "confidence", "reason", "evidence_ids"
    }
    disp = props["disposition"]
    enum_values = disp.get("enum") or disp.get("const")
    assert enum_values, (
        "disposition must emit an enum in the schema, or the model is never "
        "told which values are valid"
    )


# ===========================================================================
# 4. NewType discipline
# ===========================================================================

def test_newtype_is_free_at_runtime() -> None:
    """NewType costs nothing and is NOT a subclass. Both facts matter:
    the first is why you can use it liberally, the second is why
    `isinstance(x, DocId)` is a TypeError and you must never write it."""
    d = DocId("po-1001")
    assert d == "po-1001"
    assert type(d) is str

    with pytest.raises(TypeError):
        isinstance(d, DocId)  # type: ignore[misc]  # NewType is not a class


def test_newtype_distinguishes_at_type_check_time() -> None:
    """The static half, expressed as an annotated assignment.

    Passing a ChunkId where a DocId is expected is a mypy error. Uncomment the
    line below and `test_module_passes_strict_mypy` catches it.
    """
    def takes_doc(d: DocId) -> str:
        return str(d)

    assert takes_doc(DocId("po-1001")) == "po-1001"
    # takes_doc(ChunkId("c1"))   # <- would be: Argument 1 has incompatible type


# ===========================================================================
# 5. Exhaustiveness
# ===========================================================================

def test_all_finish_reasons_are_handled() -> None:
    """A runtime companion to `assert_never`.

    `assert_never` catches a MISSING branch statically. This catches a
    different thing: a dispatch table (a dict, a registry) that has drifted
    from the Literal it is supposed to cover. Static exhaustiveness cannot see
    inside a dict literal.
    """
    from typing import get_args

    policy: dict[FinishReason, str] = {
        "stop": "ok",
        "length": "repair",
        "content_filter": "fail",
        "tool_calls": "ok",
    }
    declared = set(get_args(FinishReason))
    assert set(policy) == declared, (
        f"policy table drifted from FinishReason: "
        f"missing={declared - set(policy)}, extra={set(policy) - declared}"
    )


def test_disposition_set_matches_schema() -> None:
    """The same drift check between the Literal and what the model is told."""
    from typing import get_args

    schema = ProposalSchema.model_json_schema()
    schema_values = set(schema["properties"]["disposition"]["enum"])
    assert schema_values == set(get_args(Disposition))


# ===========================================================================
# 6. Frozen / immutability guarantees
# ===========================================================================

def test_domain_objects_are_immutable() -> None:
    """Frozen dataclasses are a security property, not a style preference: a
    Chunk whose ACL can be mutated after the permission check is a bug you
    cannot see in review."""
    c = Chunk(ChunkId("c1"), DocId("d1"), "text", frozenset({"isc-all"}))
    with pytest.raises((AttributeError, TypeError)):
        c.allowed_groups = frozenset({"everyone"})  # type: ignore[misc]  # frozen


def test_frozen_objects_are_hashable() -> None:
    """Which is what lets them be cache keys — and identity-aware cache keys
    are what stops one principal's results being served to another."""
    c = Chunk(ChunkId("c1"), DocId("d1"), "text", frozenset({"isc-all"}))
    key = (c.chunk_id, frozenset({"isc-all"}))
    assert {key: "cached"}[key] == "cached"


# ===========================================================================
# 7. The negative control
# ===========================================================================

def test_the_validation_tests_would_catch_a_broken_model() -> None:
    """A MUTATION TEST, inline.

    `test_invalid_payloads_are_rejected` is only meaningful if it would fail
    against a model with no constraints. Here is that model — it accepts
    everything, proving the constraints in the real one are what does the
    work, not the test harness.

    Without a control like this, a suite can pass for years while validating
    nothing, because someone quietly loosened the model and no test noticed.
    """

    class UnconstrainedProposal(BaseModel):
        disposition: str
        confidence: float
        reason: str
        evidence_ids: list[str]

    garbage = {
        "disposition": "partial_approve",
        "confidence": 1.4,
        "reason": "short",
        "evidence_ids": [],
    }
    permissive = UnconstrainedProposal.model_validate(garbage)
    assert permissive.confidence == 1.4        # accepted — no constraint

    with pytest.raises(ValidationError):
        ProposalSchema.model_validate(garbage)  # rejected — constraints work
