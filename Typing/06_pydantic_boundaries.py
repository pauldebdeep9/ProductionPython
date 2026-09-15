"""
06 — Pydantic at the boundary: validating what you cannot trust.

THE FOUR BOUNDARIES, CONCRETELY
-------------------------------
    1. HTTP request      someone else's client
    2. LLM output        a stochastic process that has never read your schema
    3. Search index      documents ingested at some point by some pipeline
    4. Config / env      a YAML file someone edited under time pressure

Boundary 2 is the one this script spends the most time on, because it is both
the least reliable input in an LLM system and the least rigorously validated
in most codebases. And it has a property the others do not: the schema you
declare can be SENT TO THE MODEL, so validation and prompting are the same
artefact. Get that right and a whole class of failure disappears.

Run:  python 06_pydantic_boundaries.py
      mypy --strict 06_pydantic_boundaries.py
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Tag,
    ValidationError,
    field_validator,
    model_validator,
)

from typing_lab import Disposition, banner, section, show


# ---------------------------------------------------------------------------
# PART 1 — coercion: useful, and dangerous
# ---------------------------------------------------------------------------

class LaxModel(BaseModel):
    """Default (lax) mode. Pydantic coerces where it reasonably can."""

    count: int
    score: float
    enabled: bool


class StrictModel(BaseModel):
    """`strict=True` disables coercion. Types must match exactly."""

    model_config = ConfigDict(strict=True)

    count: int
    score: float
    enabled: bool


def part1_coercion() -> None:
    banner("PART 1 — coercion is a feature and a footgun")

    payloads = [
        {"count": 5, "score": 0.9, "enabled": True},
        {"count": "5", "score": "0.9", "enabled": "true"},
        {"count": 5.0, "score": 1, "enabled": 1},
        {"count": 5.7, "score": 0.9, "enabled": True},
    ]

    print(f"    {'payload':<44} {'lax':<22} strict")
    print(f"    {'-' * 44} {'-' * 22} {'-' * 14}")
    for p in payloads:
        try:
            lax = str(LaxModel.model_validate(p).model_dump())
        except ValidationError as e:
            lax = f"rejected ({e.error_count()})"
        try:
            StrictModel.model_validate(p)
            strict = "accepted"
        except ValidationError:
            strict = "rejected"
        print(f"    {str(p):<44} {lax[:22]:<22} {strict}")

    print("""
    Row 2 is coercion earning its place: JSON has no integer/float
    distinction worth trusting and no decimal type, so `"5"` and `"0.9"`
    arriving as strings is normal, not an error.

    Row 4 is the footgun: `count: 5.7` is REJECTED even in lax mode, because
    Pydantic v2 refuses lossy float->int conversion. (v1 would have silently
    truncated to 5. If you are migrating, this is a behaviour change worth
    knowing about.)

    WHEN TO USE STRICT MODE:
      * internal service-to-service traffic where both sides are yours and a
        type mismatch is a bug you want reported, not papered over
      * anything financial, where a silently coerced type is a real risk
      * when you want a type mismatch to page someone rather than proceed

    WHEN NOT TO:
      * LLM output. Models emit "0.85" and 0.85 interchangeably, and both are
        the number you wanted. Strict mode here turns a free fix into a
        repair call. Coerce, then validate the RANGE.
      * browser form data, where everything is a string by construction

    PER-FIELD CONTROL: `Field(strict=True)` on individual fields lets you be
    strict where it matters and lax where it does not, which is usually the
    right shape.""")


# ---------------------------------------------------------------------------
# PART 2 — the structured-output model
# ---------------------------------------------------------------------------

class DispositionProposal(BaseModel):
    """The LLM boundary model. THREE JOBS AT ONCE:

      1. Validates the model's JSON output.
      2. GENERATES the JSON Schema you send to the API for structured
         outputs / function calling.
      3. Documents the contract for humans reading the code.

    Because (1) and (2) come from the same declaration, they cannot drift.
    That is the single biggest practical win of using Pydantic here rather
    than hand-written validation plus a hand-written schema: the classic bug
    where the prompt says one thing and the parser expects another simply
    cannot occur.
    """

    model_config = ConfigDict(
        extra="forbid",
        # `extra="forbid"` matters more than it looks. A model that invents an
        # extra field is telling you it misunderstood the task; silently
        # dropping it hides that signal. It is also REQUIRED for OpenAI strict
        # structured outputs, which needs additionalProperties: false.
        str_strip_whitespace=True,
    )

    disposition: Disposition = Field(
        description="The action to take on this invoice exception."
    )
    """Note the Literal type does double duty: it validates the value AND
    emits an `enum` in the JSON Schema, so the model is TOLD the valid set
    instead of being left to guess. That single fact removes most
    hallucinated-enum failures at the source rather than repairing them."""

    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description="0.0-1.0. Use <0.6 if the evidence is ambiguous."
    )

    reason: str = Field(
        min_length=10, max_length=500,
        description="One or two sentences citing the specific discrepancy.",
    )

    evidence_ids: list[str] = Field(
        min_length=1,
        description="Chunk ids supporting this decision. Must be non-empty.",
    )
    """`min_length=1` encodes a policy: no proposal without evidence. That is
    a business rule, enforced by the type, checked at the boundary — rather
    than a sentence in a prompt that the model may ignore."""

    @field_validator("evidence_ids")
    @classmethod
    def ids_must_look_like_chunks(cls, v: list[str]) -> list[str]:
        """A field validator for a rule the type system cannot express.

        THE REAL POINT: this catches HALLUCINATED CITATIONS. A model that
        invents `["source_1", "the goods receipt"]` instead of using the ids
        you gave it produces an answer that looks cited and is not.
        """
        bad = [x for x in v if not x.startswith(("c", "doc-", "inv-", "gr-"))]
        if bad:
            raise ValueError(
                f"evidence ids do not match any known chunk id format: {bad}"
            )
        return v

    @model_validator(mode="after")
    def high_confidence_needs_more_evidence(self) -> Self:
        """A model validator sees ALL fields, so it can enforce
        cross-field rules. Runs after field validation, hence `mode="after"`.

        This one encodes: you may not be 95% sure on the basis of one chunk.
        """
        if self.confidence >= 0.95 and len(self.evidence_ids) < 2:
            raise ValueError(
                "confidence >= 0.95 requires at least 2 pieces of evidence"
            )
        return self


def part2_structured_output() -> None:
    banner("PART 2 — one model: validation AND the schema you send")

    section("the JSON Schema generated from the model")
    schema = DispositionProposal.model_json_schema()
    print(json.dumps(
        {"properties": {k: {kk: vv for kk, vv in v.items()
                            if kk in ("type", "enum", "minimum", "maximum",
                                      "minLength", "maxLength", "description")}
                        for k, v in schema["properties"].items()},
         "required": schema["required"],
         "additionalProperties": schema.get("additionalProperties")},
        indent=2)[:900])

    print("""
    That schema goes straight into the API call:

        response_format={"type": "json_schema",
                         "json_schema": {"name": "disposition",
                                         "strict": True,
                                         "schema": Model.model_json_schema()}}

    With strict structured outputs the model is CONSTRAINED to emit conforming
    JSON, which eliminates the parse-failure and hallucinated-enum categories
    entirely. You still validate on receipt — constrained decoding guarantees
    shape, not semantic correctness, and you may be talking to an endpoint or
    model version that does not support it.""")

    section("validating real-world model outputs")
    outputs = {
        "well-formed": '{"disposition":"adjust_quantity","confidence":0.82,'
                       '"reason":"Invoice billed 402 against a 400 unit receipt.",'
                       '"evidence_ids":["c3","c4"]}',
        "confidence as string": '{"disposition":"adjust_quantity","confidence":"0.82",'
                                '"reason":"Invoice billed 402 against 400 received.",'
                                '"evidence_ids":["c3"]}',
        "hallucinated enum": '{"disposition":"partial_approve","confidence":0.6,'
                             '"reason":"Partially acceptable given the variance.",'
                             '"evidence_ids":["c3"]}',
        "out of range": '{"disposition":"approve","confidence":1.4,'
                        '"reason":"Everything matches the purchase order fine.",'
                        '"evidence_ids":["c1"]}',
        "hallucinated citation": '{"disposition":"approve","confidence":0.7,'
                                 '"reason":"The goods receipt confirms the count.",'
                                 '"evidence_ids":["the goods receipt"]}',
        "no evidence": '{"disposition":"approve","confidence":0.7,'
                       '"reason":"Looks fine to me on balance, no issues.",'
                       '"evidence_ids":[]}',
        "overconfident": '{"disposition":"approve","confidence":0.99,'
                         '"reason":"Everything matches the purchase order fine.",'
                         '"evidence_ids":["c1"]}',
        "extra field": '{"disposition":"approve","confidence":0.7,'
                       '"reason":"Everything matches the purchase order fine.",'
                       '"evidence_ids":["c1"],"notes":"also I invented this"}',
    }
    for label, raw in outputs.items():
        try:
            p = DispositionProposal.model_validate_json(raw)
            show(label, f"OK  {p.disposition} @ {p.confidence}")
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(x) for x in first["loc"]) or "(model)"
            show(label, f"REJECT  {loc}: {first['msg'][:44]}")


# ---------------------------------------------------------------------------
# PART 3 — error messages the model can act on
# ---------------------------------------------------------------------------

def repair_hint(exc: ValidationError) -> str:
    """Turn a ValidationError into a REPAIR PROMPT.

    A raw Pydantic error dump is verbose, structured for humans, and full of
    `url: https://errors.pydantic.dev/...` noise. Feeding it straight back to
    the model wastes context and reads worse than one imperative sentence.

    Compress it: field, what was wrong, what to do. This function is the
    bridge between the typing world and the repair loop from the
    failure-handling tutorial.
    """
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(whole object)"
        if err["type"] == "literal_error":
            lines.append(
                f"- '{loc}': {err.get('input')!r} is not permitted. "
                f"Allowed values: {err.get('ctx', {}).get('expected', '')}"
            )
        elif err["type"] == "too_short":
            ctx = err.get("ctx", {})
            min_len = ctx.get("min_length", ctx.get("actual_length"))
            lines.append(f"- '{loc}': needs at least {min_len} item(s).")
        elif err["type"] == "string_too_short":
            # BUG I SHIPPED FIRST TIME: this originally said "must not be
            # empty", which is wrong and actively misleading for a
            # `min_length=10` field — the model would return a 3-character
            # string and fail again. Always report the actual constraint.
            min_len = err.get("ctx", {}).get("min_length", "?")
            lines.append(
                f"- '{loc}': too short; must be at least {min_len} characters."
            )
        elif err["type"] == "string_too_long":
            max_len = err.get("ctx", {}).get("max_length", "?")
            lines.append(
                f"- '{loc}': too long; must be at most {max_len} characters."
            )
        elif err["type"] in ("greater_than_equal", "less_than_equal"):
            lines.append(f"- '{loc}': {err.get('input')!r} is out of range. "
                         f"{err['msg']}.")
        elif err["type"] == "extra_forbidden":
            lines.append(f"- '{loc}': unexpected field; remove it.")
        else:
            lines.append(f"- '{loc}': {err['msg']}.")
    return (
        "Your previous JSON did not satisfy the schema:\n"
        + "\n".join(lines)
        + "\nReturn ONLY a corrected JSON object."
    )


def part3_repair_hints() -> None:
    banner("PART 3 — ValidationError -> a prompt the model can act on")

    bad = ('{"disposition":"partial_approve","confidence":1.4,'
           '"reason":"short","evidence_ids":[]}')
    try:
        DispositionProposal.model_validate_json(bad)
    except ValidationError as e:
        show("raw error count", e.error_count())
        print("\n    generated repair prompt:\n")
        for line in repair_hint(e).splitlines():
            print(f"      {line}")

    print("""
    Every problem named, in the model's own terms, in about 200 tokens.
    Compare with `str(exc)`, which for this input is several hundred tokens of
    JSON paths and documentation URLs.

    This is where the typing tutorial and the failure-handling tutorial meet:
    the Pydantic model defines the contract, the ValidationError says exactly
    how the output violated it, and that becomes the REPAIR — a different
    request, not a retry of the same one.""")


# ---------------------------------------------------------------------------
# PART 4 — discriminated unions of tool calls
# ---------------------------------------------------------------------------

class GetGoodsReceipt(BaseModel):
    tool: Literal["get_goods_receipt"]
    gr_id: str = Field(pattern=r"^GR-\d{4}$")


class GetPurchaseOrder(BaseModel):
    tool: Literal["get_purchase_order"]
    po_id: str = Field(pattern=r"^PO-\d{4}$")


class WriteProposal(BaseModel):
    tool: Literal["write_proposal"]
    case_id: str
    disposition: Disposition
    idempotency_key: str = Field(min_length=8)
    """The ONE write tool requires an idempotency key, enforced by the type.

    A tool call missing it does not validate, so it cannot reach the executor.
    That is a safety invariant expressed in the type system rather than in a
    code review comment — and it is checked on data produced by a model, which
    is exactly where you want a machine enforcing the rule."""


ToolCall = Annotated[
    Annotated[GetGoodsReceipt, Tag("get_goods_receipt")]
    | Annotated[GetPurchaseOrder, Tag("get_purchase_order")]
    | Annotated[WriteProposal, Tag("write_proposal")],
    Field(discriminator="tool"),
]
"""A DISCRIMINATED union. Pydantic reads `tool` first and validates against
only that member.

WHY THE DISCRIMINATOR MATTERS, beyond speed: without it, Pydantic tries every
member and reports the failures from ALL of them. For a 3-member union that is
three times the error text, and the error you actually needed is buried. With
it, you get "unknown tool 'x'" or the specific member's errors, and nothing
else."""


class ToolCallEnvelope(BaseModel):
    calls: list[ToolCall]


def part4_tool_unions() -> None:
    banner("PART 4 — discriminated unions for tool calls")

    payloads = {
        "valid read": '{"calls":[{"tool":"get_goods_receipt","gr_id":"GR-0055"}]}',
        "valid write": '{"calls":[{"tool":"write_proposal","case_id":"EXC-001",'
                       '"disposition":"approve","idempotency_key":"EXC-001:tr-9f2a"}]}',
        "write without key": '{"calls":[{"tool":"write_proposal","case_id":"EXC-001",'
                             '"disposition":"approve"}]}',
        "malformed id": '{"calls":[{"tool":"get_goods_receipt","gr_id":"55"}]}',
        "unknown tool": '{"calls":[{"tool":"delete_everything","target":"*"}]}',
    }
    for label, raw in payloads.items():
        try:
            env = ToolCallEnvelope.model_validate_json(raw)
            show(label, f"OK  {type(env.calls[0]).__name__}")
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(x) for x in first["loc"])
            show(label, f"REJECT  {loc}: {first['msg'][:40]}")

    print("""
    Read the last two rows. A model that hallucinates a tool name, or emits a
    malformed identifier, is rejected AT THE BOUNDARY — before an executor
    ever sees it. The tool registry does not need a defensive
    `if name not in TOOLS` check, because a non-conforming call cannot be
    constructed.

    And 'write without key' is rejected by the same mechanism that rejects an
    unknown tool. The idempotency requirement is now structural.""")


# ---------------------------------------------------------------------------
# PART 5 — the `from __future__ import annotations` interaction
# ---------------------------------------------------------------------------

def part5_future_annotations() -> None:
    banner("PART 5 — the postponed-annotations gotcha")

    print("""
    Every file in this tutorial starts with `from __future__ import
    annotations`, which turns annotations into strings. Pydantic must resolve
    those strings at class-creation time via `typing.get_type_hints()`, in the
    MODULE'S namespace.

    THE FAILURE:

        from __future__ import annotations
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from myapp.domain import Chunk     # not imported at runtime!

        class Response(BaseModel):
            chunk: Chunk                       # NameError at class creation

    mypy is perfectly happy. Pydantic raises
    `PydanticUndefinedAnnotation: name 'Chunk' is not defined`.

    THE FIXES, in order of preference:
      1. Import the type at RUNTIME for anything a Pydantic model references.
         `TYPE_CHECKING` imports are for annotations nothing inspects.
      2. `Model.model_rebuild()` after the import is available — the standard
         escape hatch for genuine circular imports.
      3. Restructure so the circular import goes away.

    THE SAME TRAP applies to anything that reads annotations at runtime:
    FastAPI route signatures, `dataclasses.fields()` with `get_type_hints`,
    attrs validators, and `typing.get_type_hints` in your own code.

    RULE OF THUMB: if a library INSPECTS your annotations, every name in them
    must exist at runtime. If only a type checker reads them, TYPE_CHECKING is
    free.""")

    section("demonstrating a forward reference that DOES work")

    class Node(BaseModel):
        """Self-reference resolves fine: `Node` is in the module namespace by
        the time validation happens."""

        name: str
        children: list[Node] = Field(default_factory=list)

    tree = Node.model_validate(
        {"name": "root", "children": [{"name": "a"}, {"name": "b"}]})
    show("recursive model", f"{tree.name} -> {[c.name for c in tree.children]}")


# ---------------------------------------------------------------------------
# PART 6 — what NOT to validate
# ---------------------------------------------------------------------------

def part6_restraint() -> None:
    banner("PART 6 — where validation is ceremony")

    print("""
    DO NOT VALIDATE:

      * Data you validated one function ago. If `resolve()` takes a
        `DispositionProposal` that only `parse()` can produce, re-validating
        inside `resolve()` asserts that your own code corrupted it.

      * Internal function arguments. That is what mypy is for, and it is free.
        A runtime type check on a private method is a test you are running in
        production.

      * Test fixtures. `model_construct()` skips validation and is the right
        call for building known-good fixtures quickly.

    DO VALIDATE, always:

      * Anything deserialised: JSON, YAML, pickle, msgpack, a DB row.
      * Anything a model produced.
      * Anything from an environment variable — AT STARTUP, so a bad config
        fails immediately rather than on the first request that touches it.
      * Anything crossing a trust boundary in either direction, including
        data you are about to WRITE somewhere durable. A malformed record in
        your index outlives the bug that created it.

    THE TEST: could this data have been shaped by anything outside this
    process? If yes, validate. If no, annotate and let mypy do it.

    ONE MORE, easy to miss: validate at STARTUP, not lazily. A pydantic-settings
    model constructed at import time turns 'we have been running for six hours
    and just discovered AZURE_OPENAI_DEPLOYMENT is empty' into a container that
    refuses to start. Fail fast, at the earliest possible moment.""")


def main() -> None:
    part1_coercion()
    part2_structured_output()
    part3_repair_hints()
    part4_tool_unions()
    part5_future_annotations()
    part6_restraint()

    banner("SUMMARY")
    print("""
  * One Pydantic model does three jobs: validates output, GENERATES the JSON
    Schema you send, and documents the contract. They cannot drift apart.
  * A Literal field emits an `enum` in the schema, so the model is told the
    valid set instead of guessing. Removes hallucinated enums at the source.
  * `extra="forbid"` — an invented field is a signal the model misunderstood;
    do not silently drop it. Also required for OpenAI strict mode.
  * Field constraints encode POLICY (`min_length=1` on evidence_ids = no
    proposal without evidence) and validators catch hallucinated citations.
  * Compress ValidationError into a short imperative repair prompt. That is
    the join between typing and the repair loop.
  * Discriminated unions for tool calls: unknown tools and missing idempotency
    keys are rejected before an executor sees them.
  * Coerce for LLM output; be strict for money and internal service traffic.
  * With `from __future__ import annotations`, every name a Pydantic model
    references must exist at RUNTIME, not just under TYPE_CHECKING.
  * Validate at boundaries and at startup. Never in the middle.
""")


if __name__ == "__main__":
    main()
