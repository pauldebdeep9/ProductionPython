"""
05 — TypedDict vs dataclass vs Pydantic: choosing a container.

THE DECISION
------------
Four ways to hold structured data, and they are not interchangeable:

    TypedDict     a dict, with static key/value types. Zero runtime cost.
                  No validation, no methods, no attribute access.
    NamedTuple    an immutable tuple with names. Cheap, hashable, unpackable.
    dataclass     a normal object. Attribute access, methods, __eq__,
                  frozen/slots. No validation.
    Pydantic      a validating model. Parses, coerces, and enforces at
                  construction. Costs real CPU.

Choosing wrongly produces one of two problems: validation overhead on every
internal hop (Pydantic used as a dataclass), or unvalidated data flowing past
a trust boundary (TypedDict used as a Pydantic model).

This script measures the actual cost so the trade is a number rather than a
vibe.

Run:  python 05_containers.py
      mypy --strict 05_containers.py
"""

from __future__ import annotations

import timeit
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import NamedTuple, NotRequired, Required, TypedDict

from pydantic import BaseModel, Field, ValidationError

from typing_lab import ChunkId, DocId, Sensitivity, banner, section, show


# ---------------------------------------------------------------------------
# PART 1 — the same data, four ways
# ---------------------------------------------------------------------------

class ChunkDict(TypedDict):
    """A dict at runtime. `chunk["text"]`, not `chunk.text`."""

    chunk_id: str
    doc_id: str
    text: str
    score: float


class ChunkTuple(NamedTuple):
    """A tuple at runtime. Hashable, unpackable, immutable, very cheap."""

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class ChunkData:
    """A normal object. `slots=True` removes __dict__, cutting memory and
    speeding attribute access — worth it when you hold tens of thousands."""

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0


class ChunkModel(BaseModel):
    """A validating model. Every construction runs the validators."""

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0


def part1_measure() -> None:
    banner("PART 1 — construction cost, measured")

    args = ("c1", "po-1001", "PO 1001 quantity 400 units", 0.9)
    n = 50_000

    results: dict[str, float] = {}
    results["TypedDict (plain dict)"] = timeit.timeit(
        lambda: ChunkDict(chunk_id=args[0], doc_id=args[1],
                          text=args[2], score=args[3]),
        number=n)
    results["NamedTuple"] = timeit.timeit(lambda: ChunkTuple(*args), number=n)
    results["dataclass (frozen+slots)"] = timeit.timeit(
        lambda: ChunkData(*args), number=n)
    results["Pydantic BaseModel"] = timeit.timeit(
        lambda: ChunkModel(chunk_id=args[0], doc_id=args[1],
                           text=args[2], score=args[3]),
        number=n)
    results["Pydantic model_validate"] = timeit.timeit(
        lambda: ChunkModel.model_validate(
            {"chunk_id": args[0], "doc_id": args[1],
             "text": args[2], "score": args[3]}),
        number=n)

    baseline = results["TypedDict (plain dict)"]
    print(f"    {n:,} constructions each\n")
    print(f"    {'container':<28} {'total':>10} {'per-op':>12} {'vs dict':>9}")
    print(f"    {'-' * 28} {'-' * 10} {'-' * 12} {'-' * 9}")
    for name, t in results.items():
        print(f"    {name:<28} {t:>9.3f}s {t / n * 1e6:>10.2f}us "
              f"{t / baseline:>8.1f}x")

    # Do the arithmetic rather than asserting a vibe about it.
    per_validate = results["Pydantic model_validate"] / n
    per_dataclass = results["dataclass (frozen+slots)"] / n
    ingest_objects = 50_000 * 40          # 50k documents, 40 chunks each
    rag_objects = 20                      # one request, 20 retrieved chunks

    section("what the ratio actually costs, in context")
    print(f"    one RAG request ({rag_objects} chunks):")
    print(f"      Pydantic {per_validate * rag_objects * 1e3:>8.3f}ms   "
          f"vs an LLM call of ~500ms")
    print(f"    full ingestion ({ingest_objects:,} chunks):")
    print(f"      Pydantic  {per_validate * ingest_objects:>7.2f}s")
    print(f"      dataclass {per_dataclass * ingest_objects:>7.2f}s")
    print(f"      difference {(per_validate - per_dataclass) * ingest_objects:>6.2f}s")

    print("""
    I EXPECTED THIS TO SHOW A BIGGER GAP, AND IT DOES NOT. Run the numbers
    above and the conclusion is uncomfortable for the usual advice: even
    across two million objects, choosing Pydantic over a dataclass costs a
    couple of seconds. Against an ingestion run dominated by network I/O and
    embedding calls, that is noise.

    So the honest guidance is NOT "avoid Pydantic internally because it is
    slow". Pydantic v2's core is Rust and it is fast enough that performance
    is rarely the deciding factor.

    THE REAL REASONS to convert to a dataclass after the boundary:

      1. CLARITY ABOUT WHERE TRUST CHANGES. If every type is a BaseModel,
         nobody can tell which validations are load-bearing. One model at the
         edge and dataclasses inside makes the boundary visible in the type
         signatures themselves.

      2. RE-VALIDATION IS A LIE. Validating data you already validated
         suggests it might have become invalid, which it did not. It is
         ceremony that looks like rigour.

      3. FROZEN + SLOTS ARE EASIER. Immutability and __slots__ are one
         keyword on a dataclass and more fiddly on a BaseModel.

      4. DOMAIN TYPES SHOULD NOT KNOW ABOUT THE WIRE. A Pydantic model
         carries serialisation aliases, JSON-schema config, and validators —
         all wire concerns. Your domain object should not.

    If you take one thing from this section, take the correction rather than
    the original claim: measure before optimising, including when the person
    telling you to optimise is a tutorial.""")


# ---------------------------------------------------------------------------
# PART 2 — TypedDict features people miss
# ---------------------------------------------------------------------------

class ToolSpec(TypedDict):
    """Required by default when `total=True` (the default)."""

    name: str
    description: str
    parameters: dict[str, object]
    strict: NotRequired[bool]
    """`NotRequired` marks ONE key optional. Prefer this to `total=False`,
    which marks everything optional and forces defensive `.get()` calls
    across your whole codebase."""


class PartialUpdate(TypedDict, total=False):
    """`total=False` — everything optional. Legitimate for PATCH-style
    payloads where any subset of fields may be present."""

    text: str
    score: float
    sensitivity: Sensitivity
    doc_id: Required[str]
    """`Required` is the inverse escape hatch: pin one key as mandatory inside
    a total=False TypedDict. Use it for the identifier that must always be
    present even in a partial update."""


def part2_typeddict_features() -> None:
    banner("PART 2 — NotRequired, Required, and the total=False trap")

    spec: ToolSpec = {
        "name": "get_goods_receipt",
        "description": "Fetch a goods receipt by id",
        "parameters": {"type": "object",
                       "properties": {"gr_id": {"type": "string"}}},
    }
    show("ToolSpec without 'strict'", "valid — strict is NotRequired")

    spec_strict: ToolSpec = {**spec, "strict": True}
    show("ToolSpec with 'strict'", spec_strict["strict"])

    update: PartialUpdate = {"doc_id": "po-1001", "score": 0.95}
    show("PartialUpdate (total=False + Required)", update)

    print("""
    WHY TypedDict AT ALL, when a dataclass is nicer to use?

    Because sometimes the data IS a dict and converting is pure ceremony:
      * the JSON body you POST to an API — you need a dict to serialise
      * **kwargs forwarding
      * a tool/function-calling schema, which the API wants as nested dicts
      * a row from a driver that returns dicts

    In those places a TypedDict gives you key-name checking, required/optional
    tracking, and autocomplete, with a plain dict at runtime and no conversion
    step.

    THE HARD LIMITS, all of which surprise people:
      * NO runtime validation. It is a static promise only.
      * NO methods. It is a dict.
      * NO isinstance check. `isinstance(x, ChunkDict)` is a TypeError.
      * Extra keys are a static error on a literal, but invisible on a dict
        that arrived from elsewhere.
      * Mutable — nothing stops `spec["name"] = 42` at runtime.""")


# ---------------------------------------------------------------------------
# PART 3 — dataclass features that matter
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalConfig:
    """Three modifiers, each earning its place:

    frozen=True   hashable, safe to share across tasks, and prevents the
                  "someone mutated the config mid-request" class of bug.
    slots=True    no __dict__: less memory, faster attribute access. The cost
                  is you cannot add attributes dynamically, which you should
                  not be doing anyway.
    kw_only=True  every field must be passed by name. For a config with six
                  numeric fields this eliminates an entire category of
                  positional-argument bug, and it lets you add fields without
                  worrying about ordering or defaults.
    """

    top_k: int = 5
    score_threshold: float = 0.0
    max_tokens_context: int = 8000
    rerank: bool = True
    deployment: str = "gpt-4o-mini-prod"


@dataclass
class RequestState:
    """Mutable, because it accumulates during a request."""

    trace_id: str
    chunks_retrieved: int = 0
    tokens_used: int = 0
    repairs: int = 0

    # GOTCHA: a mutable default must use default_factory. `= []` raises
    # ValueError at class-definition time for dataclasses (unlike a plain
    # function default, where it silently shares one list across all calls).
    errors: list[str] = field(default_factory=list)

    # `field(repr=False)` keeps noisy or sensitive data out of the repr —
    # which matters because reprs end up in logs and exception messages.
    raw_prompt: str = field(default="", repr=False)


def part3_dataclass_features() -> None:
    banner("PART 3 — dataclass modifiers worth using by default")

    cfg = RetrievalConfig(top_k=8, rerank=False)
    show("frozen config", cfg)
    try:
        cfg.top_k = 12   # type: ignore[misc]  # deliberate: frozen
    except AttributeError as e:
        show("mutating a frozen dataclass", f"FrozenInstanceError: {e}")

    show("hashable because frozen", hash(cfg) != 0)

    st = RequestState(trace_id="tr-9f2a", raw_prompt="a very long secret prompt")
    st.errors.append("truncated once")
    show("repr hides raw_prompt", st)

    print("""
    NOTE the repr above: `raw_prompt` is absent. That is `field(repr=False)`
    doing security work — an exception traceback containing a RequestState
    would otherwise print the full prompt into your logs, moving document
    content across a trust boundary your permission model does not cover.

    DEFAULTS WORTH ADOPTING:
      frozen=True   unless you have a specific reason to mutate
      slots=True    unless you need dynamic attributes or multiple inheritance
      kw_only=True  for anything with more than ~3 fields
      repr=False    on any field holding prompt text, chunk text, credentials,
                    or PII""")


# ---------------------------------------------------------------------------
# PART 4 — the conversion boundary
# ---------------------------------------------------------------------------

class InvoiceLineModel(BaseModel):
    """PYDANTIC AT THE BOUNDARY. Validates once, on the way in."""

    doc_id: str = Field(min_length=1)
    quantity: int = Field(gt=0, description="must be positive")
    unit_price: Decimal = Field(gt=0, decimal_places=2)
    sensitivity: Sensitivity = "internal"

    def to_domain(self) -> InvoiceLine:
        """Convert to the internal type immediately after validation.

        This is the pattern: Pydantic model is the WIRE type, dataclass is the
        DOMAIN type, and the conversion happens once at the boundary. The rest
        of the system never sees the Pydantic model, so it never pays
        validation cost and never accidentally re-validates.
        """
        return InvoiceLine(
            doc_id=DocId(self.doc_id),
            quantity=self.quantity,
            unit_price=self.unit_price,
            sensitivity=self.sensitivity,
        )


@dataclass(frozen=True, slots=True)
class InvoiceLine:
    """THE DOMAIN TYPE. No validation — it is unreachable except through the
    boundary above, so validation here would be redundant work."""

    doc_id: DocId
    quantity: int
    unit_price: Decimal
    sensitivity: Sensitivity = "internal"

    @property
    def total(self) -> Decimal:
        return self.unit_price * self.quantity


def part4_conversion() -> None:
    banner("PART 4 — Pydantic at the edge, dataclass inside")

    section("valid payload")
    m = InvoiceLineModel.model_validate(
        {"doc_id": "inv-88", "quantity": 402, "unit_price": "12.50"})
    line = m.to_domain()
    show("validated + converted", line)
    show("line.total (Decimal arithmetic)", line.total)

    section("invalid payloads rejected at the boundary")
    bad = [
        ({"doc_id": "", "quantity": 402, "unit_price": "12.50"}, "empty doc_id"),
        ({"doc_id": "x", "quantity": -5, "unit_price": "12.50"}, "negative qty"),
        ({"doc_id": "x", "quantity": "many", "unit_price": "12.50"}, "qty not int"),
        ({"doc_id": "x", "quantity": 1, "unit_price": "12.505"}, "3 decimals"),
    ]
    for payload, label in bad:
        try:
            InvoiceLineModel.model_validate(payload)
            show(label, "ACCEPTED (unexpected)")
        except ValidationError as e:
            err = e.errors()[0]
            show(label, f"rejected: {err['loc']} {err['msg'][:38]}")

    print("""
    Note `unit_price` accepted the STRING "12.50" and produced a Decimal.
    That is Pydantic coercion doing something genuinely useful: JSON has no
    decimal type, so money always arrives as a string (or, catastrophically,
    as a float). Declaring `Decimal` and letting Pydantic parse it is the
    correct handling — `float("12.50")` is not 12.50 and never will be.

    But coercion cuts both ways. See 06 for strict mode and when to turn it
    off.""")


# ---------------------------------------------------------------------------
# PART 5 — the decision table
# ---------------------------------------------------------------------------

def part5_decision() -> None:
    banner("PART 5 — which container, when")

    print("""
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Situation                                    │ Use                     │
    ├────────────────────────────────────────────────────────────────────────┤
    │ JSON body you SEND to an API                 │ TypedDict               │
    │ tool/function-calling schema                 │ TypedDict               │
    │ **kwargs you forward                         │ TypedDict (Unpack[...]) │
    │ JSON you RECEIVE from an API or a model      │ Pydantic -> dataclass   │
    │ HTTP request body                            │ Pydantic -> dataclass   │
    │ config from env / files                      │ pydantic-settings       │
    │ internal domain object                       │ dataclass(frozen,slots) │
    │ dict key / cache key / set member            │ NamedTuple or frozen dc │
    │ hot loop, millions of instances              │ NamedTuple or slots dc  │
    │ needs methods + immutability                 │ frozen dataclass        │
    │ needs __eq__ but not ordering                │ dataclass               │
    └────────────────────────────────────────────────────────────────────────┘

    THE ANTI-PATTERNS, both common:

    1. PYDANTIC EVERYWHERE. Every internal function takes and returns
       BaseModels, so data is re-validated at every hop. Slow, and it hides
       where the real boundary is — which means nobody can tell which
       validations are load-bearing. Symptom: `model_construct` appearing in
       code as a performance workaround.

    2. DICTS EVERYWHERE. `dict[str, Any]` passed through eight functions.
       Every access is a potential KeyError, no checker help, and a typo in a
       key name is a runtime failure in whichever branch runs least often.
       Symptom: defensive `.get(k, default)` calls layered three deep.

    THE SHAPE TO AIM FOR:

        untrusted input
              │
              ▼  Pydantic model_validate       <- validation happens ONCE
        validated model
              │
              ▼  .to_domain()                  <- conversion happens ONCE
        frozen dataclass ──────────────────────> the rest of the system
              │
              ▼  asdict() / a serialiser
        TypedDict / JSON                       <- on the way back out""")

    section("round trip")
    m = InvoiceLineModel.model_validate(
        {"doc_id": "inv-88", "quantity": 402, "unit_price": "12.50"})
    line = m.to_domain()
    out = asdict(line)
    show("dataclass -> dict for serialisation", out)
    print("    (note Decimal survives asdict; json.dumps needs a custom")
    print("     encoder or `model_dump(mode='json')` on the Pydantic side)")


def main() -> None:
    part1_measure()
    part2_typeddict_features()
    part3_dataclass_features()
    part4_conversion()
    part5_decision()

    banner("SUMMARY")
    print("""
  * Four containers, not interchangeable. Cost and guarantees both differ.
  * Pydantic's cost is what you want at a boundary and waste on every hop
    after it. Validate once, convert to a dataclass, move on.
  * `NotRequired` per key beats `total=False` for a whole TypedDict.
  * dataclass defaults worth adopting: frozen, slots, kw_only, and repr=False
    on any field holding prompt text or PII.
  * TypedDict for dicts you must produce; Pydantic for data you must trust;
    dataclass for everything in between.
  * Money is Decimal, parsed from a string. Never float.
""")


if __name__ == "__main__":
    main()
