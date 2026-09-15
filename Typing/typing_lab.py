"""
typing_lab.py — shared domain types for the typing tutorial.

WHY A SHARED DOMAIN MODULE
--------------------------
Typing lessons taught on `def add(a: int, b: int) -> int` do not transfer. The
questions that actually matter — where to put a Protocol, when a TypedDict
beats a dataclass, which boundary needs runtime validation — only have answers
in the context of a real system with real boundaries.

So everything here models one system: a permission-aware RAG service with an
agent that calls tools. It has the four boundaries that matter:

    1. WIRE IN     untrusted JSON from an HTTP client
    2. MODEL OUT   untrusted JSON from an LLM (the boundary people forget)
    3. STORE       rows/documents from a search index
    4. INTERNAL    your own code talking to itself

Static typing is load-bearing at boundary 4 and useless at 1-3. Runtime
validation is load-bearing at 1-3 and mostly noise at 4. That split is the
whole tutorial.

NOTE ON `from __future__ import annotations`
-------------------------------------------
Present in every file here. It makes all annotations lazily-evaluated strings,
which buys:
  * forward references without quotes (`def f() -> Node:` before Node exists)
  * `X | Y` and `list[str]` syntax on older runtimes
  * no import-time cost for typing-only imports

And costs one real thing: anything that INSPECTS annotations at runtime —
Pydantic, dataclasses with `InitVar`, FastAPI, attrs — must resolve those
strings, which it does via `typing.get_type_hints()`. That resolution happens
in the module's namespace, so a type imported only under `if TYPE_CHECKING:`
will raise `NameError` at validation time. See 06 for the concrete failure.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Literal, NewType, TypedDict


# ===========================================================================
# 1. NewType — free nominal typing over primitives
# ===========================================================================
# THE PROBLEM: every id in your system is a `str`, so the type checker
# cheerfully lets you pass a doc_id where a chunk_id was expected. That bug is
# invisible in review and produces an empty result set at runtime.
#
# NewType costs ZERO at runtime (it is the identity function; the constructor
# call is optimised away in 3.10+) but the checker treats them as distinct.
#
# GOTCHA: NewType is not a subclass. `DocId("x")` IS a `str` at runtime, so
# `isinstance(d, DocId)` is a TypeError. Use it for static discipline only.

DocId = NewType("DocId", str)
ChunkId = NewType("ChunkId", str)
TraceId = NewType("TraceId", str)
PrincipalId = NewType("PrincipalId", str)
DeploymentName = NewType("DeploymentName", str)
"""Azure OpenAI takes a DEPLOYMENT name, which is an arbitrary string someone
chose, not a model name. Making it a distinct type stops `complete(model=...)`
from silently accepting `"gpt-4o-mini"` where a deployment was required — a
real bug that only shows up in a tenant that named things differently."""


# ===========================================================================
# 2. Literal and Enum — closed sets
# ===========================================================================
# WHEN TO USE WHICH:
#   Literal  — the value IS the string. Serialises for free, no import needed
#              by consumers, exhaustiveness-checkable. Best for wire formats
#              and small closed sets.
#   Enum     — you want methods, a canonical object identity, or the set is
#              large enough that autocomplete matters. Costs a `.value` at
#              every serialisation boundary.

FinishReason = Literal["stop", "length", "content_filter", "tool_calls"]
"""Exactly the values an OpenAI-compatible API returns. Typing this as `str`
throws away the one thing you know for certain about it."""

Disposition = Literal[
    "approve", "request_credit", "adjust_quantity",
    "hold_pending_receipt", "escalate_to_buyer",
]

Sensitivity = Literal["public", "internal", "confidential", "restricted"]


class Stage(Enum):
    """An Enum because these carry behaviour (ordering) and are used as dict
    keys across modules where autocomplete is worth the import."""

    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    VERIFY = "verify"

    @property
    def is_model_call(self) -> bool:
        return self in (Stage.RERANK, Stage.GENERATE, Stage.VERIFY)


# ===========================================================================
# 3. TypedDict — the shape of JSON you do not own
# ===========================================================================
# TypedDict describes a dict's keys statically while leaving it a plain dict at
# runtime. That makes it the right tool for WIRE FORMATS: the payload you send
# to or receive from an API, where converting to an object and back would be
# pure ceremony.
#
# CRITICAL LIMITATION: a TypedDict is NOT validated. It is a promise to the
# checker, and a lie if the data does not match. Never use one to describe
# untrusted input you have not validated — use Pydantic there (see 06).

class ChatMessage(TypedDict):
    """One message in an OpenAI-compatible request body."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ToolCallPayload(TypedDict):
    """The wire shape of a tool call the model emits."""

    id: str
    name: str
    arguments: str          # JSON-encoded STRING, not an object. A classic
                            # source of bugs: the API nests JSON inside JSON.


class UsagePayload(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionPayload(TypedDict, total=False):
    """`total=False` makes every key optional.

    Prefer per-key `NotRequired[...]` in new code — `total=False` is
    all-or-nothing and usually overshoots, marking required keys optional and
    forcing defensive `.get()` calls everywhere.
    """

    id: str
    finish_reason: FinishReason
    content: str
    tool_calls: list[ToolCallPayload]
    usage: UsagePayload


# ===========================================================================
# 4. Dataclasses — internal domain objects
# ===========================================================================
# Once data has crossed a boundary and been validated, it should stop being a
# dict. Dataclasses give you attribute access, a real type, `__eq__`, and
# `frozen=True` for immutability — at near-zero cost and with no validation
# overhead on every internal hop.

@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieved passage.

    `frozen=True` because a chunk that mutates after its ACL was checked is a
    security bug waiting to happen. `slots=True` because there may be tens of
    thousands of these in memory during a rerank.
    """

    chunk_id: ChunkId
    doc_id: DocId
    text: str
    allowed_groups: frozenset[str]
    sensitivity: Sensitivity = "internal"
    score: float = 0.0

    def visible_to(self, groups: frozenset[str]) -> bool:
        return bool(self.allowed_groups & groups)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: DocId
    kind: Literal["po", "gr", "invoice"]
    quantity: int
    unit_price: Decimal      # Decimal, never float, for money


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query: str
    total_candidates: int
    trimmed_by_acl: int = 0


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    """The INTERNAL representation, after the wire payload is validated.

    Note it is a different type from CompletionPayload. That separation is
    deliberate: the wire shape can change with an API version without any
    internal code changing, because only the adapter knows both.
    """

    text: str
    finish_reason: FinishReason
    usage: Usage
    deployment: DeploymentName
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, object]     # already parsed, unlike the wire form


# ===========================================================================
# 5. Fake providers used across the scripts
# ===========================================================================

def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


CORPUS: list[Chunk] = [
    Chunk(ChunkId("c1"), DocId("po-1001"), "PO 1001 quantity 400 units",
          frozenset({"isc-all"}), "internal", 0.0),
    Chunk(ChunkId("c2"), DocId("po-1001"), "PO 1001 unit price 12.50 USD",
          frozenset({"isc-all"}), "internal", 0.0),
    Chunk(ChunkId("c3"), DocId("inv-88"), "Invoice 88 billed 402 units",
          frozenset({"isc-all"}), "internal", 0.0),
    Chunk(ChunkId("c4"), DocId("gr-55"), "Goods receipt 55 recorded 400 units",
          frozenset({"isc-all"}), "internal", 0.0),
    Chunk(ChunkId("c5"), DocId("hr-comp"), "Compensation band for plant leads",
          frozenset({"hr-only"}), "confidential", 0.0),
    Chunk(ChunkId("c6"), DocId("legal-1"), "Settlement terms, confidential",
          frozenset({"legal-only"}), "restricted", 0.0),
]


def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def section(title: str) -> None:
    print(f"\n  --- {title} ---")


def show(label: str, value: object) -> None:
    print(f"    {label:<44} {value}")
