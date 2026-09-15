"""
04 — Narrowing: unions, guards, exhaustiveness, and overloads.

THE IDEA
--------
A union type (`Chunk | None`, `ToolCall | TextResponse | Refusal`) is only
useful if the checker can figure out WHICH member you have at each point. That
figuring-out is called narrowing, and Python's type system has surprisingly
strong support for it — most of which people never use.

The payoff in an LLM system is concrete. A model response is genuinely a union:

    text answer | tool call | refusal | truncated | filtered

Modelling it as a union with a discriminant and then checking exhaustively
means that when someone adds a sixth case, the checker names every place that
has to handle it. Modelling it as `dict[str, Any]` with `if "tool_calls" in
resp:` means the sixth case is silently ignored until a user reports it.

Run:  python 04_narrowing.py
      mypy --strict 04_narrowing.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypeGuard, assert_never, overload

from typing_lab import (
    Chunk,
    ChunkId,
    CORPUS,
    Disposition,
    DocId,
    FinishReason,
    banner,
    section,
    show,
)


# ---------------------------------------------------------------------------
# PART 1 — the narrowing forms the checker understands
# ---------------------------------------------------------------------------

def part1_forms() -> None:
    banner("PART 1 — what narrows, and what silently does not")

    def find(chunk_id: str) -> Chunk | None:
        return next((c for c in CORPUS if c.chunk_id == chunk_id), None)

    # 1. `is None` / `is not None` — the workhorse.
    c = find("c1")
    if c is not None:
        show("narrowed by `is not None`", c.doc_id)

    # 2. Truthiness. Works, but is a trap for types where falsy != absent:
    #    an empty list, 0, "" are all falsy. `if not chunks:` and
    #    `if chunks is None:` mean different things and the bug is silent.
    chunks: list[Chunk] | None = None
    show("truthiness on `list | None`", "use `is None` — [] is falsy too")

    # 3. isinstance — narrows unions of classes.
    value: str | int = "hello"
    if isinstance(value, str):
        show("narrowed by isinstance", value.upper())

    # 4. Literal comparison — narrows Literal unions.
    reason: FinishReason = "length"
    if reason == "length":
        show("narrowed by `==` on a Literal", "truncation branch")

    # 5. `in` on a tuple of literals.
    if reason in ("length", "content_filter"):
        show("narrowed by `in`", "abnormal-termination branch")

    # 6. assert — narrows for the rest of the scope.
    maybe = find("c3")
    assert maybe is not None, "c3 must exist in the fixture corpus"
    show("narrowed by assert", maybe.text[:28])

    print("""
    WHAT DOES *NOT* NARROW, and catches everyone:

      a) Narrowing through a method call.
             if self.chunk is not None:
                 self.process()          # may set self.chunk = None
                 self.chunk.text         # mypy still thinks it is narrowed!
         mypy narrows ATTRIBUTES optimistically and does not re-check after
         arbitrary calls. Assign to a local first:
             chunk = self.chunk
             if chunk is not None: chunk.text

      b) Narrowing that crosses a closure or a comprehension boundary.

      c) `if len(items) > 0:` does NOT narrow `list[T] | None`. Only an
         explicit None check does.

      d) A helper function returning bool:
             def has_text(c: Chunk | None) -> bool: ...
             if has_text(c): c.text          # ERROR — bool tells mypy nothing
         This is exactly what TypeGuard / TypeIs exist to fix. See PART 3.""")


# ---------------------------------------------------------------------------
# PART 2 — discriminated unions: the right shape for a model response
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextAnswer:
    kind: Literal["text"] = "text"      # THE DISCRIMINANT
    content: str = ""
    citations: tuple[ChunkId, ...] = ()


@dataclass(frozen=True)
class ToolRequest:
    kind: Literal["tool"] = "tool"
    call_id: str = ""
    tool_name: str = ""
    arguments: str = "{}"               # JSON string, as the API sends it


@dataclass(frozen=True)
class Refusal:
    kind: Literal["refusal"] = "refusal"
    reason: str = ""


@dataclass(frozen=True)
class Truncated:
    kind: Literal["truncated"] = "truncated"
    partial: str = ""
    tokens_used: int = 0


@dataclass(frozen=True)
class Filtered:
    kind: Literal["filtered"] = "filtered"
    category: str = ""


ModelResponse = TextAnswer | ToolRequest | Refusal | Truncated | Filtered
"""The five things a model call can actually produce.

WHY THIS BEATS `dict[str, Any]`:
  * each variant carries exactly the fields it needs, and no others
  * accessing `.citations` on a Refusal is a checker error, not a KeyError
  * `assert_never` makes the set closed and additions loud
  * the discriminant makes narrowing free — one `match` and mypy knows the type

WHY IT BEATS a single class with optional fields:
    class Response:
        content: str | None
        tool_name: str | None
        refusal_reason: str | None
  ...because that shape permits nonsense — content AND refusal_reason set at
  once — and forces None checks on every field at every use.
"""


def handle_response(r: ModelResponse) -> str:
    """Exhaustive dispatch, verified by the checker.

    `assert_never` is the load-bearing line. If someone adds a sixth variant to
    ModelResponse and forgets this function, mypy reports it here.

    VERIFIED: adding a `RateLimited` variant to the union and re-running
    `mypy --strict` produces exactly one error, on the assert_never line:

        error: Argument 1 to "assert_never" has incompatible type
               "RateLimited"; expected "Never"  [arg-type]

    One error, pointing at the exact line that must change, in every function
    that dispatches on the union. Without it the new variant silently falls
    through to whatever the last branch happens to be — and in a `match` with
    no `case _`, it falls through to returning None from a function annotated
    `-> str`, which `warn_no_return` may or may not catch depending on your
    config.
    """
    match r:
        case TextAnswer(content=content, citations=cites):
            return f"answer ({len(cites)} citations): {content[:32]}"
        case ToolRequest(tool_name=name, arguments=args):
            return f"tool call: {name}({args})"
        case Refusal(reason=reason):
            return f"refused: {reason}"
        case Truncated(tokens_used=n):
            return f"truncated at {n} tokens — raise max_tokens and repair"
        case Filtered(category=cat):
            return f"content filtered ({cat}) — do not retry"
        case _:
            assert_never(r)


def part2_discriminated() -> None:
    banner("PART 2 — discriminated unions for model responses")

    responses: list[ModelResponse] = [
        TextAnswer(content="Invoice 88 billed 2 extra units",
                   citations=(ChunkId("c3"), ChunkId("c4"))),
        ToolRequest(call_id="call_1", tool_name="get_goods_receipt",
                    arguments='{"gr_id":"GR-0055"}'),
        Refusal(reason="request outside supported scope"),
        Truncated(partial='{"disposition":"adjust_qu', tokens_used=64),
        Filtered(category="self_harm"),
    ]
    for r in responses:
        show(r.kind, handle_response(r))

    print("""
    Every branch above knows its exact type. `case Refusal(reason=...)` cannot
    accidentally reference `.citations`; `case TextAnswer(...)` cannot
    reference `.tool_name`. The checker enforces it.

    AND THE HANDLING DIFFERS PER VARIANT, which is the practical point:
      Truncated  -> REPAIR (raise max_tokens)
      Filtered   -> FAIL FAST (same prompt, same rejection)
      Refusal    -> its own outcome, tracked separately, not an error
      ToolRequest-> execute and loop
    Collapsing these into `dict[str, Any]` is how they end up all going down
    the same retry path.""")


# ---------------------------------------------------------------------------
# PART 3 — TypeGuard and TypeIs
# ---------------------------------------------------------------------------

def is_chunk_list(value: object) -> TypeGuard[list[Chunk]]:
    """A user-defined narrowing function.

    Returning `bool` tells the checker nothing. Returning
    `TypeGuard[list[Chunk]]` says "if this returns True, the argument is a
    list[Chunk]" — and mypy narrows accordingly at the call site.

    THE OBLIGATION: the checker TRUSTS you. A TypeGuard whose body does not
    actually verify the claim is a silent lie, exactly like `cast`. Write the
    check honestly, including the element check below that people skip.
    """
    return (
        isinstance(value, list)
        and all(isinstance(c, Chunk) for c in value)   # element check matters
    )


def is_valid_disposition(value: str) -> TypeGuard[Disposition]:
    """Narrowing a `str` to a `Literal` union — the pattern for validating
    model output against a closed set."""
    return value in {
        "approve", "request_credit", "adjust_quantity",
        "hold_pending_receipt", "escalate_to_buyer",
    }


def part3_typeguard() -> None:
    banner("PART 3 — TypeGuard: making your own narrowing functions")

    payload: object = list(CORPUS[:2])
    if is_chunk_list(payload):
        show("narrowed to list[Chunk]", [c.chunk_id for c in payload])
    else:
        show("not a chunk list", type(payload).__name__)

    section("narrowing model output to a Literal")
    for raw in ("adjust_quantity", "partial_approve"):
        if is_valid_disposition(raw):
            # `raw` is `Disposition` here, not `str`. It can be passed to
            # anything expecting the Literal union.
            show(f"{raw!r}", f"valid -> {raw}")
        else:
            show(f"{raw!r}", "rejected — hallucinated enum value")

    print("""
    TypeGuard vs TypeIs (PEP 742, Python 3.13 / typing_extensions):

      TypeGuard[T]  narrows only in the POSITIVE branch. In the `else`, the
                    variable keeps its original type. Use when the function
                    checks something narrower than a pure type test — e.g.
                    "is this a non-empty list of Chunks".

      TypeIs[T]     narrows in BOTH branches: True -> T, False -> the original
                    type minus T. Closer to isinstance, and usually what you
                    want. Requires T to be a subtype of the parameter type.

    Concretely, with `x: Chunk | Document`:
        TypeGuard[Chunk] -> else-branch: still `Chunk | Document`
        TypeIs[Chunk]    -> else-branch: narrowed to `Document`

    On 3.12 (this environment) `TypeIs` lives in `typing_extensions`. Prefer
    it for anything that is genuinely a type test; keep TypeGuard for
    value-dependent checks where the negative case really is un-narrowable.""")


# ---------------------------------------------------------------------------
# PART 4 — exhaustiveness over Literals, not just classes
# ---------------------------------------------------------------------------

def retry_policy(reason: FinishReason) -> Literal["retry", "repair", "fail", "ok"]:
    """Exhaustive dispatch over a Literal union.

    Same `assert_never` trick, applied to strings. If the API adds a new
    finish_reason and you extend the Literal, this function fails to compile
    until you decide what the new value means — which is exactly the review
    conversation you want to be forced into.
    """
    match reason:
        case "stop":
            return "ok"
        case "length":
            return "repair"          # raise max_tokens, not a retry
        case "content_filter":
            return "fail"            # same prompt, same result
        case "tool_calls":
            return "ok"              # not a failure at all
        case _:
            assert_never(reason)


def part4_exhaustive_literals() -> None:
    banner("PART 4 — exhaustiveness over Literal unions")

    for r in ("stop", "length", "content_filter", "tool_calls"):
        show(f"finish_reason={r!r}", retry_policy(r))

    print("""
    VERIFIED BEHAVIOUR: adding "unknown" to the FinishReason Literal without
    adding a case here produces, at the `assert_never` line:

        error: Argument 1 to "assert_never" has incompatible type
               "Literal['unknown']"; expected "Never"  [arg-type]

    This is the single highest-value typing pattern for LLM code, because the
    set of things a model API can return grows over time and the default
    failure mode is silence — a new finish_reason falling into an `else` that
    treats it as success.

    A SMALL SURPRISE worth knowing, verified here: the loop above iterates a
    tuple of plain string literals and passes each to `retry_policy`, which
    requires `FinishReason`. That type-checks with NO `type: ignore`.

    mypy infers the tuple's type from the literal constants in it, keeping
    them as `Literal["stop"] | Literal["length"] | ...` rather than widening
    to `str`, because the context demands it. Add a variable that is a plain
    `str` to that tuple and it widens and fails.

    The practical lesson: literal inference is context-sensitive. If you see a
    surprising narrowing success or failure around Literals, the answer is
    usually that mypy widened (or did not widen) at an assignment. Annotate
    the collection explicitly — `tuple[FinishReason, ...]` — when you want the
    behaviour to be obvious to a reader rather than inferred.""")


# ---------------------------------------------------------------------------
# PART 5 — overload
# ---------------------------------------------------------------------------

@overload
def get_chunk(chunk_id: ChunkId) -> Chunk | None: ...
@overload
def get_chunk(chunk_id: ChunkId, *, required: Literal[True]) -> Chunk: ...
@overload
def get_chunk(chunk_id: ChunkId, *, required: Literal[False]) -> Chunk | None: ...


def get_chunk(chunk_id: ChunkId, *, required: bool = False) -> Chunk | None:
    """The implementation. Note it is NOT decorated with @overload, and its
    signature must be compatible with all the overloads above.

    WHAT THIS BUYS: the return type depends on an ARGUMENT VALUE.

        get_chunk(cid)                  -> Chunk | None   (must check)
        get_chunk(cid, required=True)   -> Chunk          (no check needed)

    Without overloads the signature is `-> Chunk | None` always, so every
    caller writes a None check they can prove is unnecessary — and eventually
    someone writes `assert x is not None` or `# type: ignore` instead, and the
    discipline erodes.

    WHEN TO USE: a `required` / `strict` / `default` flag that changes
    optionality; str-vs-bytes duality; a function returning a scalar or a list
    depending on whether the input was one item or many.

    WHEN NOT TO: if the overloads differ in more than the return type, you
    probably want two functions with different names. Overloads are read by
    humans too, and four of them describing genuinely different behaviours is
    worse than `get_chunk` and `get_chunk_or_raise`.
    """
    found = next((c for c in CORPUS if c.chunk_id == chunk_id), None)
    if required and found is None:
        raise KeyError(f"chunk {chunk_id!r} not found")
    return found


def part5_overload() -> None:
    banner("PART 5 — overload: return type depending on an argument value")

    maybe = get_chunk(ChunkId("c1"))
    show("get_chunk('c1') -> Chunk | None", maybe.doc_id if maybe else None)

    definite = get_chunk(ChunkId("c1"), required=True)
    # No None check needed — mypy knows this overload returns `Chunk`.
    show("get_chunk('c1', required=True) -> Chunk", definite.doc_id)

    try:
        get_chunk(ChunkId("nope"), required=True)
    except KeyError as e:
        show("get_chunk('nope', required=True)", f"KeyError: {e}")

    print("""
    GOTCHA worth knowing: overload resolution picks the FIRST matching
    signature, so order them most-specific first. With

        @overload def f(x: object) -> str: ...
        @overload def f(x: int) -> int: ...

    the `int` overload is unreachable, and mypy warns about it. Put the
    specific ones on top.""")


# ---------------------------------------------------------------------------
# PART 6 — parsing untrusted JSON into a narrowed union
# ---------------------------------------------------------------------------

def parse_model_response(raw: str, *, max_tokens: int) -> ModelResponse:
    """Turn an untrusted payload into a discriminated union — the boundary
    pattern from 01, now producing a type the rest of the system can dispatch
    on exhaustively.

    Note the ORDER of the checks. finish_reason is inspected BEFORE the content
    is parsed, because a truncated response produces a confusing JSON error
    when the real problem is the token limit.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Refusal(reason="response was not JSON")

    if not isinstance(data, dict):
        return Refusal(reason=f"expected an object, got {type(data).__name__}")

    finish = data.get("finish_reason")

    if finish == "length":
        content = data.get("content")
        return Truncated(partial=content if isinstance(content, str) else "",
                         tokens_used=max_tokens)

    if finish == "content_filter":
        cat = data.get("filter_category")
        return Filtered(category=cat if isinstance(cat, str) else "unspecified")

    tool = data.get("tool_call")
    if isinstance(tool, dict):
        name, cid = tool.get("name"), tool.get("id")
        if isinstance(name, str) and isinstance(cid, str):
            args = tool.get("arguments")
            return ToolRequest(call_id=cid, tool_name=name,
                               arguments=args if isinstance(args, str) else "{}")
        return Refusal(reason="malformed tool_call")

    content = data.get("content")
    if not isinstance(content, str) or not content.strip():
        return Refusal(reason="empty or non-string content")

    cites = data.get("citations", [])
    citations = tuple(
        ChunkId(c) for c in cites if isinstance(c, str)
    ) if isinstance(cites, list) else ()

    return TextAnswer(content=content, citations=citations)


def part6_boundary_to_union() -> None:
    banner("PART 6 — untrusted JSON in, discriminated union out")

    payloads = {
        "text answer": '{"finish_reason":"stop","content":"402 vs 400 units",'
                       '"citations":["c3","c4"]}',
        "tool call": '{"finish_reason":"tool_calls","tool_call":'
                     '{"id":"call_1","name":"get_gr","arguments":"{}"}}',
        "truncated": '{"finish_reason":"length","content":"{\\"disp"}',
        "filtered": '{"finish_reason":"content_filter","filter_category":"violence"}',
        "prose refusal": "I'm sorry, I can't help with that.",
        "null content": '{"finish_reason":"stop","content":null}',
    }
    for label, raw in payloads.items():
        r = parse_model_response(raw, max_tokens=64)
        show(label, f"{r.kind:<10} -> {handle_response(r)[:44]}")

    print("""
    After this function, every downstream consumer dispatches on a closed,
    checker-verified union. The untrusted dict does not escape the boundary,
    and the six behaviours above cannot be confused with each other.""")


def main() -> None:
    part1_forms()
    part2_discriminated()
    part3_typeguard()
    part4_exhaustive_literals()
    part5_overload()
    part6_boundary_to_union()

    banner("SUMMARY")
    print("""
  * `is None`, isinstance, Literal `==`/`in`, and assert all narrow. A helper
    returning `bool` does NOT — that is what TypeGuard/TypeIs are for.
  * Narrowing on `self.attr` is not preserved across a method call. Assign to
    a local first.
  * Model a model response as a DISCRIMINATED UNION, not `dict[str, Any]` and
    not one class with five optional fields.
  * `assert_never` in every dispatch makes the union closed: adding a variant
    produces a checker error at every site that must handle it. Highest-value
    pattern in this file.
  * Exhaustiveness works over Literal unions too, not just classes.
  * `overload` for return types that depend on an argument value; two named
    functions when the behaviours genuinely differ.
  * A boundary parser should return the union, so untrusted dicts never
    escape into the rest of the system.
""")


if __name__ == "__main__":
    main()
