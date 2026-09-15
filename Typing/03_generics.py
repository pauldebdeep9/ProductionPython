"""
03 — Generics: TypeVar, bounds, ParamSpec, and Self.

WHAT GENERICS BUY YOU
---------------------
Without them you write `-> Any` or `-> object` and the type information dies
at the function boundary:

    def first(items: list[Any]) -> Any: ...
    chunk = first(chunks)          # chunk is Any. checking is now OFF.

With them the relationship between input and output is preserved:

    def first[T](items: Sequence[T]) -> T: ...
    chunk = first(chunks)          # chunk is Chunk. checking continues.

That is the whole idea: generics propagate type information through your
utility code instead of erasing it. In an LLM pipeline, the utilities that
most need this are the wrappers — retry, cache, instrument, bounded_map —
because they sit between typed code and typed code, and an untyped wrapper
severs the connection for everything downstream.

PYTHON 3.12 SYNTAX
------------------
PEP 695 introduced `def f[T](x: T) -> T` and `class Box[T]:`, replacing the
`TypeVar("T")` declaration. The old syntax still works and is what you will
see in most existing code, so both appear below. New code on 3.12+ should
prefer the new form: the scoping is clearer and variance is inferred.

Run:  python 03_generics.py
      mypy --strict 03_generics.py
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import ParamSpec, Protocol, Self, TypeVar

from typing_lab import (
    Chunk,
    ChunkId,
    Completion,
    CORPUS,
    DeploymentName,
    DocId,
    Usage,
    banner,
    section,
    show,
)


# ---------------------------------------------------------------------------
# PART 1 — the basic TypeVar, old and new syntax
# ---------------------------------------------------------------------------

# Old syntax: a module-level TypeVar, reusable across functions.
T_old = TypeVar("T_old")


def first_old(items: Sequence[T_old]) -> T_old | None:
    return items[0] if items else None


# PEP 695: the type parameter is scoped to the function. Cleaner, and it
# cannot accidentally be shared with an unrelated function.
def first[T](items: Sequence[T]) -> T | None:
    """Note `Sequence`, not `list` — see the variance rule in 02 part 7."""
    return items[0] if items else None


def part1_basic() -> None:
    banner("PART 1 — a TypeVar preserves the input/output relationship")

    chunks: list[Chunk] = list(CORPUS)
    c = first(chunks)
    show("first(chunks) at runtime", c.chunk_id if c else None)

    print("""
    The value of this is entirely static. At runtime `first` is three tokens
    of Python. What the TypeVar buys is that mypy knows `c` is `Chunk | None`,
    so:
      * `c.chunk_id` without a None check is an error
      * `c.nonexistent` is an error
      * passing `c` to something expecting a `Document` is an error

    Compare `def first(items: list) -> Any` — every one of those becomes
    silence, and the silence propagates to everything `c` touches.""")

    section("mypy knows the element type")
    strings = first(["a", "b"])
    numbers = first([1, 2, 3])
    show("first(['a','b'])  -> str | None", repr(strings))
    show("first([1,2,3])    -> int | None", repr(numbers))


# ---------------------------------------------------------------------------
# PART 2 — bounds and constraints (they are different)
# ---------------------------------------------------------------------------

class Scored(Protocol):
    """A structural bound: anything with a float `score`."""

    @property
    def score(self) -> float: ...


# BOUND: T can be Chunk, or SpecialChunk, or anything else with `.score`.
# The type is preserved — pass Chunks, get Chunks back.
def top_k[T: Scored](items: Iterable[T], k: int) -> list[T]:
    """`T: Scored` means "T is SOME subtype of Scored".

    This is the one you want almost always. It accepts an open set of types
    and preserves the specific one the caller passed.
    """
    return sorted(items, key=lambda x: -x.score)[:k]


# CONSTRAINT: T must be EXACTLY str or EXACTLY bytes. Not a subclass, not
# anything else. mypy type-checks the body once per constraint.
def join_ids[T: (str, bytes)](parts: Sequence[T], sep: T) -> T:
    """Constrained TypeVars are rare and usually a sign you want an overload
    or a Protocol instead. The legitimate use is str/bytes duality, where the
    two implementations genuinely cannot be unified.

    GOTCHA: with a constraint, `T` binds to one of the listed types exactly.
    A `str` subclass binds T to `str`, so the return type is `str`, not the
    subclass. That surprises people who expected bound-like behaviour.
    """
    # NOTE: this needs no `type: ignore`. mypy checks the body ONCE PER
    # CONSTRAINT — once with T=str, once with T=bytes — and `sep.join(parts)`
    # is valid in both worlds. That per-constraint re-checking is precisely
    # what distinguishes a constraint from a bound, and it is why the body
    # can use operations that only exist on some of the constrained types.
    return sep.join(parts)


def part2_bounds() -> None:
    banner("PART 2 — bound (open) vs constraint (closed)")

    ranked = top_k(CORPUS, 3)
    show("top_k returns list[Chunk], not list[Scored]",
         [c.chunk_id for c in ranked])
    print("""
    That is the point of the BOUND. With `items: Iterable[Scored]` the return
    would be `list[Scored]`, and `ranked[0].doc_id` would be an error because
    `Scored` has no `doc_id`. The TypeVar keeps the caller's concrete type.""")

    section("constraint")
    show("join_ids(['a','b'], '-')", join_ids(["a", "b"], "-"))
    show("join_ids([b'a', b'b'], b'-')", join_ids([b"a", b"b"], b"-"))
    print("""
    You cannot mix: `join_ids(['a'], b'-')` is a mypy error, because T cannot
    simultaneously be str and bytes. That IS the value of a constraint — but
    note how narrow the useful case is.""")


# ---------------------------------------------------------------------------
# PART 3 — generic classes: a Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ok[T]:
    """PEP 695 generic class syntax."""

    value: T


@dataclass(frozen=True)
class Err[E]:
    error: E


type Result[T, E] = Ok[T] | Err[E]
"""PEP 695 `type` statement — a real type alias, lazily evaluated.

Better than `Result = Ok[T] | Err[E]` because it is unambiguously a type
alias to the checker (rather than a variable that happens to hold a type),
and because the right-hand side is not evaluated until needed, so forward
references work.
"""


@dataclass
class Cache[K: Hashable, V]:
    """Two type parameters, one with a bound.

    `K: Hashable` is doing real work: without it, mypy would let you use a
    `list` as a key, which fails at runtime with `unhashable type`.
    """

    _data: dict[K, tuple[float, V]] = field(default_factory=dict)
    ttl: float = 60.0
    hits: int = 0
    misses: int = 0

    def get(self, key: K) -> V | None:
        entry = self._data.get(key)
        if entry is None or time.monotonic() - entry[0] > self.ttl:
            self.misses += 1
            return None
        self.hits += 1
        return entry[1]

    def put(self, key: K, value: V) -> None:
        self._data[key] = (time.monotonic(), value)

    def with_ttl(self, ttl: float) -> Self:
        """`Self` is the return type that means "whatever the actual class is".

        With `-> Cache[K, V]`, a subclass's method would be typed as returning
        the BASE class, silently losing the subclass type for every chained
        call. `Self` fixes that — essential for builders, fluent APIs, and
        `from_*` classmethods.
        """
        self.ttl = ttl
        return self


def part3_generic_classes() -> None:
    banner("PART 3 — generic classes, Result, and Self")

    section("a typed cache")
    # The key is a (query, groups) tuple — identity is part of the key,
    # which is the permission-correctness point from the failure tutorial.
    cache: Cache[tuple[str, frozenset[str]], list[Chunk]] = Cache(ttl=30.0)
    key = ("units billed", frozenset({"isc-all"}))
    show("cache.get before put", cache.get(key))
    cache.put(key, list(CORPUS[:2]))
    got = cache.get(key)
    show("cache.get after put", [c.chunk_id for c in got] if got else None)
    show("hits / misses", f"{cache.hits} / {cache.misses}")
    print("""
    mypy knows `got` is `list[Chunk] | None`. It will reject `got[0].nope`,
    reject using it without a None check, and reject `cache.put(key, "wrong")`.
    None of that costs anything at runtime.""")

    section("Result, for errors that are values")

    def parse_score(raw: str) -> Result[float, str]:
        try:
            v = float(raw)
        except ValueError:
            return Err(f"{raw!r} is not a number")
        if not 0.0 <= v <= 1.0:
            return Err(f"{v} out of range [0,1]")
        return Ok(v)

    for raw in ("0.85", "high", "1.5"):
        r = parse_score(raw)
        # Narrowing on the union: mypy knows which branch has `.value`
        # and which has `.error`. Accessing the wrong one is an error.
        match r:
            case Ok(value):
                show(f"parse_score({raw!r})", f"Ok({value})")
            case Err(error):
                show(f"parse_score({raw!r})", f"Err({error})")

    print("""
    WHEN A Result TYPE IS WORTH IT: at a boundary where failure is EXPECTED
    and the caller must handle it — parsing model output, validating input.
    The type forces the caller to deal with the error case; an exception can
    be silently propagated past three layers that should have handled it.

    WHEN IT IS NOT: everywhere else. Python has exceptions and they are
    idiomatic. A codebase that returns Result from every function reads like
    Rust written by someone homesick, and fights every library you use.
    Reserve it for boundaries.""")

    section("Self preserves the subclass")

    class MetricsCache(Cache[str, int]):
        def record(self) -> None: ...

    mc = MetricsCache().with_ttl(10.0)
    show("MetricsCache().with_ttl(10) is a", type(mc).__name__)
    mc.record()   # works because `with_ttl` returned Self, not Cache
    print("    With `-> Cache[K,V]` instead of `-> Self`, the `.record()`")
    print("    above would be a mypy error: Cache has no attribute 'record'.")


# ---------------------------------------------------------------------------
# PART 4 — ParamSpec: decorators that keep their signature
# ---------------------------------------------------------------------------

P = ParamSpec("P")
R = TypeVar("R")


def naive_retry(fn: Callable[..., R]) -> Callable[..., R]:
    """THE COMMON BUG. `Callable[..., R]` means "any arguments at all".

    Decorate a function with this and you have thrown away its entire
    signature. `search(query, groups)` becomes callable as `search()`,
    `search(1, 2, 3, 4)`, or `search(nonsense=True)` — all accepted by mypy.

    A decorator is exactly the wrong place to lose type information, because
    it sits between callers and a function whose signature they depend on.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> R:
        return fn(*args, **kwargs)

    return wrapper


def typed_retry(
    attempts: int = 3, base_delay: float = 0.01
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """A retry decorator that PRESERVES the wrapped signature.

    `P` captures the full parameter list — positional, keyword, defaults,
    keyword-only, everything. `R` captures the return type. The decorated
    function is indistinguishable from the original to a caller and to mypy.

    Read the return type from the inside out:
        Callable[P, Awaitable[R]]                  the function we decorate
        Callable[[that], that]                     the decorator itself
        Callable[..., that]                        the factory taking `attempts`

    That three-level nesting is why decorator factories look intimidating.
    Write it once, put it in a shared module, and never think about it again.
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last: Exception | None = None
            for n in range(attempts):
                try:
                    return await fn(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    last = e
                    if n < attempts - 1:
                        await asyncio.sleep(base_delay * (2 ** n))
            raise last if last else RuntimeError("unreachable")

        return wrapper

    return decorator


class FlakyLLM:
    def __init__(self) -> None:
        self.calls = 0

    @typed_retry(attempts=4)
    async def complete(
        self, prompt: str, *, max_tokens: int = 256, deployment: str = "d"
    ) -> Completion:
        """Decorated, and mypy still sees the exact signature below."""
        self.calls += 1
        if self.calls < 3:
            raise ConnectionError("connection reset")
        return Completion(
            text=f"ok after {self.calls} attempts", finish_reason="stop",
            usage=Usage(1, 1), deployment=DeploymentName(deployment),
        )


async def part4_paramspec() -> None:
    banner("PART 4 — ParamSpec: decorators that do not erase signatures")

    llm = FlakyLLM()
    c = await llm.complete("classify this exception", max_tokens=64)
    show("result", c.text)
    show("underlying calls made", llm.calls)

    print("""
    The decorated method still type-checks properly. mypy rejects all of:

        await llm.complete()                     missing 'prompt'
        await llm.complete("q", 64)              max_tokens is keyword-only
        await llm.complete("q", max_tokens="64") wrong type
        await llm.complete("q", maxtokens=64)    typo in a keyword name

    With `Callable[..., R]` every one of those is accepted, and you find out
    at runtime — or worse, do not, because `**kwargs` swallowed it.

    THE REVIEW RULE: any decorator in a shared module that is typed
    `Callable[..., Any]` or `Callable[..., R]` is erasing the signature of
    every function it decorates. In a codebase where `@retry`, `@traced`, and
    `@cached` are on every service method, that is most of your API surface
    silently unchecked.""")


# ---------------------------------------------------------------------------
# PART 5 — a generic async utility, fully typed
# ---------------------------------------------------------------------------

async def bounded_map[T, U](
    fn: Callable[[T], Awaitable[U]],
    items: Sequence[T],
    *,
    limit: int = 8,
) -> list[U]:
    """The concurrency helper from the async tutorial, now generically typed.

    Two type parameters, and the relationship between them is what matters:
    given a `Callable[[Chunk], Awaitable[float]]` and a `list[Chunk]`, mypy
    infers `list[float]`. Pass a mismatched function and it says so.

    Without generics this would be `fn: Callable[[Any], Awaitable[Any]]` and
    the result `list[Any]` — and every downstream use of the result is
    unchecked. Utilities are where type erasure does the most damage, because
    the erasure spreads to all their callers.
    """
    sem = asyncio.Semaphore(limit)
    results: list[U | None] = [None] * len(items)

    async def run(idx: int, item: T) -> None:
        async with sem:
            results[idx] = await fn(item)

    async with asyncio.TaskGroup() as tg:
        for i, item in enumerate(items):
            # KNOWN FALSE POSITIVE: with `enable_error_code = ["unused-awaitable"]`
            # mypy flags this as "Value of type Task[None] must be used. Are you
            # missing an await?" — but a TaskGroup OWNS its children and awaits
            # them at block exit, so discarding the handle is correct here.
            #
            # Worth knowing before you enable that error code: it is a genuinely
            # valuable check (it catches the forgotten `await` that leaves a
            # coroutine object where you expected a value) with exactly this one
            # recurring false positive in structured-concurrency code.
            _task = tg.create_task(run(i, item))
            del _task

    # The cast is EARNED: the TaskGroup guarantees every slot was assigned, a
    # fact mypy cannot see. Note the runtime assertion backing it up — the
    # pattern from 01: verify, then assert to the checker.
    assert all(r is not None for r in results)
    return [r for r in results if r is not None]


async def part5_generic_utility() -> None:
    banner("PART 5 — generic utilities keep the chain of types intact")

    async def score_chunk(c: Chunk) -> float:
        await asyncio.sleep(0.001)
        return len(c.text) / 100.0

    scores = await bounded_map(score_chunk, CORPUS, limit=3)
    show("bounded_map(score_chunk, CORPUS) ->", [round(s, 2) for s in scores])
    show("mypy infers", "list[float]")

    async def get_id(c: Chunk) -> ChunkId:
        return c.chunk_id

    ids = await bounded_map(get_id, CORPUS, limit=3)
    show("bounded_map(get_id, CORPUS) ->", ids)
    show("mypy infers", "list[ChunkId]")

    print("""
    Same function, two different inferred return types, both checked. If
    `score_chunk` were changed to return `str`, every caller doing arithmetic
    on the result would light up immediately — the error surfaces at the
    definition of the mismatch, not three layers downstream.""")


# ---------------------------------------------------------------------------
# PART 6 — when NOT to reach for generics
# ---------------------------------------------------------------------------

def part6_restraint() -> None:
    banner("PART 6 — generics you should not write")

    print("""
    Generics are for UTILITIES — code that works uniformly over types it does
    not care about. They are not for domain code.

    DO NOT write:

        class Repository[T]:                    # "generic repository"
            def get(self, id: str) -> T: ...
            def save(self, entity: T) -> None: ...

    A ChunkRepository and an InvoiceRepository do not have the same queries,
    the same permissions model, or the same lifecycle. The generic base ends
    up either empty (so it bought nothing) or full of `if isinstance` branches
    (so it made things worse). Write the two concrete classes.

    DO NOT write a TypeVar used exactly once in a signature:

        def process[T](item: T) -> None: ...    # T appears once — pointless

    If the type parameter does not connect two positions (parameter to return,
    or parameter to parameter), it is not doing anything. Use `object`.

    DO NOT reach for generics to avoid a union. Three known types is a union
    and often a discriminated one (see 04); it is not a type parameter.

    THE TEST: can you name at least two genuinely different types that will
    instantiate it, and does the type parameter connect two positions in the
    signature? If not, you want a concrete type, a Protocol, or a union.

    WHERE THEY DO EARN THEIR KEEP in an LLM codebase:
      * retry / cache / trace / rate-limit decorators (ParamSpec)
      * bounded_map, gather-with-limit, batching helpers
      * a Result or Validated wrapper at boundaries
      * typed caches and registries
      * pipeline stages: Stage[TIn, TOut] composed into a chain""")


async def main() -> None:
    part1_basic()
    part2_bounds()
    part3_generic_classes()
    await part4_paramspec()
    await part5_generic_utility()
    part6_restraint()

    banner("SUMMARY")
    print("""
  * Generics propagate type information through utilities instead of erasing
    it at the boundary. `Any` in a utility disables checking for every caller.
  * Prefer PEP 695 (`def f[T]`, `class C[T]`, `type X = ...`) on 3.12+.
  * BOUND (`T: Scored`) is the common case and preserves the caller's concrete
    type. CONSTRAINT (`T: (str, bytes)`) is rare and closed.
  * `Self` for anything returning "the same class" — builders, fluent APIs,
    classmethod constructors.
  * ParamSpec is mandatory for decorators. `Callable[..., R]` erases the
    signature of every function it wraps.
  * A Result type is for boundaries where failure is expected, not for
    replacing exceptions everywhere.
  * If a TypeVar appears once, or you cannot name two instantiating types, do
    not write the generic.
""")


if __name__ == "__main__":
    asyncio.run(main())
