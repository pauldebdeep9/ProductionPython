"""
02 — Protocols: structural typing for swappable providers.

THE PROBLEM PROTOCOLS SOLVE
---------------------------
You want your RAG pipeline to work with Azure OpenAI, Bedrock, a local model,
and a deterministic fake in tests. The nominal (inheritance) approach:

    class LLMProvider(ABC): ...
    class AzureProvider(LLMProvider): ...
    class FakeProvider(LLMProvider): ...

That works, but it requires every implementation to IMPORT AND INHERIT from
your base class. Which means:
  * you cannot adapt a third-party client you do not own
  * your test fake must import production code
  * the vendor's own client can never satisfy your interface directly
  * `pipeline.py` and `azure_provider.py` both depend on `interfaces.py`,
    creating a dependency you have to maintain

Protocols invert this. A Protocol says "anything with this shape qualifies",
checked structurally by mypy. The implementation does not import or inherit
anything. The dependency arrow points one way: only the CONSUMER declares the
Protocol, and providers just happen to fit.

That is the Dependency Inversion Principle with no runtime cost and no
import coupling.

Run:  python 02_protocols.py
      mypy --strict 02_protocols.py
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

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
# PART 1 — the Protocol
# ---------------------------------------------------------------------------

class CompletionClient(Protocol):
    """What the pipeline NEEDS from a model provider. Nothing more.

    DESIGN RULES FOR PROTOCOLS, in order of importance:

    1. Define it where it is CONSUMED, not where it is implemented. This file
       is `pipeline`-side. The Azure adapter does not import it.

    2. Keep it MINIMAL. Every method you add is a method every fake must
       implement. A Protocol with 12 methods produces test fakes nobody wants
       to write, and people start reaching for MagicMock — which is how you
       lose type safety in tests entirely.

    3. Name it for the ROLE, not the implementation. `CompletionClient`, not
       `AzureOpenAIInterface`.

    4. Prefer several small Protocols over one big one. A reranker that only
       needs `complete` should not depend on a Protocol that also declares
       `embed` and `stream`.
    """

    async def complete(
        self, prompt: str, *, max_tokens: int = 256
    ) -> Completion:
        """Note the body is `...`. A Protocol method has no implementation.

        The parameter names and kinds are PART OF THE CONTRACT: because
        `max_tokens` is keyword-only here, an implementation declaring it
        positionally is a mypy error. That is usually what you want — it means
        callers can rely on `complete(p, max_tokens=10)` working everywhere.
        """
        ...


class Embedder(Protocol):
    """Separate Protocol, because retrieval needs this and generation does not.

    Splitting them means the generation-only fake does not have to implement
    `embed`, and a change to the embedding signature cannot break a component
    that never embeds.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class Retriever(Protocol):
    """The search side. Note `groups` is required, not optional.

    A Protocol is where you can ENFORCE a security invariant structurally: any
    retriever plugged into this pipeline must accept the caller's group set.
    An implementation that ignores permission trimming can still be written,
    but it cannot silently omit the parameter — someone has to actively drop
    it on the floor, which is visible in review.
    """

    async def search(
        self, query: str, groups: frozenset[str], *, k: int = 5
    ) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# PART 2 — implementations that never import the Protocol
# ---------------------------------------------------------------------------

class AzureOpenAIAdapter:
    """Wraps the vendor SDK. Does NOT inherit from CompletionClient.

    In a real codebase this lives in `adapters/azure.py` and has no import
    relationship with the module defining the Protocol. mypy verifies the
    shape matches wherever it is passed to something expecting a
    CompletionClient.
    """

    def __init__(self, deployment: DeploymentName) -> None:
        self.deployment = deployment
        self.calls = 0

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> Completion:
        self.calls += 1
        await asyncio.sleep(0.001)
        return Completion(
            text=f"[azure:{self.deployment}] answer to {prompt[:24]}",
            finish_reason="stop",
            usage=Usage(len(prompt) // 4, 12),
            deployment=self.deployment,
        )


class DeterministicFake:
    """A test double. Also does not import or inherit anything.

    THIS IS THE PAYOFF. Compare with the alternative:

        mock = MagicMock()
        mock.complete.return_value = ...

    A MagicMock satisfies every Protocol and no Protocol. It accepts
    `complete(wrong_arg=1)`, returns a Mock for any attribute, and drifts
    silently when the real interface changes. A hand-written fake that mypy
    verifies against the Protocol breaks loudly at exactly the moment the
    interface changes — which is what you wanted.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts_seen: list[str] = []
        self._i = 0

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> Completion:
        self.prompts_seen.append(prompt)
        text = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return Completion(
            text=text, finish_reason="stop",
            usage=Usage(1, 1), deployment=DeploymentName("fake"),
        )


class InMemoryRetriever:
    """A retriever over a fixed corpus, with permission trimming applied
    PRE-ranking — the invariant the Protocol's signature makes hard to skip."""

    def __init__(self, corpus: list[Chunk]) -> None:
        self.corpus = corpus

    async def search(
        self, query: str, groups: frozenset[str], *, k: int = 5
    ) -> list[Chunk]:
        terms = set(query.lower().split())
        # Filter FIRST, then rank. Never the other way round.
        visible = [c for c in self.corpus if c.visible_to(groups)]
        scored = [
            Chunk(c.chunk_id, c.doc_id, c.text, c.allowed_groups, c.sensitivity,
                  float(len(terms & set(c.text.lower().split()))))
            for c in visible
        ]
        return sorted([c for c in scored if c.score > 0],
                      key=lambda c: -c.score)[:k]


# ---------------------------------------------------------------------------
# PART 3 — the consumer depends only on the Protocol
# ---------------------------------------------------------------------------

async def answer_question(
    query: str,
    groups: frozenset[str],
    *,
    retriever: Retriever,
    llm: CompletionClient,
) -> tuple[str, list[ChunkId]]:
    """This function is testable, provider-agnostic, and statically checked.

    It has no idea whether `llm` is Azure, Bedrock, or a fake. Swapping
    providers is a change at the composition root and nowhere else.
    """
    chunks = await retriever.search(query, groups, k=3)
    context = "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
    completion = await llm.complete(f"Q: {query}\n{context}", max_tokens=128)
    return completion.text, [c.chunk_id for c in chunks]


async def part3_consumption() -> None:
    banner("PART 3 — one consumer, three providers, zero inheritance")

    retriever = InMemoryRetriever(CORPUS)
    isc = frozenset({"isc-all"})

    section("production adapter")
    azure = AzureOpenAIAdapter(DeploymentName("gpt-4o-mini-prod"))
    text, cites = await answer_question("units billed", isc,
                                        retriever=retriever, llm=azure)
    show("answer", text[:56])
    show("citations", cites)

    section("test fake — same function, no code change")
    fake = DeterministicFake(["deterministic answer for assertions"])
    text2, cites2 = await answer_question("units billed", isc,
                                          retriever=retriever, llm=fake)
    show("answer", text2)
    show("citations", cites2)
    show("prompts the fake observed", len(fake.prompts_seen))
    print("""
    The fake also RECORDS what it saw, which a Protocol makes natural: a test
    double can carry whatever extra state assertions need, because it is a
    real class you wrote and not a constrained subclass of something.""")


# ---------------------------------------------------------------------------
# PART 4 — runtime_checkable and its sharp edge
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsStream(Protocol):
    """`@runtime_checkable` allows `isinstance(x, SupportsStream)`.

    THE SHARP EDGE, and it is sharp: isinstance against a Protocol checks ONLY
    THAT THE ATTRIBUTE NAMES EXIST. It does not check signatures, parameter
    types, or return types. A class with a `stream` attribute that is an
    integer passes.

    So it answers "does this have a stream method?" and never "is this method
    compatible?". Useful for optional-capability detection; useless as
    validation.

    It is also SLOW — a real attribute lookup per member, per call — so keep
    it out of hot paths.
    """

    def stream(self, prompt: str) -> object: ...


class StreamingProvider:
    def stream(self, prompt: str) -> object:
        return iter(["a", "b"])


class NotReallyStreaming:
    """Has the right ATTRIBUTE NAME and completely the wrong thing behind it."""

    stream = 42


def part4_runtime_checkable() -> None:
    banner("PART 4 — runtime_checkable checks names, not signatures")

    show("isinstance(StreamingProvider(), SupportsStream)",
         isinstance(StreamingProvider(), SupportsStream))
    show("isinstance(NotReallyStreaming(), SupportsStream)",
         isinstance(NotReallyStreaming(), SupportsStream))
    show("isinstance(object(), SupportsStream)",
         isinstance(object(), SupportsStream))

    print("""
    The second line is the warning. `stream = 42` passes the isinstance check
    and explodes the moment you call it.

    USE IT FOR: optional capability detection at a plugin boundary —
    "does this provider support streaming? if so, use the streaming path".
    A wrong answer there degrades gracefully.

    DO NOT USE IT FOR: validating that an object satisfies a contract. That is
    what mypy does, statically, correctly, for free.

    Also note: `@runtime_checkable` on a Protocol with non-method members
    raises TypeError on isinstance in older versions and is generally
    discouraged. Keep runtime-checkable Protocols to methods only.""")


# ---------------------------------------------------------------------------
# PART 5 — Protocol vs ABC
# ---------------------------------------------------------------------------

class AbstractProvider(ABC):
    """The nominal alternative, for comparison."""

    @abstractmethod
    async def complete(self, prompt: str, *, max_tokens: int = 256) -> Completion:
        ...

    # The genuine advantage of an ABC: SHARED IMPLEMENTATION. Every subclass
    # gets this for free, and you can change it in one place.
    async def complete_with_retry(self, prompt: str, attempts: int = 3) -> Completion:
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return await self.complete(prompt)
            except Exception as e:  # noqa: BLE001 — demo
                last = e
        raise last if last else RuntimeError("unreachable")


class ConcreteProvider(AbstractProvider):
    """Must import and inherit. That is the cost."""

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> Completion:
        return Completion("abc answer", "stop", Usage(1, 1),
                          DeploymentName("abc"))


async def part5_protocol_vs_abc() -> None:
    banner("PART 5 — Protocol or ABC?")

    p = ConcreteProvider()
    c = await p.complete_with_retry("q")
    show("ABC gives shared behaviour for free", c.text)

    print("""
    USE A PROTOCOL WHEN:
      * you are defining what YOU need from a dependency (the common case)
      * implementations are third-party, or in another package, or test fakes
      * you want zero import coupling between consumer and provider
      * you are adapting something you do not own

    USE AN ABC WHEN:
      * you are providing SHARED IMPLEMENTATION, not just a contract
      * you want `isinstance` to be meaningful and cheap
      * you need to enforce at INSTANTIATION time that subclasses are complete
        (an ABC raises TypeError; a Protocol mismatch is a mypy error only)
      * the hierarchy is genuinely an "is-a", not a "can-do"

    USE BOTH — and this is the pattern worth knowing: declare the Protocol for
    consumers, and ALSO ship an ABC or a base class implementing the common
    parts, for providers who want it. Providers may inherit for convenience or
    just match the shape. Consumers only ever see the Protocol.

    A PRACTICAL NOTE ON ENFORCEMENT: because Protocol conformance is a
    static-only check, a provider that drifts is caught by mypy but not at
    import time. If you want a runtime guarantee, add a one-line conformance
    assertion in your tests — see 09.""")


# ---------------------------------------------------------------------------
# PART 6 — Protocol conformance, asserted statically
# ---------------------------------------------------------------------------

def _assert_conformance() -> None:
    """A static conformance check, costing nothing at runtime.

    Assigning an instance to a variable annotated with the Protocol forces
    mypy to verify the shape at THIS line, in THIS module — rather than
    wherever the object happens to be passed later. The error message names
    the class and the missing/incompatible member.

    Put one of these next to each adapter, or in a test (see 09). It converts
    "you will find out when someone passes it somewhere" into "you find out in
    the file you just edited".
    """
    _a: CompletionClient = AzureOpenAIAdapter(DeploymentName("d"))
    _f: CompletionClient = DeterministicFake([])
    _r: Retriever = InMemoryRetriever([])
    del _a, _f, _r


def part6_conformance() -> None:
    banner("PART 6 — asserting conformance where you can see it")

    _assert_conformance()
    print("""
    `_assert_conformance()` above type-checks under mypy --strict, which
    proves all three classes satisfy their Protocols.

    VERIFIED: adding a required `temperature: float` to
    `CompletionClient.complete` and re-running mypy produces exactly this —

      error: Argument "llm" to "answer_question" has incompatible type
             "AzureOpenAIAdapter"; expected "CompletionClient"  [arg-type]
      note: Following member(s) of "AzureOpenAIAdapter" have conflicts:
      note:     Expected:
      note:         def complete(self, prompt: str, *, max_tokens: int = ...,
                                 temperature: float) -> Coroutine[...]
      note:     Got:
      note:         def complete(self, prompt: str, *, max_tokens: int = ...)
                                 -> Coroutine[...]

    Note what the notes give you: the expected signature and the actual one,
    side by side. That is a materially better failure than "AttributeError:
    'AzureOpenAIAdapter' object has no attribute..." six months later.

    Note also WHERE the errors land. Without the `_assert_conformance` helper,
    mypy reports them at every CALL SITE that passes the adapter — which may
    be in a different package from the adapter you just broke. With it, you
    additionally get an error in the adapter's own module, which is where the
    person who needs to fix it is looking.""")


# ---------------------------------------------------------------------------
# PART 7 — variance, briefly but concretely
# ---------------------------------------------------------------------------

class ChunkSink(Protocol):
    """A Protocol with a method that CONSUMES chunks."""

    def accept(self, chunk: Chunk) -> None: ...


class ChunkSource(Protocol):
    """A Protocol with a method that PRODUCES chunks."""

    def produce(self) -> Chunk: ...


def part7_variance() -> None:
    banner("PART 7 — the variance rule you actually need")

    print("""
    The rule in one line: BE LIBERAL IN WHAT YOU ACCEPT, STRICT IN WHAT YOU
    RETURN. Concretely, for function parameters:

        def rank(chunks: Sequence[Chunk]) -> None      GOOD
        def rank(chunks: list[Chunk]) -> None          worse

    `list` is INVARIANT: a `list[SpecialChunk]` is NOT a `list[Chunk]`, because
    someone holding the `list[Chunk]` reference could append a plain Chunk and
    corrupt the caller's list. mypy is right to reject it, and it surprises
    everyone the first time.

    `Sequence` is COVARIANT (read-only), so `Sequence[SpecialChunk]` IS a
    `Sequence[Chunk]`. Since `rank` only reads, `Sequence` is both more
    permissive and more honest about what the function does.

    THE PRACTICAL TABLE for parameters:
        reading a list      -> Sequence[T]        (or Iterable[T] if one pass)
        reading a dict      -> Mapping[K, V]
        reading a set       -> AbstractSet[T]  /  Collection[T]
        MUTATING it         -> list[T] / dict[K, V]   (invariance is correct
                                                       here; you need it)

    For RETURN types, do the opposite — return the concrete type. Returning
    `Sequence[Chunk]` when you have a `list` forces every caller to copy it if
    they want to sort. Return `list[Chunk]`.

    ONE MORE, since it bites in async code: `Awaitable[T]` is covariant and
    `Callable[[A], R]` is contravariant in A, covariant in R. That is why a
    callback taking `object` can be passed where one taking `Chunk` is
    expected, but not the reverse.""")

    section("demonstration")
    chunks: list[Chunk] = list(CORPUS)

    def read_only(cs: list[Chunk]) -> int:
        return len(cs)

    show("len via list[Chunk] parameter", read_only(chunks))
    print("    (mypy accepts this; the invariance problem only appears with")
    print("     a list of a SUBCLASS, which is why Sequence is the safer")
    print("     default for any parameter you do not mutate.)")


async def main() -> None:
    banner("PART 1-2 — Protocols defined and implemented (see source)")
    print("    CompletionClient, Embedder, Retriever declared above;")
    print("    AzureOpenAIAdapter, DeterministicFake, InMemoryRetriever")
    print("    implement them WITHOUT importing or inheriting anything.")
    await part3_consumption()
    part4_runtime_checkable()
    await part5_protocol_vs_abc()
    part6_conformance()
    part7_variance()

    banner("SUMMARY")
    print("""
  * Define Protocols where they are CONSUMED. The provider never imports them.
  * Keep them minimal and split by role — every method is a method every fake
    must implement.
  * A hand-written fake that mypy verifies beats a MagicMock, which satisfies
    every interface and none.
  * `runtime_checkable` checks attribute NAMES only. Capability detection,
    never validation.
  * ABC when you have shared implementation or need instantiation-time
    enforcement; Protocol for everything else; both when you want to offer
    convenience without demanding inheritance.
  * A one-line annotated assignment per adapter proves conformance in the file
    you are editing rather than wherever it is later passed.
  * Parameters: Sequence/Mapping/Iterable. Returns: the concrete type.
""")


if __name__ == "__main__":
    asyncio.run(main())
