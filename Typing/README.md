# Typing Beyond Annotations — Production Python, GenAI Focus

Twelve files, ~4,700 lines. Runnable offline; needs `mypy`, `pydantic`, `pytest`.

**Verified state:** 9/9 scripts execute clean · `mypy` passes on all 10 modules
under `strict = true` plus five extra error codes, with **zero bare
`type: ignore`** · 32/32 tests pass · validation tests **mutation-tested**
(removing the model's constraints breaks 11 of them).

---

## Files

| File | Topic | The thing worth taking away |
|---|---|---|
| `typing_lab.py` | Shared domain types | `NewType` for every id, free at runtime |
| `01_static_vs_runtime.py` | The core distinction | Static is a proof inside, a **lie** outside |
| `02_protocols.py` | Structural typing | Define the Protocol where it's **consumed** |
| `03_generics.py` | TypeVar, ParamSpec, Self | `Callable[..., R]` erases every decorated signature |
| `04_narrowing.py` | Unions and exhaustiveness | `assert_never` — highest-value pattern here |
| `05_containers.py` | TypedDict/dataclass/Pydantic | Measured: the perf argument is **wrong** |
| `06_pydantic_boundaries.py` | Validation at the edge | One model validates *and* generates the schema |
| `07_mypy_in_practice.py` | Running the checker live | 6 of 7 planted bugs caught; the escapee is instructive |
| `08_typed_pipeline.py` | Capstone | Fully checked, zero runtime type cost internally |
| `09_test_typing.py` | Testing types | mypy as a pytest test |
| `broken/` | Deliberately broken code | Input for 07's live checker runs |
| `pyproject.toml` | Strict config worth copying | Five error codes beyond `--strict` |

Run any file: `python 01_static_vs_runtime.py`. Check: `mypy .`. Test:
`pytest 09_test_typing.py -v`.

---

## Suggested path

**Two hours** — 01, 04, 06. The boundary model, discriminated unions with
`assert_never`, and Pydantic doing double duty as validator and schema source.
That's most of the practical value for LLM work.

**Reviewing someone's types** — 07 (config and suppression discipline), then
the checklist below.

**Designing a provider abstraction** — 02, then 08.

---

## The central claim

Annotations do nothing at runtime. So the same `x: int` is **load-bearing
inside your package and worthless on data you receive**. That gives two tools
for two jobs, and using one where the other belongs is the most common typing
mistake in production Python:

| Boundary | Trusted? | Static | Runtime validation |
|---|---|---|---|
| HTTP request body | no | weak | **required** |
| **LLM output** | no | weak | **required** |
| Search index rows | no | weak | required at ingest |
| Config / env | no | weak | required, **at startup** |
| Internal calls | yes | **strong** | none — wasted CPU |

Boundary 2 is the one people miss. Everyone validates the HTTP body; almost
nobody validates the model's JSON with the same rigour, despite it being the
least reliable input in the system.

---

## Measured results

**mypy caught 6 of 7 planted bugs** (`07`, run live against `broken/bugs.py`).
It caught: `Optional` used without a check, a missing return, an impossible
Literal comparison, list invariance, a wrong return type, a bad `await`.

**The one that escaped** was an argument-order swap between two `str`
parameters — invisible to the checker, to review, and to any test with
symmetric fixtures. Applying `NewType` catches it immediately with two errors
naming both positions, at zero runtime cost.

**Default vs `--strict`:** 5 errors vs 7 on the same file. The two extras are
`no-any-return` and the Literal `comparison-overlap` check.

**`assert_never` works exactly as advertised** (`04`) — adding a sixth variant
to a five-member union produced one error, on the `assert_never` line, naming
the new type.

**Protocol conformance failures are well-diagnosed** (`02`) — adding a required
parameter produces the expected and actual signatures side by side.

---

## The claim I had to retract

**"Convert Pydantic to dataclasses because Pydantic is slow" is wrong.**

`05` measures it: Pydantic `model_validate` is ~4.2× a plain dict construction.
Sounds decisive until you do the arithmetic the script now does for you:

```
one RAG request (20 chunks):      0.025ms   vs an LLM call of ~500ms
full ingestion (2,000,000 chunks):
  Pydantic      2.45s
  dataclass     1.02s
  difference    1.43s
```

1.43 seconds across two million objects. Against an ingestion run dominated by
network I/O, that's noise. I'd written "real minutes" — off by orders of
magnitude.

The real reasons to convert at the boundary survive, and they're better ones:
it makes the trust transition **visible in the type signatures**; re-validating
data you already validated implies it might have become invalid; `frozen` and
`slots` are one keyword on a dataclass; and domain types shouldn't carry wire
concerns like serialisation aliases and JSON-schema config.

---

## Two bugs I shipped and then caught

**A misleading repair hint** (`06`). My `ValidationError` → prompt converter
mapped every `*_too_short` error to "must not be empty". For a `min_length=10`
field that's actively wrong — the model returns a 3-character string and fails
again. Now it reports the actual constraint.

**Lint-by-grep** (`09`). `test_no_bare_type_ignores` scanned lines for the
substring `# type: ignore` and reported seven false positives — all prose in
docstrings discussing the rule. Rewritten with `tokenize`, which distinguishes
a real COMMENT token from identical text in a string. A check that fails on its
own documentation is telling you it's the wrong kind of check.

`warn_unused_ignores` also caught **four** suppressions I'd added defensively
for errors mypy never raised. Every one was me guessing at what the checker
would complain about instead of running it.

---

## Decision tables

### Which container

| Situation | Use |
|---|---|
| JSON body you **send**; tool/function-calling schema | `TypedDict` |
| JSON you **receive** from an API or a model | Pydantic → dataclass |
| HTTP request body | Pydantic → dataclass |
| Config from env/files | `pydantic-settings`, at startup |
| Internal domain object | `dataclass(frozen=True, slots=True)` |
| Cache key / set member | `NamedTuple` or frozen dataclass |

The shape to aim for:

```
untrusted input → model_validate → validated model → .to_domain()
                                                          ↓
                              frozen dataclass → the rest of the system
```

### Protocol or ABC

**Protocol** when defining what *you* need from a dependency, when
implementations are third-party or test fakes, when you want zero import
coupling. **ABC** when you're providing shared implementation, or need
instantiation-time enforcement. **Both** when you want to offer convenience
without demanding inheritance.

### Narrowing

Narrows: `is None`, `isinstance`, Literal `==`/`in`, `assert`.
**Does not narrow:** a helper returning `bool` (use `TypeGuard`/`TypeIs`), or
`self.attr` across a method call (assign to a local first).

---

## Review checklist

**Boundaries**
- [ ] Every deserialised payload validated before use
- [ ] **LLM output validated with the same rigour as HTTP input**
- [ ] Config validated at startup, not lazily
- [ ] Validation happens once; internal calls take domain types
- [ ] No `cast()` on untrusted data — only where you just performed the check

**Any**
- [ ] `warn_return_any` on
- [ ] `ignore_missing_imports` scoped per-module, **never global**
- [ ] `object` preferred over `Any` where the type is genuinely unknown
- [ ] `reveal_type` used to confirm checking hasn't silently stopped

**Protocols**
- [ ] Defined where consumed; providers don't import them
- [ ] Minimal and split by role
- [ ] A conformance assignment per implementation
- [ ] `runtime_checkable` only for capability detection, never validation
- [ ] Hand-written fakes, not `MagicMock`

**Generics**
- [ ] Decorators use `ParamSpec`, not `Callable[..., R]`
- [ ] `Self` on anything returning "the same class"
- [ ] Parameters take `Sequence`/`Mapping`; returns are concrete
- [ ] No TypeVar that appears only once

**Unions**
- [ ] Model responses are discriminated unions, not `dict[str, Any]`
- [ ] `assert_never` in every dispatch
- [ ] Truncation, filtering, and parse failure are distinct types

**Suppressions**
- [ ] Every `type: ignore` has an error code and a reason
- [ ] `warn_unused_ignores` on

---

## GenAI-specific notes

- **A `Literal` field emits an `enum` in the JSON Schema**, so the model is
  *told* the valid set instead of guessing. This removes most
  hallucinated-enum failures at the source rather than repairing them after.
- **`extra="forbid"`** — an invented field signals the model misunderstood;
  don't silently drop it. Also required for OpenAI strict structured outputs
  (`additionalProperties: false`).
- **Field constraints encode policy.** `evidence_ids: list[str] =
  Field(min_length=1)` means "no proposal without evidence" — a business rule
  enforced by the type, not a sentence in a prompt the model may ignore.
- **A validator catches hallucinated citations** — ids that don't match any
  known chunk-id format produce an answer that looks cited and isn't.
- **Compress `ValidationError` into a short imperative repair prompt.** That's
  where this tutorial meets the failure-handling one: the error says exactly
  how the output violated the contract, which becomes the *repair*.
- **`DeploymentName` as a `NewType`** stops `complete(model="gpt-4o-mini")`
  from being accepted where a deployment was required.
- **`from __future__ import annotations`** means every name a Pydantic model
  references must exist at **runtime**, not just under `TYPE_CHECKING`.

---

## The limit, demonstrated

`08` demo 4 makes this explicit. The `Retriever` Protocol forces every
implementation to *accept* a `groups` parameter — a new retriever can't
silently drop permission trimming from its signature. But `LeakyRetriever`
conforms perfectly and ignores the argument entirely.

**Types constrain shape, never behaviour.** Every security invariant needs a
runtime assertion too. `09` has a test asserting exactly this, so it's a
tested fact rather than folklore.

---

## Connections to your existing work

**`isc-docint`'s ACL invariant is a typing story as well as a runtime one.**
The Protocol makes `groups` structurally required; the post-retrieval assertion
catches a retriever that accepts and ignores it. `08` shows both layers
together — and the demo proves the second is not redundant.

**The P2 disposition schema should be a Pydantic model, not hand-written
validation.** Six exception classes and seven dispositions as a `Literal` gives
you three things from one declaration: validation of the agent's output, the
`enum` in the JSON Schema you send (so the model is told the valid set), and a
reviewable contract. The hallucinated-enum failures you'd otherwise repair stop
occurring.

**`write_proposal`'s idempotency key can be structural.** `06` part 4 shows a
discriminated union of tool calls where `WriteProposal` requires
`idempotency_key: str = Field(min_length=8)`. A tool call missing it doesn't
validate, so it can't reach the executor — the safety rule is enforced by the
type system on model-generated data, rather than by a code-review comment.

**Retrofit order:** `NewType` for the id types first (mechanical, catches the
argument-swap class immediately), then the Pydantic boundary model for
dispositions, then `assert_never` on the disposition dispatch. All three are
additive. Turn on `mypy` with everything off first and let CI ratchet the error
count down — `07` part 7 has the sequence.
