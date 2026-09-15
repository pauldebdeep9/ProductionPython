# pytest at Depth — Production Python, GenAI Focus

Nine files, ~3,700 lines of tests against a ~330-line system under test.

**Verified state:** 133 passed, 1 skipped, 1 xfailed · 99% coverage of
`rag/service.py` · every tier selects correctly · the contract suite is
**mutation-tested** (a retriever that stops trimming fails 2 tests by name).

---

## Layout

```
rag/service.py                    the system under test — and its fakes
tests/conftest.py                 fixture hierarchy, hooks, CLI options
tests/unit/test_01_fixtures.py    scope, teardown, factories, request
tests/unit/test_02_parametrize.py ids, stacking, indirect, xfail
tests/unit/test_03_mocking.py     where to draw the mock boundary
tests/unit/test_04_async.py       pytest-asyncio, cancellation, contention
tests/unit/test_07_test_quality.py mutation testing, coverage lies, lints
tests/contract/test_06_contracts.py one test body, many implementations
tests/eval/test_05_eval_as_test.py golden sets, k/n, variance, properties
pytest.ini                        marker tiers
Makefile                          the commands that shape the loop
```

Run: `pytest -q`. Fast loop: `make test-fast`. Evals: `make test-eval`.

---

## Suggested path

**Two hours** — 03, 05, 07. Where to mock, how to test something
nondeterministic, and how to tell whether your tests would notice a bug.

**If you read one file, read 05.** Testing a nondeterministic system is the
part with no obvious answer, and it's where most GenAI test suites are either
flaky or absent.

**Reviewing someone's test PR** — 07's lints, then the checklist below.

---

## The four tiers

| Tier | Model | Runs | A failure means |
|---|---|---|---|
| `unit` | Scripted | Every save | **Blocks the merge** |
| `contract` | Both fake and real | Every push | A fake has drifted |
| `eval` | Noisy/live | Nightly | A **regression** blocks; a score does not |
| `security` | Any | Always | Incident. **Never** skipped or xfailed |

Markers are assigned **by directory** in a `pytest_collection_modifyitems`
hook, not by remembering to decorate. Relying on authors to remember fails
within a month.

---

## Measured results

**The scope-coupling demo works exactly as designed** — `test_scope_hazard_second`
passes in suite order and fails when run alone:

```
$ pytest "tests/unit/test_01_fixtures.py::test_scope_hazard_second"
E   AssertionError: this assertion depends on test execution order
1 failed
```

That's the cost of a module-scoped mutable fixture, made concrete.

**Contract tests catch a drifting fake.** Removing the permission filter from
`SortedListRetriever` fails 2 tests, both naming the implementation:

```
FAILED test_never_returns_invisible_chunks[sorted_list]
FAILED test_empty_groups_returns_nothing[sorted_list]
```

**Eval variance is real and large.** Across 5 runs of a 5-case golden set:
`min=2/5 median=4/5 max=5/5`. A spread of three cases from noise alone — which
is why the absolute floor in that suite is set at 2 and the *regression* test
is what actually gates a merge.

**99% coverage, and the first test in file 07 asserts nothing.** Both true at
once, which is the whole argument about what coverage measures.

---

## Four things I got wrong while writing this

**A strict xfail caught a stale assumption on day one.** I marked
percent-string confidence as a known gap; pytest reported `XPASS(strict)` —
the parser already handled it. A non-strict xfail would have documented a
non-existent limitation forever. Every xfail now has `strict=True`,
`raises=`, a reason, and a ticket.

**My sleep guard fired on intent, not effect.** The autouse fixture raised on
`sleep(10)` — failing three correct cancellation tests that request a long
sleep and cancel immediately. A guardrail that blocks correct code gets
disabled. It now measures *elapsed* time.

**Monkeypatching a dataclass default is a no-op.** `mutate(Config,
"min_confidence", 0.0)` did nothing, because `@dataclass` bakes defaults into
`__init__` at class creation. The mutation "survived" and I nearly wrote it up
as a test gap. It's `dataclasses.replace` instead.

**Two of my lints matched their own source.** Regex-scanning whole files for
`xfail(...)` flagged the lint that contains that pattern as a string. They now
filter to decorator lines. A lint that fails on its own implementation is
operating on the wrong representation.

Separately, my own assertion-free-test lint caught a real one in file 04 —
`test_taskgroup_leaves_nothing_behind`, whose only assertion lived in an
autouse fixture. It now states its own claim.

---

## Decision tables

### Where to mock

```
your code (RagService.answer, .parse, the repair loop)  ← NEVER mock
─────────────────────────────────────────────────────
YOUR PROTOCOLS (Retriever, ModelClient)                 ← substitute a FAKE
─────────────────────────────────────────────────────
the HTTP client                                         ← or intercept here
─────────────────────────────────────────────────────
Azure OpenAI, Azure AI Search (not yours)
```

Prefer **fakes** (real behaviour, assert on outcomes) over **mocks** (canned
values, assert on interactions). A mock-based suite asserts that your code
makes particular calls; a fake-based suite asserts that it produces particular
results. Only the second survives a refactor.

Intercept at HTTP **only** when testing the adapter itself — wire format,
429 handling, truncated bodies. Everything above that uses a Protocol fake.

### Parametrise or not

| Rows differ in… | Do |
|---|---|
| **Data** | `@parametrize` with explicit `ids` |
| **What is asserted** | Separate tests |
| A parametrised body full of `if expected_x is not None:` | Three tests wearing a trench coat |

### Fixture scope

Default to `function`. Widen only when setup is expensive **and** the fixture
is immutable or resets itself. `corpus` is session-scoped safely because
`Chunk` is frozen.

---

## Review checklist

**Fixtures**
- [ ] Narrowest conftest that needs it; conftest is for fixtures and hooks, not helpers
- [ ] `function` scope unless the object is immutable
- [ ] Factory fixtures instead of a dozen near-identical ones
- [ ] `autouse` only for guardrails — it's invisible at the call site

**Assertions**
- [ ] Every test asserts something (lint it)
- [ ] `pytest.raises` has a `match=` when the type alone is too coarse
- [ ] No assertion on full message prose
- [ ] No `if` or bare `try` in a test body

**Async**
- [ ] `asyncio_mode = auto` — in strict mode an unmarked async test is silently **skipped**
- [ ] `@pytest_asyncio.fixture` for async fixtures
- [ ] Assert on peak concurrency, never elapsed time
- [ ] `asyncio.Event`, never `sleep`, for ordering
- [ ] Cancellation tested — including the **double cancel**
- [ ] Contention turned well past production levels

**Markers**
- [ ] `--strict-markers` on; every marker declared in `pytest.ini`
- [ ] Assigned by directory, not by memory
- [ ] Every `skip` has a reason; every `xfail` is strict with `raises=`
- [ ] Nothing marked `security` is ever skipped or xfailed

**Evals**
- [ ] k/n over **named** scenarios, never a percentage
- [ ] Gate on **regression from a committed baseline**, not an absolute score
- [ ] Known failures recorded in the baseline, not hidden
- [ ] Security cases per-case and absolute, across many seeds
- [ ] Variance reported across runs
- [ ] Live tests opt-in behind a flag, nightly, non-blocking

---

## The GenAI-specific problem

An LLM breaks the basic contract of a test: same input, same output. Two ways
teams get this wrong — treating an eval as a unit test (flaky, then skipped,
then deleted), or as not-a-test (quality lives in a notebook and regressions
ship).

The resolution in file 05:

1. **Run the golden set twice** — once against a scripted model (deterministic,
   fast tier, tests the *pipeline*) and once against a noisy one (eval tier,
   tests *quality*). When an eval regresses, the first tier tells you whether
   the pipeline broke or the model did.

2. **Separate wrong-answer from malformed output.** This one surprised me. With
   only malformed failures injected, the suite scored 5/5 every run — the
   repair loop fixes them, doing its job. The failure that matters is
   *syntactically valid JSON containing the wrong answer*: no parser catches
   it, no retry helps, only a graded golden set finds it. An eval harness that
   injects only malformed output reports a reassuring score against a system
   that is wrong.

3. **Gate on regression, not score.** An aggregate threshold hides the change
   that fixes two cases and breaks one.

4. **Security cases don't get graded on a curve.** "4/5, and the one that
   failed was the HR isolation case" is an incident, not a score.

---

## Connections to your existing work

**Your P2 k/12 discipline is exactly the reporting model in file 05** — named
scenarios, no percentages, because n=12 can't support a rate estimate. What
the variance test adds is a number for how much of that k/12 is noise. At five
cases the spread was three; your twelve will be narrower but not negligible,
and comparing two runs without measuring it is comparing two noise draws.

**Your five stable failures (S02, S06, S08, S11, S12) are a baseline file.**
`test_no_case_regresses_from_the_baseline` is the pattern: record them as
`False`, gate on no *new* failures, and flip one to `True` in the PR that fixes
it. That turns "6.4/12 mean" into something a merge can be gated on.

**The `ScriptedModel` pattern replaces live-model testing for the pipeline.**
Your seven passing scenarios can run deterministically on every commit against
scripted output, leaving the live tier for quality only. That's the split
between "did the pipeline break" and "did the model get worse" — currently
indistinguishable in your setup.

**`test_the_permission_check_is_load_bearing` is worth stealing directly.** It
mutates `visible_to` to always return `True` and asserts the suite notices. For
`isc-docint`, where permission trimming fails silently and upward, "would we
notice if this stopped working" is a more useful question than "does it work".

---

That's six topics: async concurrency, failure handling, typing, configuration
and secrets, observability, and pytest.
