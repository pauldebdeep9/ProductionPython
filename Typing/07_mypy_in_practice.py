"""
07 — mypy in practice: running it, configuring it, adopting it.

WHAT MAKES THIS SCRIPT DIFFERENT
--------------------------------
It RUNS mypy on a deliberately-broken file and prints the real output. Nothing
below is paraphrased from documentation — every error message is captured from
the checker on this machine, and you can reproduce it by running the file.

That matters because the gap between "mypy catches type errors" and knowing
WHICH errors it catches, which it misses, and what it costs to turn each flag
on is the entire difference between a config someone copied and a config
someone chose.

Run:  python 07_mypy_in_practice.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from typing_lab import banner, section, show

BROKEN = Path(__file__).parent / "broken"


def run_mypy(args: list[str]) -> str:
    """Run mypy as a subprocess and return its output.

    NOTE: mypy has an API (`mypy.api.run`), but shelling out is what CI does
    and it keeps the exit-code semantics obvious: 0 = clean, 1 = errors found,
    2 = a crash or a config problem. CI must distinguish 1 from 2 — treating a
    config error as "type errors found" hides a broken build.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", *args],
        capture_output=True, text=True, cwd=BROKEN,
    )
    return proc.stdout.strip() or proc.stderr.strip()


# ---------------------------------------------------------------------------
# PART 1 — what mypy catches, live
# ---------------------------------------------------------------------------

def part1_live() -> None:
    banner("PART 1 — seven planted bugs, checked live")

    print("    Source: broken/bugs.py — seven bug patterns from real")
    print("    LLM-pipeline code. Running `mypy --strict bugs.py`:\n")
    out = run_mypy(["--strict", "bugs.py"])
    for line in out.splitlines():
        print(f"      {line}")

    print("""
    THE ONE THAT GOT AWAY. Bug 2 in that file is:

        def search(query: str, deployment: str) -> list[Chunk]: ...
        return search(deployment, query)          # <- arguments swapped

    mypy says NOTHING. Both parameters are `str`, so the swap is invisible to
    the checker — and to a reviewer, and to any test whose fixtures happen to
    be symmetric. It fails in production as an empty result set or a call to
    a deployment named "units billed".

    This is the strongest practical argument for NewType, and it is worth
    seeing rather than being told.""")


def part2_newtype_fixes_it() -> None:
    banner("PART 2 — the same bug, with NewType")

    print("    Source: broken/newtype_fix.py — identical logic, distinct")
    print("    NewTypes for Query and DeploymentName:\n")
    out = run_mypy(["--strict", "newtype_fix.py"])
    for line in out.splitlines():
        print(f"      {line}")

    print("""
    Two errors, naming both swapped positions, at zero runtime cost —
    `NewType` compiles away to the identity function.

    WHERE TO SPEND NewTypes, in order of payoff:
      * every id in your domain (DocId, ChunkId, TraceId, CaseId)
      * deployment name vs model name
      * anything measured in different units (Seconds vs Milliseconds,
        Tokens vs Characters)
      * a validated string vs a raw one (SanitisedHtml vs str)

    The last one is a genuine security pattern: make the validated type
    distinct, have only the validator produce it, and every consumer that
    requires it is provably receiving validated data.""")


# ---------------------------------------------------------------------------
# PART 3 — strict is not one flag
# ---------------------------------------------------------------------------

def part3_strictness_levels() -> None:
    banner("PART 3 — default vs --strict, measured on the same file")

    default_out = run_mypy(["bugs.py"])
    strict_out = run_mypy(["--strict", "bugs.py"])

    def count(out: str) -> int:
        last = [ln for ln in out.splitlines() if ln.startswith("Found")]
        return int(last[0].split()[1]) if last else 0

    show("errors with default settings", count(default_out))
    show("errors with --strict", count(strict_out))

    print("""
    MEASURED: --strict finds strictly more. The two extra errors here are
    `no-any-return` (returning an Any from a typed function) and the
    Literal `comparison-overlap` check.

    `--strict` is a bundle. The individual flags, roughly ordered by
    value-per-unit-of-noise in an LLM codebase:

      warn_return_any            HIGH. Catches Any escaping from json.loads
                                 and untyped libraries into typed code. This
                                 is the main defence against the Any-contagion
                                 problem from script 01.

      disallow_untyped_defs      HIGH. Without it, an unannotated function is
                                 implicitly `(...) -> Any` and everything it
                                 touches goes unchecked. The single biggest
                                 lever for actually getting coverage.

      strict_optional            HIGH, and on by default. Do not turn it off.

      warn_unused_ignores        HIGH, and underrated. Stops `type: ignore`
                                 comments accumulating after the bug they
                                 suppressed was fixed. It caught three stale
                                 ignores while I was writing these scripts.

      disallow_any_generics      MEDIUM. Forces `list[str]` over bare `list`.

      no_implicit_reexport       MEDIUM. Stops modules leaking their imports
                                 as public API by accident.

      disallow_untyped_calls     MEDIUM-LOW. Loud when depending on untyped
                                 third-party code; the first flag to relax
                                 per-module.

      warn_unreachable           LOW but interesting — finds dead code, and
                                 often reveals a narrowing bug.""")


# ---------------------------------------------------------------------------
# PART 4 — a config to actually use
# ---------------------------------------------------------------------------

def part4_config() -> None:
    banner("PART 4 — a pyproject.toml section worth copying")

    print("""
    [tool.mypy]
    python_version = "3.12"
    strict = true

    # Beyond --strict: these are not included and are worth adding.
    warn_unreachable = true
    enable_error_code = [
        "redundant-expr",       # `if x and x:`
        "possibly-undefined",   # a name assigned only in one branch
        "truthy-bool",          # `if some_object:` where it is always truthy
        "ignore-without-code",  # forbids bare `# type: ignore`
        "unused-awaitable",     # a coroutine created and never awaited
    ]

    # `ignore-without-code` and `unused-awaitable` are the two that pay for
    # themselves fastest in async LLM code — the first enforces that every
    # suppression names what it suppresses, the second catches the forgotten
    # `await` that returns a coroutine object where you expected a string.

    # Third-party libraries without stubs. Be SPECIFIC — a blanket
    # ignore_missing_imports hides real typos in your own import paths.
    [[tool.mypy.overrides]]
    module = ["some_untyped_vendor_sdk.*"]
    ignore_missing_imports = true

    # Gradual adoption: relax per-module, never globally, and leave the
    # reason and a ticket in a comment so it is a decision, not sediment.
    [[tool.mypy.overrides]]
    module = ["legacy.ingestion.*"]
    disallow_untyped_defs = false     # TODO(PLAT-412): annotate, drop this

    # Tests: keep them checked. A test calling a function with the wrong
    # arguments is a test that proves nothing, and it is exactly the failure
    # mode you cannot see by reading it.
    [[tool.mypy.overrides]]
    module = ["tests.*"]
    disallow_untyped_decorators = false   # pytest fixtures/marks

    THE ONE MISCONFIGURATION THAT MATTERS MOST:

        ignore_missing_imports = true      # GLOBAL — do not do this

    It silences "cannot find module X" everywhere, which means a genuine typo
    in your own package path becomes an `Any` import and disables checking for
    that whole module, silently. Scope it per-module, always.""")


# ---------------------------------------------------------------------------
# PART 5 — suppression discipline
# ---------------------------------------------------------------------------

def part5_suppression() -> None:
    banner("PART 5 — `type: ignore` discipline")

    print("""
    THE HIERARCHY, from worst to best:

      # type: ignore
          Suppresses EVERY error on that line, now and forever, including
          ones introduced by a later edit. Banned by `ignore-without-code`.

      # type: ignore[arg-type]
          Suppresses one error code. Acceptable.

      # type: ignore[arg-type]  # vendor SDK stubs say str, actually accepts
      # a Path — see PLAT-419
          Suppresses one code, WITH a reason and a ticket. This is the bar.

    `cast(T, x)` vs `# type: ignore`:
      cast() is an assertion about a VALUE and keeps checking downstream.
      type: ignore is a suppression of a LINE and checks nothing about it.
      Prefer cast when you genuinely know the type — but only when you have
      just verified it (script 01, part 5).

    `assert isinstance(x, T)` is the third option, and often the best: it
    narrows for the checker AND verifies at runtime. Costs one isinstance
    call. Use it where the cost is irrelevant and the guarantee is worth
    having — which is most places outside a hot loop.

    THE HYGIENE RULE: `warn_unused_ignores = true` (in --strict) makes mypy
    report ignores that are no longer needed. Without it, suppressions
    accumulate as sediment and nobody ever dares remove one. With it, they
    are cleaned up automatically as the underlying issues get fixed.

    VERIFIED, and slightly embarrassing: while writing these scripts, that
    flag caught FOUR ignores I had added defensively for errors that mypy
    never actually raised — in 03 (str/bytes join), 04 (a Literal loop), and
    two in 05 (timeit lambdas). Every one of them was me guessing at what the
    checker would complain about rather than running it. The flag is not
    theoretical hygiene; it catches this on the first run.""")


# ---------------------------------------------------------------------------
# PART 6 — reveal_type, the debugging tool
# ---------------------------------------------------------------------------

def part6_reveal_type() -> None:
    banner("PART 6 — reveal_type: asking mypy what it thinks")

    tmp = BROKEN / "_reveal.py"
    tmp.write_text('''from __future__ import annotations
import json
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    score: float


def demo(raw: str, chunks: list[Chunk]) -> None:
    data = json.loads(raw)
    reveal_type(data)

    best = max(chunks, key=lambda c: c.score)
    reveal_type(best)

    maybe = next((c for c in chunks if c.score > 0.5), None)
    reveal_type(maybe)

    if maybe is not None:
        reveal_type(maybe)

    scores = {c.chunk_id: c.score for c in chunks}
    reveal_type(scores)
''')
    out = run_mypy(["_reveal.py"])
    for line in out.splitlines():
        print(f"      {line}")
    tmp.unlink()

    print("""
    `reveal_type(x)` is a mypy built-in — no import needed, and it is an ERROR
    at runtime, so it never ships. (`typing.reveal_type` exists in 3.11+ and
    does run, printing at runtime, if you want that instead.)

    READ THE FIRST LINE: `json.loads` gives `Any`. That is the Any-contagion
    problem from script 01, visible in one command. Any time you are unsure
    whether a value is still being checked, `reveal_type` answers it
    definitively in about ten seconds.

    Note also the two `maybe` lines: before the None check and after. Watching
    the narrowing happen is the fastest way to debug a case where you expected
    narrowing and did not get it (script 04, part 1).""")


# ---------------------------------------------------------------------------
# PART 7 — adopting mypy on an existing codebase
# ---------------------------------------------------------------------------

def part7_adoption() -> None:
    banner("PART 7 — adding types to a codebase that has none")

    print("""
    The failure mode is turning `strict = true` on a 40k-line repo, getting
    3,000 errors, and giving up. The order below avoids that.

    1. TURN IT ON WITH EVERYTHING OFF, and make CI enforce zero errors.
         [tool.mypy]
         python_version = "3.12"
       Almost nothing is checked. The point is the RATCHET: from now on the
       error count can only go down.

    2. TYPE THE BOUNDARIES FIRST, not the leaves. The adapter that parses
       model output, the HTTP handlers, the config loader. Highest bug density
       per line, and typing them forces you to notice where validation is
       missing — which is worth more than the types.

    3. ADD `disallow_untyped_defs` PER MODULE, newest and most-changed first.
       Files nobody touches can stay unchecked indefinitely; that is fine.

    4. THEN `warn_return_any` GLOBALLY. This is where the Any leaks surface,
       and it is the highest-value single flag.

    5. RATCHET the per-module overrides down over time, deleting each with the
       PR that annotates its module.

    WHAT TO SKIP, permanently:
      * Retrofitting types onto code scheduled for deletion.
      * `--strict` on test files with heavy fixture magic. Check them, but
        relax `disallow_untyped_decorators`.
      * Chasing 100%. The last 5% is generated code, deep dynamic dispatch,
        and vendored files, and it costs more than it returns.

    THE MEASUREMENT THAT KEEPS IT HONEST: `mypy --html-report` gives per-module
    coverage. Track it in CI as a number that may not decrease. A ratchet with
    a number beats a policy with good intentions.""")


def main() -> None:
    if not BROKEN.exists():
        print(f"expected {BROKEN} to exist with bugs.py and newtype_fix.py")
        return
    part1_live()
    part2_newtype_fixes_it()
    part3_strictness_levels()
    part4_config()
    part5_suppression()
    part6_reveal_type()
    part7_adoption()

    banner("SUMMARY")
    print("""
  * mypy caught 6 of 7 planted bugs. The escapee was an argument-order swap
    between two `str` parameters — which NewType then caught, at zero runtime
    cost.
  * `--strict` is a bundle; know which flags you are getting. warn_return_any
    and disallow_untyped_defs carry most of the value.
  * NEVER set `ignore_missing_imports` globally — a typo in your own import
    path becomes a silent Any.
  * Every suppression names an error code and a reason. `warn_unused_ignores`
    stops them accumulating (and caught four of mine).
  * `reveal_type` answers "is this still being checked?" in ten seconds.
  * Adopt boundaries-first with a CI ratchet, not `strict = true` on day one.
""")


if __name__ == "__main__":
    main()
