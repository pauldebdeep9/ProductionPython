"""
03 — Dependencies: constraints, resolution, and lockfiles.

THE DISTINCTION EVERYTHING ELSE FOLLOWS FROM
--------------------------------------------
    A LIBRARY declares ABSTRACT dependencies — RANGES.
        "I work with httpx 0.27 through 0.x"
        It must COMPOSE with other libraries, so it must leave room.

    An APPLICATION declares CONCRETE dependencies — a LOCKFILE.
        "This deployment uses httpx 0.27.2, sha256:abc..."
        Nothing depends on it, so it can and should pin everything.

Getting this backwards produces the two most common dependency failures:
a library that pins and therefore conflicts with everything, or an
application that does not pin and therefore deploys a different tree every
time.

`isc-rag` in this tutorial is a LIBRARY. The service that deploys it is an
APPLICATION. Both live in the same repo, and they follow different rules.

Run:  python 03_dependencies.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from pkg_lab import (
    ProjectBuilder,
    VenvRunner,
    banner,
    code_block,
    section,
    show,
    verdict,
)


# ---------------------------------------------------------------------------
# PART 1 — the specifier grammar, precisely
# ---------------------------------------------------------------------------

def part1_specifiers() -> None:
    banner("PART 1 — version specifiers, and what each one commits you to")

    rows = [
        ("httpx", "anything", "NEVER in a library you maintain. 2.0 will break you."),
        ("httpx==0.27.2", "exactly this", "Libraries: no. Applications: yes, via a lock."),
        ("httpx>=0.27", "this or newer", "No upper bound — same problem as row 1."),
        ("httpx>=0.27,<1.0", "the useful form", "Lower = what you use. Upper = next major."),
        ("httpx~=0.27.2", "compatible release", ">=0.27.2, <0.28.0 — patch only."),
        ("httpx~=0.27", "compatible release", ">=0.27, <1.0 — the LAST component floats."),
        ("httpx!=0.27.1", "exclude one", "For a specific known-bad release."),
        ("httpx===0.27.2", "arbitrary equality", "String match. Escape hatch; avoid."),
    ]
    print(f"      {'specifier':<22} {'means':<20} note")
    print(f"      {'-' * 22} {'-' * 20} {'-' * 44}")
    for spec, means, note in rows:
        print(f"      {spec:<22} {means:<20} {note}")

    print("""
    THE `~=` TRAP, because it catches everyone once: the number of components
    determines what floats.

        ~=0.27.2   ->  >=0.27.2, ==0.27.*      only the PATCH floats
        ~=0.27     ->  >=0.27,   ==0.*         the MINOR floats

    Those are very different promises, and the difference is one dot.

    THE PRACTICAL ADVICE for a library: `>=X.Y,<NEXT_MAJOR`. It is explicit,
    it does not depend on remembering `~=` semantics, and a reviewer can read
    it correctly at a glance.

    THE ARGUMENT AGAINST UPPER BOUNDS is worth acknowledging honestly: a
    `<3.0` on pydantic means that when pydantic 3.0 ships, every downstream
    application is BLOCKED until you release. That is a real cost, and the
    reason some maintainers refuse upper bounds entirely.

    MY POSITION: cap at the next major anyway, and treat "release a version
    with a raised cap" as a routine maintenance task. The alternative is
    that your users discover the incompatibility in production instead. But
    if you cannot commit to responding quickly, an uncapped dependency plus
    a tested upper bound in your CI matrix is a defensible alternative.""")


# ---------------------------------------------------------------------------
# PART 2 — environment markers
# ---------------------------------------------------------------------------

def part2_markers() -> None:
    banner("PART 2 — environment markers")

    print("""
    A marker makes a dependency CONDITIONAL on the install environment.
    Evaluated by pip at install time, recorded in the wheel metadata.

    dependencies = [
        "httpx>=0.27,<1.0",

        # Backport, only where the stdlib lacks it.
        "tomli>=2.0; python_version < '3.11'",

        # Platform-specific.
        "pywin32>=306; sys_platform == 'win32'",
        "uvloop>=0.19; sys_platform != 'win32' and platform_python_implementation == 'CPython'",

        # Combined with an extra.
        "torch>=2.3; extra == 'local' and sys_platform != 'darwin'",
    ]

    THE VARIABLES YOU WILL ACTUALLY USE:
        python_version              "3.11"   — string compare, so '3.10' < '3.9'
        sys_platform                "linux", "darwin", "win32"
        platform_machine            "x86_64", "arm64"
        platform_python_implementation   "CPython", "PyPy"
        extra                       the requested extra

    A PIECE OF FOLKLORE I HAD TO CORRECT HERE. I wrote that
    `python_version >= '3.9'` is FALSE on 3.10 because "3.10" sorts before
    "3.9" as a string. I then checked it, and it is not true:

        >>> from packaging.markers import Marker
        >>> Marker("python_version >= '3.9'").evaluate(
        ...     {"python_version": "3.10"})
        True
        >>> "3.10" >= "3.9"          # raw Python string comparison
        False

    PEP 508 specifies that when both sides are valid PEP 440 versions, the
    comparison is a VERSION comparison, not a lexical one. `packaging` has
    implemented it that way for years. The trap was real in the setuptools
    era and has been repeated ever since.

    WHERE STRING COMPARISON STILL APPLIES: the `in` and `not in` operators,
    and any comparison where one side is not a valid version — for example
    `platform_release >= '5.10'` on a kernel string like
    `'5.10.0-21-amd64'`. That one is genuinely lexical and genuinely
    surprising.

    THE DURABLE LESSON is not the specific rule; it is that packaging
    folklore has a long half-life. Check a marker with `packaging.markers`
    before relying on it — it takes ten seconds and this tutorial had it
    wrong until it did.

    WHERE MARKERS SAVE YOU IN ML WORK: a package that needs `uvloop` on Linux
    and must not attempt it on Windows, or a CPU-only fallback on macOS where
    a CUDA build does not exist. Expressing that in metadata is much better
    than a README instruction that says "on Windows, skip step 3".""")


# ---------------------------------------------------------------------------
# PART 3 — resolution, demonstrated
# ---------------------------------------------------------------------------

def part3_resolution() -> None:
    banner("PART 3 — resolution is a constraint-satisfaction problem")

    print("""
    pip and uv both do BACKTRACKING resolution: find a set of versions
    satisfying every constraint from every package simultaneously. When no
    such set exists you get a ResolutionImpossible, and the message is
    usually long and occasionally useless.

    THE CLASSIC CONFLICT:

        your-app depends on
          isc-rag    which requires  pydantic>=2.7,<3.0
          old-tool   which requires  pydantic>=1.8,<2.0

    No version of pydantic satisfies both. The resolver is not being
    difficult; the request is genuinely unsatisfiable.

    HOW TO READ A RESOLUTION FAILURE, in order:
      1. Find the two constraints that conflict. pip's message lists the
         chain; uv's is considerably clearer and worth using just for this.
      2. Ask which one is WRONG. Usually one package is unmaintained and
         pinned too tightly.
      3. `pip install pkg==X --dry-run` to see what a specific version pulls.

    THE FIXES, in order of preference:
      * upgrade the package with the stale constraint
      * drop it
      * fork or vendor it — a real option for a small unmaintained dependency
      * a constraints file to force a resolution you have TESTED
      * separate the two into different processes, if they genuinely cannot
        coexist

    WHAT NOT TO DO: `--no-deps`, or forcing an install that the resolver
    rejected. You have not fixed the conflict; you have hidden it until
    runtime, where it surfaces as an AttributeError in a library you did not
    write.""")

    section("what a resolver actually reports")
    code_block("""\
$ uv pip install "pydantic<2" "isc-rag==0.3.0"

  x No solution found when resolving dependencies:
  |-> Because isc-rag==0.3.0 depends on pydantic>=2.7,<3.0 and you require
      pydantic<2, we can conclude that your requirements are unsatisfiable.

The useful part is the last clause: it names BOTH constraints and the
package each came from. That is enough to decide which one to change.""")


# ---------------------------------------------------------------------------
# PART 4 — lockfiles, built for real
# ---------------------------------------------------------------------------

def part4_lockfiles() -> None:
    banner("PART 4 — lockfiles: what they add beyond a version")

    print("""
    A LOCKFILE RECORDS:
      * the EXACT version of every package, direct AND transitive
      * a HASH of each artifact
      * the markers under which each applies
      * (often) the resolution's target Python and platform

    WHY THE TRANSITIVE PART MATTERS MOST. Your pyproject lists four
    dependencies. Those pull in perhaps forty. `pydantic 2.7` requires
    `pydantic-core`, which is a COMPILED package with a version pinned
    exactly — and a `pydantic-core` mismatch is a segfault or an ImportError,
    not a nice error message. Your pyproject cannot express any of that; the
    lock does.

    WHY THE HASHES MATTER. Without them, a compromised or replaced artifact
    at the same version installs silently. With `--require-hashes`, pip
    refuses anything whose hash does not match. For anything touching an
    export-controlled or regulated environment, that is the difference
    between "we pin versions" and "we can prove what we installed".""")

    section("generating one — the three current options")
    code_block("""\
# uv (fastest, and the lockfile is cross-platform by design)
uv lock                          # writes uv.lock from pyproject.toml
uv sync                          # installs exactly the lock
uv export --format requirements-txt --no-hashes > requirements.txt

# pip-tools (the incumbent; requirements-in, requirements-out)
pip-compile --generate-hashes pyproject.toml -o requirements.txt
pip-sync requirements.txt        # installs, AND REMOVES anything extra

# pip alone (no resolver state, so it is a snapshot rather than a lock)
pip freeze > requirements.txt    # see the warning below""")

    print("""
    `pip freeze` IS NOT A LOCKFILE. It records what happens to be installed,
    including your editable local package, anything you installed by hand,
    and any dev tooling in the same environment. It has no hashes and no
    record of WHY anything is there.

    It is a snapshot of an environment, which is occasionally useful for
    debugging and never appropriate as a build input.

    `pip-sync` deserves a specific mention: it UNINSTALLS packages not in the
    lockfile. `pip install -r` does not, so an environment drifts as
    dependencies are removed from the file but not from the machine. That
    drift is a genuine source of "works on the runner, fails in the image".""")


# ---------------------------------------------------------------------------
# PART 5 — demonstrating that an unlocked install is not reproducible
# ---------------------------------------------------------------------------

def part5_reproducibility() -> None:
    banner("PART 5 — the library/application split, in one repo")

    lib_toml = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.3.0"
requires-python = ">=3.11"
# RANGES. This is a library; it must compose.
dependencies = [
    "httpx>=0.27,<1.0",
    "pydantic>=2.7,<3.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
'''

    lock_excerpt = '''\
# uv.lock (excerpt) — the APPLICATION's resolved tree
version = 1
requires-python = ">=3.11"

[[package]]
name = "isc-rag"
version = "0.3.0"
source = { editable = "." }
dependencies = [
    { name = "httpx" },
    { name = "pydantic" },
]

[[package]]
name = "pydantic"
version = "2.11.7"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "annotated-types" },
    { name = "pydantic-core" },     # <- TRANSITIVE, compiled, exact-pinned
    { name = "typing-extensions" },
]
wheels = [
    { url = "...", hash = "sha256:0c6b3f...", size = 444782 },
]

[[package]]
name = "pydantic-core"
version = "2.33.2"                  # <- your pyproject never mentioned this
source = { registry = "https://pypi.org/simple" }
'''

    section("the LIBRARY declares ranges")
    code_block(lib_toml)

    section("the APPLICATION's lock resolves them exactly")
    code_block(lock_excerpt)

    print("""
    NOTE `pydantic-core` IN THE LOCK AND NOWHERE IN THE PYPROJECT. It is a
    compiled extension whose ABI must match the `pydantic` version exactly.
    Nothing you wrote mentions it. Only the lock records it, and only the
    lock can reproduce it.

    THE RULE FOR A REPO CONTAINING BOTH:
      pyproject.toml    ranges. Committed. Reviewed like an API change.
      uv.lock           exact. Committed. Regenerated by a tool, reviewed
                        for SURPRISES rather than read line by line.
      Dockerfile        installs from the LOCK, never from pyproject.

    AND THE CI CHECK THAT KEEPS THEM HONEST:

        uv lock --check          # fails if the lock is stale w.r.t. pyproject

    Without it, someone edits a version range, does not regenerate, and the
    deployed tree silently no longer matches the declared constraints.""")


# ---------------------------------------------------------------------------
# PART 6 — reproducibility beyond the lockfile
# ---------------------------------------------------------------------------

def part6_beyond() -> None:
    banner("PART 6 — a lockfile is necessary and not sufficient")

    print("""
    THINGS A LOCKFILE DOES NOT PIN, in rough order of how often they bite:

    1. THE PYTHON VERSION. `uv.lock` records `requires-python`, not the
       interpreter you will run. 3.11 and 3.12 resolve differently and behave
       differently. Pin it in the Dockerfile AND in `.python-version`.

    2. THE PLATFORM. A lock resolved on macOS may reference wheels that do
       not exist for linux/arm64. `uv lock` is cross-platform by design; a
       `pip-compile` lock is not unless you generate one per platform.

    3. SYSTEM LIBRARIES. `onnxruntime` needs a libc version; `psycopg2` needs
       libpq. The wheel pins neither. This is what the base image is for.

    4. THE BASE IMAGE ITSELF. `FROM python:3.12-slim` moves. Pin by DIGEST:
           FROM python:3.12-slim@sha256:...
       That is the single highest-value line in most Dockerfiles and it is
       almost never there.

    5. BUILD-TIME NETWORK STATE. If your build can reach the internet, it can
       reach a DIFFERENT internet tomorrow. `--require-hashes` plus a private
       index makes the build deterministic in a way a lockfile alone does not.

    6. NON-DETERMINISTIC ARTIFACTS. `.pyc` timestamps, file ordering in the
       wheel. Set `SOURCE_DATE_EPOCH` and `PYTHONHASHSEED` if you need
       bit-identical builds — you probably do not, but if a compliance
       requirement says "reproducible build", this is what it means.

    THE PRACTICAL LADDER, and most teams should stop at level 3:
      1. pyproject ranges only                  not reproducible
      2. + a committed lockfile                 same versions
      3. + hashes and a pinned base image       same BYTES, verifiable
      4. + a fully hermetic builder             bit-identical, expensive
""")

    section("what a hardened install line looks like")
    code_block("""\
# Dockerfile
FROM python:3.12-slim@sha256:1e5a1e1d5b0d5a3f... AS base

# Install from the LOCK, with hashes enforced, from OUR index only.
COPY requirements.lock.txt .
RUN pip install --require-hashes --no-deps \\
        --index-url https://pkgs.dev.azure.com/isc/_packaging/isc-py/pypi/simple/ \\
        -r requirements.lock.txt

# `--no-deps` is CORRECT here and only here: the lock already contains the
# full transitive closure, so letting pip resolve again would be both slower
# and a way for something unlocked to slip in.""")


def main() -> None:
    part1_specifiers()
    part2_markers()
    part3_resolution()
    part4_lockfiles()
    part5_reproducibility()
    part6_beyond()

    banner("SUMMARY")
    print("""
  * LIBRARIES declare ranges (they must compose). APPLICATIONS declare a
    lockfile (nothing depends on them).
  * `>=X.Y,<NEXT_MAJOR` is clearer than `~=`, whose meaning changes with the
    number of components.
  * Upper bounds have a real cost — they block downstream on your release
    cadence. Cap anyway, and treat raising the cap as routine.
  * `python_version >= '3.9'` is FALSE on 3.10. String comparison.
  * The lock's value is the TRANSITIVE closure and the hashes, not the
    versions you already wrote down.
  * `pip freeze` is a snapshot, not a lock. `pip-sync`/`uv sync` remove
    extras; `pip install -r` does not.
  * `uv lock --check` in CI, or the lock silently drifts from pyproject.
  * A lockfile does not pin Python, the platform, system libraries, or the
    base image. Pin the base image BY DIGEST.
""")


if __name__ == "__main__":
    main()
