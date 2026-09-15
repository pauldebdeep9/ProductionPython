"""
02 — pyproject.toml: metadata, dependencies, and extras.

WHAT THIS FILE REPLACED
-----------------------
setup.py (executable, and therefore unanalysable without running it),
setup.cfg, MANIFEST.in, requirements.txt, and a tool-specific config file per
tool. PEP 517/518/621 collapsed them into one declarative file.

"Declarative" is the important word. A tool can read your dependencies without
executing arbitrary code, which is why `pip install` no longer needs to run
your setup script to find out what it needs.

THE THREE SECTIONS, and they do genuinely different jobs:

    [build-system]   how to BUILD this. Read FIRST, before anything else.
    [project]        WHAT this is. PEP 621 standard metadata.
    [tool.*]         per-tool config. Not standardised, and that is fine.

Run:  python 02_pyproject.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pkg_lab import (
    ProjectBuilder,
    VenvRunner,
    WheelInspector,
    banner,
    build_project,
    code_block,
    section,
    show,
    verdict,
)

# ---------------------------------------------------------------------------
# A complete, annotated pyproject.toml
# ---------------------------------------------------------------------------

FULL_PYPROJECT = '''\
# ===========================================================================
# [build-system] — read FIRST, by the build frontend, before anything else.
# ===========================================================================
[build-system]
# What must be installed IN THE BUILD ENVIRONMENT to build this package.
# NOT your runtime dependencies. A very common confusion.
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

# WHICH BACKEND, briefly, because the choice matters less than people think:
#   hatchling    good defaults, clean config, fast. A reasonable default.
#   setuptools   the incumbent. Choose it if you need C extensions or a
#                plugin that only exists for setuptools.
#   pdm-backend  good if you use PDM.
#   poetry-core  only if you use Poetry, whose metadata is non-PEP-621.
#   maturin      Rust extensions.
# For a pure-Python library, any of the first three is fine. Do not spend an
# afternoon on this.

# ===========================================================================
# [project] — PEP 621. The same keys work with every backend.
# ===========================================================================
[project]
name = "isc-rag"
# THE NAME IS NORMALISED (PEP 503): "isc-rag", "isc_rag", and "ISC.RAG" are
# the SAME project to pip. But the IMPORT name is whatever directory you
# ship — here `isc_rag`. They do not have to match, and when they do not,
# say so prominently in your README or people cannot find you.

version = "0.3.0"
# Or `dynamic = ["version"]` to read it from one place. See script 07.

description = "Permission-aware RAG toolkit for ISC document intelligence."
readme = "README.md"
requires-python = ">=3.11"
# THE MOST UNDER-SPECIFIED FIELD. It is a PROMISE, and pip enforces it: a
# user on 3.10 gets a clear "requires a different Python" instead of a
# confusing SyntaxError from your match statement.
# Set it to the OLDEST version you actually test against. Not the newest.

license = "MIT"
# PEP 639. Older projects use `license = {text = "MIT"}` or a classifier;
# both still work, and the SPDX string is now preferred.

authors = [{ name = "ISC AI CoE", email = "isc-ai@example.com" }]
keywords = ["rag", "retrieval", "llm", "supply-chain"]

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Typing :: Typed",
    # For an INTERNAL package, add this and it becomes an error to upload
    # to public PyPI. Cheap insurance against a mis-typed repository URL.
    "Private :: Do Not Upload",
]

# ---------------------------------------------------------------------------
# RUNTIME dependencies. What must be present for `import isc_rag` to work.
# ---------------------------------------------------------------------------
dependencies = [
    "httpx>=0.27,<1.0",
    "pydantic>=2.7,<3.0",
    "tenacity>=8.2,<10.0",
    "azure-identity>=1.17,<2.0",
]
# CONSTRAINT STYLE — see script 03. The short version for a LIBRARY:
#   * lower bound = the oldest version whose features you use AND test
#   * upper bound = the next MAJOR, because semver says that may break you
#   * never pin to `==` in a library. You are taking a decision away from
#     every application that depends on you, and two libraries that both
#     pin will eventually be uninstallable together.

# ---------------------------------------------------------------------------
# OPTIONAL dependencies — installed with `pip install isc-rag[azure]`
# ---------------------------------------------------------------------------
[project.optional-dependencies]
# The point: someone who only wants the retrieval helpers should not have to
# install a 200MB ML stack.
azure = [
    "azure-search-documents>=11.4,<12.0",
    "openai>=1.30,<2.0",
]
local = [
    "sentence-transformers>=3.0,<4.0",
    "torch>=2.3",
]
cli = ["typer>=0.12,<1.0", "rich>=13.7"]

# A meta-extra that pulls in the others. Note the SELF-REFERENCE, which is
# legal since pip 21.2 and is the cleanest way to express "everything".
all = ["isc-rag[azure,local,cli]"]

# DEV DEPENDENCIES DO NOT BELONG HERE — see the note after this file.
# They are not something a USER can meaningfully install.

# ---------------------------------------------------------------------------
# ENTRY POINTS — see script 04.
# ---------------------------------------------------------------------------
[project.scripts]
isc-rag = "isc_rag.cli:main"

[project.entry-points."isc_rag.providers"]
azure = "isc_rag.providers.azure:AzureProvider"

[project.urls]
Homepage = "https://internal.example.com/isc-rag"
Repository = "https://github.example.com/isc/isc-rag"
Changelog = "https://github.example.com/isc/isc-rag/blob/main/CHANGELOG.md"
Issues = "https://github.example.com/isc/isc-rag/issues"
# `Changelog` and `Issues` are the two people actually click. Include them.

# ===========================================================================
# PEP 735 dependency groups — for DEVELOPMENT dependencies (Python 3.13+/
# modern pip and uv). NOT installed by consumers of your package.
# ===========================================================================
[dependency-groups]
test = ["pytest>=8.0", "pytest-asyncio>=0.24", "pytest-cov>=5.0"]
lint = ["ruff>=0.6", "mypy>=1.11"]
dev = [{ include-group = "test" }, { include-group = "lint" }]

# ===========================================================================
# [tool.*] — everything else lives here now.
# ===========================================================================
[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
'''

MINIMAL_PYPROJECT = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.3.0"
requires-python = ">=3.11"
'''


# ---------------------------------------------------------------------------
# PART 1 — the minimum, and what each addition buys
# ---------------------------------------------------------------------------

def part1_minimum() -> None:
    banner("PART 1 — the smallest thing that builds")

    section("this is a complete, valid pyproject.toml")
    code_block(MINIMAL_PYPROJECT)

    b = ProjectBuilder.in_temp("minimal")
    b.add("pyproject.toml", MINIMAL_PYPROJECT)
    b.add("src/isc_rag/__init__.py", '__version__ = "0.3.0"\n')
    b.add("pyproject_extra", "")
    b.write()
    # hatchling needs to know where the package is when using src layout.
    (b.root / "pyproject.toml").write_text(
        MINIMAL_PYPROJECT
        + '\n[tool.hatch.build.targets.wheel]\npackages = ["src/isc_rag"]\n')

    result = build_project(b.root, sdist=False)
    if result.ok and result.wheel:
        w = WheelInspector(result.wheel)
        show("built", w.summary())
        show("Requires-Dist", w.requires_dist or "(none)")
    else:
        show("build failed", result.error_summary)
    b.cleanup()

    print("""
    Four required-ish fields and it builds. Everything else in PART 2 is
    there because it buys something specific — which is worth knowing,
    because a pyproject.toml copied wholesale from another project usually
    carries fields nobody can justify.""")


# ---------------------------------------------------------------------------
# PART 2 — the full file, built and inspected
# ---------------------------------------------------------------------------

def build_full_project(*, with_extras: bool = True) -> tuple[ProjectBuilder,
                                                             WheelInspector | None]:
    b = ProjectBuilder.in_temp("full")
    b.add("pyproject.toml", FULL_PYPROJECT)
    b.add("README.md", "# isc-rag\n\nPermission-aware RAG toolkit.\n")
    b.add("src/isc_rag/__init__.py", '__version__ = "0.3.0"\n')
    b.add("src/isc_rag/py.typed", "")
    b.add("src/isc_rag/cli.py",
          "def main() -> int:\n"
          "    print('isc-rag 0.3.0')\n"
          "    return 0\n")
    b.add("src/isc_rag/providers/__init__.py", "")
    b.add("src/isc_rag/providers/azure.py",
          "class AzureProvider:\n    name = 'azure-openai'\n")
    b.write()
    result = build_project(b.root, sdist=False)
    return b, (WheelInspector(result.wheel)
               if result.ok and result.wheel else None)


def part2_full() -> None:
    banner("PART 2 — the full file, and what the wheel records")

    b, wheel = build_full_project()
    if wheel is None:
        show("build", "FAILED")
        b.cleanup()
        return

    show("wheel", wheel.summary())

    section("Requires-Dist as recorded in the wheel METADATA")
    for req in wheel.requires_dist:
        marker = "  <-- extra" if "extra ==" in req else ""
        print(f"      {req}{marker}")

    print("""
    NOTE THE `extra == "azure"` MARKERS. Optional dependencies are recorded
    as ordinary requirements with an ENVIRONMENT MARKER. There is no separate
    mechanism — pip evaluates the marker and installs the requirement only
    when that extra was requested. Understanding that makes extras much less
    mysterious, and it explains why `pip install pkg[typo]` silently installs
    nothing rather than erroring.

    NOTE ALSO WHAT HAPPENED TO THE `all` EXTRA. It was declared as a
    self-reference:

        all = ["isc-rag[azure,local,cli]"]

    but the wheel records its six constituent requirements DIRECTLY under
    `extra == 'all'`. Hatchling resolved the self-reference at BUILD time
    rather than leaving it for pip to resolve at install time.

    That is convenient and worth knowing, because it is BACKEND-SPECIFIC. A
    different backend may emit the self-reference verbatim, which works with
    modern pip and fails with older ones. If you ship an `all` extra, build
    it and read the METADATA rather than assuming — which is a specific
    instance of the general rule in PART 4.""")

    section("py.typed made it into the wheel?")
    verdict(wheel.has("py.typed"),
            "py.typed present — consumer type checkers will see your types")

    section("entry_points.txt")
    code_block(wheel.entry_points or "(none)")

    b.cleanup()


# ---------------------------------------------------------------------------
# PART 3 — extras, installed for real
# ---------------------------------------------------------------------------

def part3_extras() -> None:
    banner("PART 3 — extras: what they are and what they are not")

    print("""
    WHAT EXTRAS ARE FOR: optional FUNCTIONALITY that needs extra
    dependencies. `isc-rag[local]` pulls in torch; `isc-rag[azure]` does not.

    WHAT THEY ARE NOT FOR:
      * DEV DEPENDENCIES. `pip install isc-rag[dev]` asks a USER to install
        your test suite's dependencies. Use PEP 735 `[dependency-groups]`,
        or a requirements-dev file. Extras are part of your PUBLIC API.
      * MUTUALLY EXCLUSIVE choices. There is no way to say "azure OR local,
        not both". If you need that, it is a runtime configuration decision,
        not a packaging one.
      * VERSION SELECTION. `pkg[cuda118]` vs `pkg[cuda121]` is a common
        pattern and a painful one — the resolver will happily install both.

    THE COST OF AN EXTRA: it is a compatibility surface. `isc-rag[azure]`
    that stops working is a breaking change, and removing an extra breaks
    every install command that mentions it. Add them deliberately.

    THE ONE THAT BITES IN ML: a package whose extras differ by ACCELERATOR
    (torch+cu121 vs torch+cpu) cannot be expressed properly here, because the
    difference is in the INDEX the wheel comes from, not the version. That
    is what `[tool.uv.sources]` and `--index-strategy` exist for, and it is
    the single most common reason an ML project's install instructions are
    three commands rather than one.""")

    section("guarding an optional import at runtime")
    code_block('''\
# src/isc_rag/providers/local.py

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:                      # pragma: no cover
    raise ImportError(
        "The local provider requires extra dependencies. "
        "Install them with:  pip install 'isc-rag[local]'"
    ) from exc

# THE POINT: the error message names the FIX. A bare ImportError leaves the
# user googling; this one tells them the exact command. Costs four lines and
# removes a support ticket.''')


# ---------------------------------------------------------------------------
# PART 4 — validating your own metadata
# ---------------------------------------------------------------------------

def part4_validation() -> None:
    banner("PART 4 — validate the metadata, do not eyeball it")

    section("a pyproject with a real mistake in it")
    broken = MINIMAL_PYPROJECT.replace(
        'requires-python = ">=3.11"',
        'requires-python = ">=3.11"\ndependencies = ["httpx >= 0.27, < 1.0", "pydantic ~= 2.7"]\n'
        'license = { text = "MIT" }\nreadme = "README.md"',
    )
    b = ProjectBuilder.in_temp("validate")
    b.add("pyproject.toml", broken + '\n[tool.hatch.build.targets.wheel]\n'
                                     'packages = ["src/isc_rag"]\n')
    b.add("src/isc_rag/__init__.py", "")
    # README.md is DECLARED but NOT CREATED — a classic, and the build should
    # catch it rather than shipping a package with no description.
    b.write()

    result = build_project(b.root, sdist=False)
    show("build succeeded", result.ok)
    if not result.ok:
        show("error", result.error_summary)
        print("""
      The build FAILED because `readme = "README.md"` points at a file that
      does not exist. That is the right behaviour, and it is why you should
      build in CI rather than only at release time — a metadata error found
      at release is found at the worst moment.""")
    b.cleanup()

    print("""
    THE CHECKS WORTH RUNNING IN CI, in order of value:

      python -m build                 does it build at all?
      twine check dist/*              is the metadata renderable? Catches a
                                      malformed README, which otherwise
                                      appears as a blank package page.
      pip install dist/*.whl          in a CLEAN venv (script 01)
      python -c "import isc_rag"      from a directory that is NOT the repo
      check-wheel-contents dist/*.whl catches missing py.typed, stray files,
                                      tests shipped by accident

    That is five lines of CI and it catches essentially every packaging
    mistake before a user does.""")


# ---------------------------------------------------------------------------
# PART 5 — what NOT to put in pyproject.toml
# ---------------------------------------------------------------------------

def part5_antipatterns() -> None:
    banner("PART 5 — pyproject.toml anti-patterns")

    print("""
    1. PINNED DEPENDENCIES IN A LIBRARY
           dependencies = ["httpx==0.27.0"]
       You have now decided httpx's version for every application that
       depends on you. Two libraries that both pin become uninstallable
       together, and the user cannot fix it. Pin in APPLICATIONS (via a
       lockfile — script 03), never in libraries.

    2. NO UPPER BOUND AT ALL
           dependencies = ["pydantic"]
       The opposite failure. Pydantic 3.0 will break you, and the breakage
       will arrive on a random Tuesday in someone else's CI. `<3.0` costs
       nothing and buys you a deliberate migration.

    3. DEV DEPENDENCIES AS AN EXTRA
       Covered in PART 3. `[dependency-groups]` (PEP 735) is the answer now.

    4. `requires-python` COPIED FROM A TEMPLATE
       `>=3.8` on a codebase using `match` and `X | Y` unions is a promise
       you break at import time, with a SyntaxError the user cannot act on.
       Set it to what you test.

    5. BUILD DEPENDENCIES CONFUSED WITH RUNTIME ONES
       `[build-system].requires` is the BUILD environment. Putting `httpx`
       there does not make it available to your users; putting `hatchling` in
       `[project].dependencies` installs a build tool onto every user's
       machine for no reason.

    6. VERSION IN THREE PLACES
       pyproject.toml, `__init__.py`, and a `VERSION` file that disagree. See
       script 07 — pick ONE source of truth.

    7. NO `py.typed` ON A TYPED PACKAGE
       All that annotation work is invisible to consumers. One empty file.
""")


def main() -> None:
    part1_minimum()
    part2_full()
    part3_extras()
    part4_validation()
    part5_antipatterns()

    banner("SUMMARY")
    print("""
  * Three sections: [build-system] (how to build), [project] (PEP 621
    metadata), [tool.*] (everything else).
  * `requires-python` is a promise pip enforces. Set it to what you TEST.
  * Libraries get RANGES, never pins. Applications get lockfiles.
  * Extras are optional FUNCTIONALITY and part of your public API. Dev
    dependencies belong in [dependency-groups].
  * Optional deps are just requirements with an `extra ==` marker — which is
    why a typo'd extra silently installs nothing.
  * Guard optional imports with an error message that names the install
    command.
  * Validate in CI: build, twine check, install into a clean venv, import
    from elsewhere.
""")


if __name__ == "__main__":
    main()
