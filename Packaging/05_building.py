"""
05 — Build artifacts: wheels, sdists, and what actually gets included.

THE TWO ARTIFACTS
-----------------
    SDIST   a .tar.gz of your SOURCE plus a pyproject.toml. Installing it
            requires running your build backend on the user's machine.
    WHEEL   a .zip of the INSTALLED LAYOUT. Installing it is an unzip and a
            metadata write. No build step, no backend, no compiler.

Ship both. The wheel is what everyone installs; the sdist is what makes your
package buildable on a platform you did not publish a wheel for, and what
some corporate policies require for auditability.

THE FAILURE THIS SCRIPT IS REALLY ABOUT
---------------------------------------
"It works in development but the installed package is missing my prompt
templates." That is the single most common packaging bug in an LLM codebase,
because prompts, JSON schemas, and golden sets are DATA FILES, and data files
are not included by default.

Run:  python 05_building.py
"""

from __future__ import annotations

import tarfile
import zipfile
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

BASE_PYPROJECT = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.3.0"
requires-python = ">=3.11"
'''

DEFAULT_WHEEL_TARGET = '''
[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
'''

PROMPT_LOADER = '''\
"""Loading a packaged data file.

THE WRONG WAY, which works in development and fails after install:

    from pathlib import Path
    PROMPTS = Path(__file__).parent / "prompts"
    text = (PROMPTS / "disposition.txt").read_text()

It fails when the package is inside a zip (a zipapp, some Lambda layouts, a
frozen build), and `__file__` is not guaranteed to exist at all. It also
silently reads from the SOURCE TREE during development, which is exactly the
shadowing problem from script 01 — so a missing data file is invisible until
someone installs the wheel.

THE RIGHT WAY: importlib.resources, which works regardless of how the package
is stored.
"""
from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    return (files("isc_rag.prompts") / f"{name}.txt").read_text(encoding="utf-8")


def load_schema(name: str) -> str:
    return (files("isc_rag.schemas") / f"{name}.json").read_text(encoding="utf-8")
'''


def make_project(*, include_data: bool, extra_config: str = "") -> ProjectBuilder:
    b = ProjectBuilder.in_temp("build")
    # A TOML table may be declared only ONCE. Appending a second
    # `[tool.hatch.build.targets.wheel]` produces:
    #     Cannot declare ('tool','hatch','build','targets','wheel') twice
    # which is a build failure, not a merge. So the caller either supplies
    # that table or gets the default — never both.
    b.add("pyproject.toml",
          BASE_PYPROJECT + (extra_config or DEFAULT_WHEEL_TARGET))
    b.add("README.md", "# isc-rag\n")
    b.add("src/isc_rag/__init__.py", '__version__ = "0.3.0"\n')
    b.add("src/isc_rag/py.typed", "")
    b.add("src/isc_rag/resources.py", PROMPT_LOADER)
    if include_data:
        # A data file needs an __init__.py to be addressable by
        # `files("isc_rag.prompts")` — it must be a PACKAGE, not just a
        # directory. This is a small detail that costs an afternoon.
        b.add("src/isc_rag/prompts/__init__.py", "")
        b.add("src/isc_rag/prompts/disposition.txt",
              "You are an ISC invoice exception analyst.\n{context}\n")
        b.add("src/isc_rag/schemas/__init__.py", "")
        b.add("src/isc_rag/schemas/proposal.json",
              '{"type": "object", "required": ["disposition"]}')
    b.add("tests/test_smoke.py", "def test_ok():\n    assert True\n")
    b.add("notes.md", "internal scratch notes, should NOT ship")
    b.write()
    return b


# ---------------------------------------------------------------------------
# PART 1 — what is in each artifact
# ---------------------------------------------------------------------------

def part1_artifacts() -> None:
    banner("PART 1 — sdist and wheel contain different things")

    proj = make_project(include_data=True)
    result = build_project(proj.root, sdist=True)
    if not result.ok or not result.wheel or not result.sdist:
        show("build", f"FAILED: {result.error_summary}")
        proj.cleanup()
        return

    section("WHEEL contents")
    wheel = WheelInspector(result.wheel)
    for name in wheel.names:
        print(f"      {name}")

    section("SDIST contents")
    with tarfile.open(result.sdist) as t:
        for name in sorted(t.getnames())[:20]:
            print(f"      {name}")

    print("""
    THE SDIST CONTAINS `tests/` AND `README.md`; THE WHEEL DOES NOT.

    That is the right default for both. The sdist is a source distribution —
    someone rebuilding from it wants the tests. The wheel is the installed
    layout — a user does not want your test suite in their site-packages,
    and shipping it means `import isc_rag.tests` works in production and your
    fixtures ship with it.

    NOTE the `.dist-info/` directory in the wheel: METADATA, RECORD, WHEEL,
    and entry_points.txt. That is the installed-distribution metadata, and it
    is what `importlib.metadata` reads at runtime.

    AND LOOK AT `notes.md` IN THE SDIST. That file is labelled "internal
    scratch notes, should NOT ship" in the fixture, and it shipped — because
    hatchling's sdist default is "everything not gitignored", and there is no
    .gitignore here.

    That is the sdist's default failure mode and it is worth internalising:
    a wheel ships too LITTLE by default (script 05 part 2), and an sdist
    ships too MUCH. PART 3 constrains the second.""")

    proj.cleanup()


# ---------------------------------------------------------------------------
# PART 2 — the data-file trap
# ---------------------------------------------------------------------------

LOAD_TEST = (
    "from isc_rag.resources import load_prompt;"
    "print('OK:', load_prompt('disposition')[:40])"
)


def part2_data_files() -> None:
    banner("PART 2 — the data-file trap, demonstrated")

    section("A) data files present in the source tree, default config")
    proj = make_project(include_data=True)
    result = build_project(proj.root, sdist=False)
    if result.wheel:
        wheel = WheelInspector(result.wheel)
        show("wheel includes prompts/*.txt?", wheel.has("prompts/disposition.txt"))
        show("wheel includes schemas/*.json?", wheel.has("schemas/proposal.json"))
        venv = VenvRunner(proj.root)
        venv.create()
        venv.install(str(result.wheel))
        out = venv.run_python(LOAD_TEST, cwd=Path("/tmp"))
        verdict(out.ok, out.output if out.ok else out.first_error)
    proj.cleanup()

    print("""
    HATCHLING INCLUDED THEM. It ships everything under the configured package
    directory by default, which is a sensible default and is why this
    tutorial uses it.

    SETUPTOOLS DOES NOT. With setuptools, the same project produces a wheel
    with the .py files and NOT the .txt and .json files, and the failure
    appears only after install:

        FileNotFoundError: .../site-packages/isc_rag/prompts/disposition.txt

    THE SETUPTOOLS FIX, and you need one of these:

        [tool.setuptools.package-data]
        isc_rag = ["prompts/*.txt", "schemas/*.json", "py.typed"]

    or

        [tool.setuptools]
        include-package-data = true      # then use MANIFEST.in for the sdist

    THE POINT IS NOT WHICH BACKEND. It is that FILE INCLUSION IS
    BACKEND-SPECIFIC AND SILENT WHEN WRONG. Never assume; build the wheel and
    look inside. Script 09 makes that a test.""")


# ---------------------------------------------------------------------------
# PART 3 — excluding what should not ship
# ---------------------------------------------------------------------------

def part3_exclusions() -> None:
    banner("PART 3 — keeping things OUT")

    extra = '''
[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
exclude = [
    "**/*.pyc",
    "**/__pycache__",
    "**/conftest.py",
    "**/_scratch/**",
]

[tool.hatch.build.targets.sdist]
# The sdist should be buildable, so it needs more than the wheel — but it
# still should not contain your CI secrets or a 2GB fixture corpus.
include = [
    "src/",
    "tests/",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
]
'''
    proj = make_project(include_data=True, extra_config=extra)
    # A file that must never ship.
    (proj.root / "src" / "isc_rag" / "_scratch").mkdir(parents=True, exist_ok=True)
    (proj.root / "src" / "isc_rag" / "_scratch" / "local_creds.py").write_text(
        'API_KEY = "sk-LOCAL-DEV-KEY-DO-NOT-SHIP"\n')
    (proj.root / "src" / "isc_rag" / "conftest.py").write_text("# test config\n")

    result = build_project(proj.root, sdist=True)
    if not result.ok:
        section("build FAILED")
        show("error", result.error_summary)
    if result.wheel:
        wheel = WheelInspector(result.wheel)
        section("exclusions honoured?")
        verdict(not wheel.has("_scratch"), "_scratch/ excluded from the wheel")
        verdict(not wheel.has("conftest.py"), "conftest.py excluded")
        verdict(wheel.has("prompts/"), "prompts/ still included")
        section("full wheel contents")
        for n in wheel.package_files:
            print(f"      {n}")
    proj.cleanup()

    print("""
    THINGS THAT SHOULD NEVER BE IN A WHEEL, and all of them have shipped in
    someone's package:

      * .env files and anything under a `secrets/` directory
      * tests and their fixtures
      * `conftest.py` (it is test config, and pytest will find it if shipped)
      * notebooks, scratch scripts, `_local/` directories
      * large fixture corpora — a 200MB wheel because someone vendored a
        golden set
      * `.git`, CI config, editor settings

    THE MECHANICAL CONTROL: `check-wheel-contents dist/*.whl` in CI, plus an
    explicit test asserting the wheel does NOT contain patterns you care
    about. Reviewing a file list by eye works until the day it does not.""")


# ---------------------------------------------------------------------------
# PART 4 — reading the wheel filename
# ---------------------------------------------------------------------------

def part4_filenames() -> None:
    banner("PART 4 — the wheel filename is metadata")

    proj = make_project(include_data=True)
    result = build_project(proj.root, sdist=False)
    if result.wheel:
        w = WheelInspector(result.wheel)
        show("filename", result.wheel.name)
        for k, v in w.filename_parts.items():
            show(f"  {k}", v)
    proj.cleanup()

    print("""
        {name}-{version}-{python}-{abi}-{platform}.whl

    py3-none-any        PURE PYTHON. Any Python 3, no ABI dependency, any
                        platform. One wheel serves everyone. This is what a
                        library like isc-rag should produce.

    cp312-cp312-manylinux_2_17_x86_64
                        COMPILED for CPython 3.12 on glibc >= 2.17, x86_64.
                        You need one wheel per (python, platform) combination
                        — which is why numpy publishes dozens.

    THE DIAGNOSTIC VALUE: if a package you believe is pure Python produces a
    `cp312` wheel, something is compiling — usually an accidentally-enabled
    Cython or mypyc build. And if `pip install` on a new platform starts
    building from source and failing, it is because no wheel matches that
    platform tag and pip fell back to the sdist.

    `manylinux` DESERVES A SENTENCE: it is a policy defining which glibc
    symbols a Linux wheel may use, so one wheel works across distributions.
    `manylinux_2_17` means glibc 2.17 or newer. A wheel built on your Ubuntu
    laptop without the manylinux toolchain gets tagged `linux_x86_64`, which
    PyPI rejects and which will not install on a different distro.""")


# ---------------------------------------------------------------------------
# PART 5 — the build pipeline
# ---------------------------------------------------------------------------

def part5_pipeline() -> None:
    banner("PART 5 — the release pipeline, in order")

    print("""
    1. CLEAN
           rm -rf dist/ build/ *.egg-info
       Stale artifacts in `dist/` get uploaded. `twine upload dist/*` with
       last month's wheel still there is a real and embarrassing failure.

    2. BUILD, WITH ISOLATION
           python -m build
       Isolation (the default) builds in a fresh environment containing only
       `build-system.requires`. That is what PROVES your requires list is
       complete — `--no-isolation` uses your ambient environment and will
       happily succeed with a missing declaration.

    3. CHECK THE METADATA
           twine check dist/*
       Catches a malformed README, which otherwise renders as a blank page.

    4. CHECK THE CONTENTS
           check-wheel-contents dist/*.whl
       Catches missing py.typed, shipped tests, common layout mistakes.

    5. INSTALL INTO A CLEAN VENV AND SMOKE-TEST
           python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl
           cd /tmp && /tmp/v/bin/python -c "import isc_rag; ..."
       The `cd /tmp` is the important part — script 01.

    6. RUN THE TESTS AGAINST THE INSTALLED PACKAGE
           cd /tmp && /tmp/v/bin/python -m pytest --pyargs isc_rag_tests
       or use `tox`/`nox`, which automate exactly this loop.

    7. UPLOAD TO A TEST INDEX FIRST
           twine upload -r testpypi dist/*
       For an internal package, a `-dev` feed serves the same purpose.

    8. UPLOAD, THEN TAG
           twine upload dist/* && git tag v0.3.0 && git push --tags
       In that ORDER. If the upload fails you have not tagged a release that
       does not exist. Tags are hard to retract and someone will have pulled
       it.

    STEPS 2-6 ARE FIVE LINES OF CI and they catch essentially every packaging
    defect this tutorial describes.""")


def main() -> None:
    part1_artifacts()
    part2_data_files()
    part3_exclusions()
    part4_filenames()
    part5_pipeline()

    banner("SUMMARY")
    print("""
  * Wheel = installed layout, no build step. Sdist = source, rebuildable.
    Ship both.
  * Data files (prompts, schemas, golden sets) are NOT included by default in
    every backend. Build the wheel and LOOK.
  * A data directory needs `__init__.py` to be addressable by
    `importlib.resources.files()`.
  * Use `importlib.resources`, never `Path(__file__).parent` — the latter
    reads the source tree in development and fails after install.
  * Exclude tests, conftest, scratch dirs, and anything credential-shaped.
  * The filename tells you whether anything compiled.
  * Build WITH isolation in CI; that is what proves build-system.requires is
    complete.
  * Upload, THEN tag.
""")


if __name__ == "__main__":
    main()
