"""
01 — Project layout: why `src/` is not a style preference.

THE CLAIM
---------
A flat layout lets your tests import the SOURCE TREE instead of the INSTALLED
PACKAGE. That means your test suite can pass against a package that is broken
when a user installs it — because your tests never exercised the thing you
shipped.

Everyone has heard this. Almost nobody has seen it. So this script builds both
layouts, ships a DELIBERATELY BROKEN wheel from each, and runs the same import
from the project root. One reports success against a broken artifact.

Run:  python 01_layout.py
"""

from __future__ import annotations

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
# The package: a tiny model-provider registry, in both layouts.
# ---------------------------------------------------------------------------

PYPROJECT_FLAT = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.1.0"
description = "ISC RAG toolkit"
requires-python = ">=3.11"
"""

# THE BUG, and it is a realistic one: the wheel ships only the top-level
# module and omits the `providers` subpackage. In a flat layout that is easy
# to do by accident — the backend has to GUESS which top-level directories
# are packages, and it guesses from what is in the project root.
PYPROJECT_FLAT_BROKEN = PYPROJECT_FLAT + """
[tool.hatch.build.targets.wheel]
# Someone listed the modules explicitly and forgot one. Or a `packages`
# setting drifted. Either way the wheel is missing a subpackage.
only-include = ["isc_rag/__init__.py", "isc_rag/config.py"]
sources = ["."]
"""

PYPROJECT_SRC = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.1.0"
description = "ISC RAG toolkit"
requires-python = ">=3.11"

[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
"""

PYPROJECT_SRC_BROKEN = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.1.0"
requires-python = ">=3.11"

[tool.hatch.build.targets.wheel]
only-include = ["src/isc_rag/__init__.py", "src/isc_rag/config.py"]
sources = ["src"]
"""

INIT = '''\
"""isc_rag — a toolkit for ISC retrieval-augmented generation."""
__version__ = "0.1.0"
'''

CONFIG = '''\
DEFAULT_TOP_K = 4
DEFAULT_DEPLOYMENT = "gpt-4o-mini-prod"
'''

PROVIDERS_INIT = '''\
"""Model providers. THIS IS THE SUBPACKAGE THE BROKEN WHEELS OMIT."""
from isc_rag.providers.azure import AzureProvider

__all__ = ["AzureProvider"]
'''

PROVIDERS_AZURE = '''\
class AzureProvider:
    name = "azure-openai"

    def __init__(self, deployment: str) -> None:
        self.deployment = deployment
'''

# The import a user (or a test) performs.
SMOKE = (
    "import isc_rag, isc_rag.providers;"
    "print('OK', isc_rag.__version__, isc_rag.providers.AzureProvider.name)"
)


def make_project(*, src_layout: bool, broken: bool) -> ProjectBuilder:
    b = ProjectBuilder.in_temp("src" if src_layout else "flat")
    prefix = "src/isc_rag" if src_layout else "isc_rag"
    if src_layout:
        b.add("pyproject.toml", PYPROJECT_SRC_BROKEN if broken else PYPROJECT_SRC)
    else:
        b.add("pyproject.toml", PYPROJECT_FLAT_BROKEN if broken else PYPROJECT_FLAT)
    b.add(f"{prefix}/__init__.py", INIT)
    b.add(f"{prefix}/config.py", CONFIG)
    b.add(f"{prefix}/providers/__init__.py", PROVIDERS_INIT)
    b.add(f"{prefix}/providers/azure.py", PROVIDERS_AZURE)
    b.add("tests/test_smoke.py",
          "import isc_rag.providers\n\n"
          "def test_provider_importable():\n"
          "    assert isc_rag.providers.AzureProvider.name == 'azure-openai'\n")
    b.write()
    return b


# ---------------------------------------------------------------------------
# PART 1 — the two layouts
# ---------------------------------------------------------------------------

def part1_layouts() -> None:
    banner("PART 1 — the two layouts, side by side")

    flat = make_project(src_layout=False, broken=False)
    src = make_project(src_layout=True, broken=False)

    section("FLAT layout")
    code_block(flat.tree())
    section("SRC layout")
    code_block(src.tree())

    print("""
    THE DIFFERENCE IS ONE DIRECTORY, and it changes one thing: whether the
    importable package sits at the project root.

    In the FLAT layout, `isc_rag/` is directly under the root. Python puts the
    current directory on `sys.path` in several common situations, so
    `import isc_rag` from the project root finds the SOURCE TREE — installed
    or not, correct or not.

    In the SRC layout there is nothing importable at the root. `import isc_rag`
    can ONLY resolve to something on `sys.path` — i.e. something that was
    actually installed.""")

    flat.cleanup()
    src.cleanup()


# ---------------------------------------------------------------------------
# PART 2 — the demonstration
# ---------------------------------------------------------------------------

def part2_broken_wheels() -> None:
    banner("PART 2 — ship a BROKEN wheel from each layout and import it")

    results: dict[str, dict[str, object]] = {}

    for label, use_src in (("flat", False), ("src", True)):
        proj = make_project(src_layout=use_src, broken=True)
        build = build_project(proj.root, sdist=False)
        if not build.ok or build.wheel is None:
            show(f"{label}: build", f"FAILED — {build.error_summary}")
            proj.cleanup()
            continue

        wheel = WheelInspector(build.wheel)
        venv = VenvRunner(proj.root)
        venv.create()
        install = venv.install(str(build.wheel))

        # THE CRUCIAL DETAIL: run from the PROJECT ROOT, which is what a test
        # runner does. That is the condition under which the flat layout
        # shadows the installed package.
        from_root = venv.run_python(SMOKE, cwd=proj.root)
        # ...and from somewhere else, which is what a USER experiences.
        from_elsewhere = venv.run_python(SMOKE, cwd=Path("/tmp"))

        results[label] = {
            "wheel_files": wheel.package_files,
            "ships_providers": wheel.has("providers/"),
            "installed": install.ok,
            "from_root": from_root,
            "from_elsewhere": from_elsewhere,
        }
        proj.cleanup()

    for label in ("flat", "src"):
        r = results.get(label)
        if not r:
            continue
        section(f"{label.upper()} layout, broken wheel")
        show("files in the wheel", r["wheel_files"])
        show("wheel ships the providers subpackage?", r["ships_providers"])
        from_root = r["from_root"]           # type: ignore[assignment]
        from_elsewhere = r["from_elsewhere"]  # type: ignore[assignment]
        show("import run FROM THE PROJECT ROOT",
             from_root.output if from_root.ok else       # type: ignore[union-attr]
             f"FAILED: {from_root.first_error}")          # type: ignore[union-attr]
        show("import run from elsewhere",
             from_elsewhere.output if from_elsewhere.ok else   # type: ignore[union-attr]
             f"FAILED: {from_elsewhere.first_error}")           # type: ignore[union-attr]

    print("""
    READ THE FLAT ROWS. The wheel does NOT contain the providers subpackage —
    it is genuinely broken. And the import run from the project root SUCCEEDS,
    because `''` (the cwd) is on `sys.path` and the source tree is right
    there. A test suite run from the project root would be green.

    The same import run from anywhere else fails, which is what every user of
    that wheel experiences.

    THE SRC ROWS FAIL IN BOTH PLACES, which is the correct outcome. There is
    nothing importable at the project root, so the only `isc_rag` available is
    the installed one — and the installed one is broken, so you find out.

    THAT IS THE ENTIRE ARGUMENT FOR src/. Not tidiness. It removes the
    possibility of testing something other than what you ship.""")


# ---------------------------------------------------------------------------
# PART 3 — the correct wheels, for contrast
# ---------------------------------------------------------------------------

def part3_correct_wheels() -> None:
    banner("PART 3 — the same layouts, built correctly")

    for label, use_src in (("flat", False), ("src", True)):
        proj = make_project(src_layout=use_src, broken=False)
        build = build_project(proj.root, sdist=False)
        section(f"{label.upper()} layout, correct wheel")
        if not build.ok or build.wheel is None:
            show("build", f"FAILED — {build.error_summary}")
            proj.cleanup()
            continue
        wheel = WheelInspector(build.wheel)
        show("wheel", wheel.summary())
        show("ships providers?", wheel.has("providers/"))

        venv = VenvRunner(proj.root)
        venv.create()
        venv.install(str(build.wheel))
        out = venv.run_python(SMOKE, cwd=Path("/tmp"))
        verdict(out.ok, f"import from elsewhere: "
                        f"{out.output if out.ok else out.first_error}")
        proj.cleanup()

    print("""
    Both work when built correctly. src/ does not make your package better;
    it makes a BROKEN package impossible to miss.""")


# ---------------------------------------------------------------------------
# PART 4 — editable installs
# ---------------------------------------------------------------------------

def part4_editable() -> None:
    banner("PART 4 — editable installs, and what they do to the argument")

    proj = make_project(src_layout=True, broken=False)
    venv = VenvRunner(proj.root)
    # system_site so the already-present build backend is visible offline.
    # See the note on VenvRunner.create — this is a tutorial affordance, not
    # something to copy into CI.
    venv.create(system_site=True)
    install = venv.install("-e", ".", offline=True,
                           no_build_isolation=True)

    section("pip install -e . with a src layout")
    show("install succeeded", install.ok)
    if install.ok:
        out = venv.run_python(SMOKE, cwd=Path("/tmp"))
        verdict(out.ok, f"import from elsewhere: "
                        f"{out.output if out.ok else out.first_error}")
        edit = proj.root / "src" / "isc_rag" / "__init__.py"
        edit.write_text(edit.read_text().replace('"0.1.0"', '"0.1.1-dev"'))
        out2 = venv.run_python(
            "import isc_rag; print(isc_rag.__version__, isc_rag.__file__)",
            cwd=Path("/tmp"))
        show("after editing the source (no reinstall)", out2.output)
        show("note", "__file__ points at src/ — that is the path hook")
    else:
        show("install output", install.first_error or "(see note)")
        show("note", "an editable install needs the build backend AND the "
                     "`editables` package available at install time")

    proj.cleanup()

    print("""
    An editable install puts a PATH HOOK on sys.path pointing at your `src/`
    directory, so edits are picked up without reinstalling — while the package
    is still resolved through the normal import machinery rather than by
    accident of cwd.

    THE HONEST CAVEAT: an editable install weakens the src-layout guarantee,
    because now `isc_rag` IS importable from anywhere regardless of what the
    wheel would contain. Editable installs are the right development
    experience; they are not a substitute for testing the built artifact.

    SO THE CI RULE THAT ACTUALLY MATTERS:
        build the wheel  ->  install it in a CLEAN venv  ->  run the tests
        AGAINST THE INSTALLED PACKAGE, from a directory that is not the
        project root.

    That is `tox`, `nox`, or four lines of shell. Without it, src/ buys you
    less than you think — and with it, even a flat layout is mostly safe.
    Script 09 has the working version.""")


# ---------------------------------------------------------------------------
# PART 5 — the rest of the tree
# ---------------------------------------------------------------------------

def part5_conventions() -> None:
    banner("PART 5 — the rest of the layout")

    print("""
    isc-rag/
      pyproject.toml          ONE file. Metadata, deps, and tool config.
      README.md               rendered on the package page. Keep it current.
      LICENSE                 required for internal distribution too.
      CHANGELOG.md            what changed, per version. See script 07.
      src/
        isc_rag/
          __init__.py         the public API. See below.
          py.typed            THE MARKER THAT MAKES YOUR TYPES VISIBLE.
          config.py
          providers/
          _internal/          leading underscore = not public API
      tests/                  OUTSIDE src/. Not shipped.
      docs/
      .github/workflows/

    THREE THINGS THAT ARE EASY TO GET WRONG:

    1. `py.typed` — an EMPTY file, but without it every downstream consumer's
       mypy silently treats your package as untyped `Any`. All your careful
       annotations become invisible at the package boundary. It must also be
       INCLUDED IN THE WHEEL, which is a separate step (script 05).

    2. TESTS OUTSIDE THE PACKAGE. `src/isc_rag/tests/` ships your test suite
       to every user, along with its fixtures and any credentials in them. It
       also means `import isc_rag.tests` works in production. Keep them at
       `tests/`.

    3. `_internal/` — a leading underscore is the only convention Python has
       for "not public". It is not enforced, but it tells both readers and
       tooling (and `__all__`) what you are committing to support.

    ON `__init__.py` — decide deliberately:

        # EXPLICIT (recommended for a library)
        from isc_rag.providers.azure import AzureProvider
        __all__ = ["AzureProvider"]

    That gives users one import path and lets you move the implementation
    without breaking them. The cost is import time: everything in `__init__`
    is imported when anyone imports anything. For a package that pulls in
    heavy dependencies, that cost is real, and a lazy `__getattr__` (PEP 562)
    is the escape hatch.""")


def main() -> None:
    part1_layouts()
    part2_broken_wheels()
    part3_correct_wheels()
    part4_editable()
    part5_conventions()

    banner("SUMMARY")
    print("""
  * A flat layout puts the source tree on sys.path when you run from the
    project root — so tests can pass against a broken wheel. DEMONSTRATED
    above, not asserted.
  * src/ removes that possibility. It does not make the package better; it
    makes a broken one impossible to miss.
  * Editable installs weaken the guarantee. The real control is: build the
    wheel, install it in a clean venv, run the tests from elsewhere.
  * `py.typed` is empty and load-bearing. Without it your annotations are
    invisible to every consumer.
  * Tests live OUTSIDE the package, or you ship them.
""")


if __name__ == "__main__":
    main()
