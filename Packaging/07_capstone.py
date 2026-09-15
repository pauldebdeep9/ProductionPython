"""
07 — Capstone: a complete package, built and verified end to end.

WHAT THIS COMPOSES
------------------
  01  src layout          -> tests cannot shadow the installed package
  02  PEP 621 metadata    -> ranges, extras, entry points, py.typed
  03  dependency policy   -> library ranges, application lock
  04  entry points        -> a console script and a provider plugin group
  05  build artifacts     -> data files included, scratch excluded
  06  versioning          -> one source of truth, read via importlib.metadata

AND THEN IT RUNS THE RELEASE GATE from script 05 part 5, for real:
build -> inspect -> install into a clean venv -> import from elsewhere ->
run the console script -> resolve the entry points -> load the data files.

Every step either passes or prints why. That gate is about fifteen lines of
CI and it catches essentially every packaging defect in this tutorial.

Run:  python 07_capstone.py
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

# ===========================================================================
# THE PACKAGE
# ===========================================================================

PYPROJECT = '''\
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
requires-python = ">=3.11"
description = "Permission-aware RAG toolkit for ISC document intelligence."
readme = "README.md"
license = "MIT"
authors = [{ name = "ISC AI CoE" }]
dynamic = ["version"]

classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Typing :: Typed",
    "Private :: Do Not Upload",
]

# RANGES, because this is a library. The application that deploys it holds
# the lock. Kept dependency-free here so the demo installs offline; the
# commented forms are what a real one looks like.
dependencies = []
# dependencies = [
#     "httpx>=0.27,<1.0",
#     "pydantic>=2.7,<3.0",
# ]

[project.optional-dependencies]
azure = []
cli = []

[project.scripts]
isc-rag = "isc_rag.cli:main"

[project.entry-points."isc_rag.providers"]
azure = "isc_rag.providers.azure:AzureProvider"

[project.urls]
Repository = "https://github.example.com/isc/isc-rag"
Changelog = "https://github.example.com/isc/isc-rag/blob/main/CHANGELOG.md"
Issues = "https://github.example.com/isc/isc-rag/issues"

[dependency-groups]
test = ["pytest>=8.0"]
lint = ["ruff>=0.6", "mypy>=1.11"]

# --- version: ONE source of truth -----------------------------------------
[tool.hatch.version]
path = "src/isc_rag/_version.py"

# --- what ships ------------------------------------------------------------
[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
exclude = ["**/__pycache__", "**/conftest.py", "**/_scratch/**"]

[tool.hatch.build.targets.sdist]
include = ["src/", "tests/", "pyproject.toml", "README.md", "LICENSE",
           "CHANGELOG.md"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
'''

VERSION_MODULE = '''\
"""The single source of truth for the version.

A separate module rather than __init__.py so the backend can read it WITHOUT
importing the package — which matters once __init__ has imports of its own,
because a build-time import would need the runtime dependencies installed.
"""
__version__ = "0.3.0"
'''

INIT = '''\
"""isc_rag — permission-aware retrieval-augmented generation for ISC."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version

from isc_rag.registry import Provider, discover, load_all
from isc_rag.resources import load_prompt

try:
    # The DISTRIBUTION name ("isc-rag"), not the import name ("isc_rag").
    __version__ = _dist_version("isc-rag")
except PackageNotFoundError:                     # pragma: no cover
    from isc_rag._version import __version__     # source checkout fallback

__all__ = ["Provider", "discover", "load_all", "load_prompt", "__version__"]
'''

REGISTRY = '''\
from __future__ import annotations

from functools import cache
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

GROUP = "isc_rag.providers"


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, prompt: str) -> str: ...


@cache
def discover() -> tuple[str, ...]:
    """Names only, cached. Discovery is cheap; LOADING is not."""
    return tuple(sorted(ep.name for ep in entry_points(group=GROUP)))


def load_all() -> tuple[dict[str, type], dict[str, str]]:
    loaded: dict[str, type] = {}
    failed: dict[str, str] = {}
    for ep in entry_points(group=GROUP):
        try:
            obj = ep.load()
        except Exception as exc:                 # noqa: BLE001
            failed[ep.name] = f"{type(exc).__name__}: {exc}"
            continue
        if not (isinstance(obj, type) and hasattr(obj, "name")):
            failed[ep.name] = f"not a Provider: {obj!r}"
            continue
        loaded[ep.name] = obj
    return loaded, failed
'''

RESOURCES = '''\
from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    """Read a packaged prompt template.

    `importlib.resources`, never `Path(__file__).parent` — the latter works
    in a source checkout and fails when the package is zipped, and it silently
    reads the source tree during development so a missing file is invisible
    until someone installs the wheel.
    """
    return (files("isc_rag.prompts") / f"{name}.txt").read_text(encoding="utf-8")


def load_schema(name: str) -> dict:
    import json
    return json.loads(
        (files("isc_rag.schemas") / f"{name}.json").read_text(encoding="utf-8"))
'''

AZURE_PROVIDER = '''\
from __future__ import annotations


class AzureProvider:
    name = "azure-openai"

    def __init__(self, deployment: str = "gpt-4o-mini-prod") -> None:
        self.deployment = deployment

    def complete(self, prompt: str) -> str:
        return f"[{self.deployment}] {prompt[:32]}"
'''

CLI = '''\
from __future__ import annotations

import sys

from isc_rag import __version__
from isc_rag.registry import load_all
from isc_rag.resources import load_prompt


def main() -> int:
    """No required arguments; returns an exit code."""
    loaded, failed = load_all()
    print(f"isc-rag {__version__}")
    print(f"providers: {sorted(loaded)}")
    print(f"prompt bytes: {len(load_prompt('disposition'))}")
    for name, why in failed.items():
        print(f"  FAILED {name}: {why}", file=sys.stderr)
    return 1 if failed else 0
'''

PROMPT = """You are an ISC invoice exception analyst.

Context:
{context}

Question: {question}

Respond with a JSON object matching the disposition schema.
"""

SCHEMA = json.dumps({
    "type": "object",
    "required": ["disposition", "confidence", "reason", "evidence_ids"],
    "properties": {
        "disposition": {"enum": ["approve", "request_credit",
                                 "adjust_quantity", "hold_pending_receipt",
                                 "escalate_to_buyer"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}, indent=2)

CHANGELOG = """\
# Changelog

All notable changes are documented here. This project follows semantic
versioning; see README for what counts as breaking.

## [0.3.0] - 2026-08-20
### Added
- `isc_rag.providers` entry-point group for third-party providers.
- `load_prompt()` / `load_schema()` for packaged resources.

### Changed
- **PROMPT** `disposition` template updated (fingerprint 36206a02 -> 9b41ce70).
  Output wording changes; re-run your evaluations. Minor bump rather than
  patch specifically because the outputs change.

### Deprecated
- `retrieve(k=...)`; use `top_k=`. Removal in 1.0.
"""


def build_package() -> ProjectBuilder:
    b = ProjectBuilder.in_temp("capstone")
    b.add("pyproject.toml", PYPROJECT)
    b.add("README.md", "# isc-rag\n\nPermission-aware RAG toolkit.\n")
    b.add("LICENSE", "MIT License\n")
    b.add("CHANGELOG.md", CHANGELOG)
    b.add("src/isc_rag/_version.py", VERSION_MODULE)
    b.add("src/isc_rag/__init__.py", INIT)
    b.add("src/isc_rag/py.typed", "")
    b.add("src/isc_rag/registry.py", REGISTRY)
    b.add("src/isc_rag/resources.py", RESOURCES)
    b.add("src/isc_rag/cli.py", CLI)
    b.add("src/isc_rag/providers/__init__.py", "")
    b.add("src/isc_rag/providers/azure.py", AZURE_PROVIDER)
    b.add("src/isc_rag/prompts/__init__.py", "")
    b.add("src/isc_rag/prompts/disposition.txt", PROMPT)
    b.add("src/isc_rag/schemas/__init__.py", "")
    b.add("src/isc_rag/schemas/proposal.json", SCHEMA)
    # Things that must NOT ship.
    b.add("src/isc_rag/_scratch/local.py", 'KEY = "sk-LOCAL-DO-NOT-SHIP"\n')
    b.add("src/isc_rag/conftest.py", "# pytest config\n")
    b.add("tests/test_smoke.py",
          "import isc_rag\n\n"
          "def test_version():\n"
          "    assert isc_rag.__version__\n")
    b.write()
    return b


# ===========================================================================
# THE RELEASE GATE
# ===========================================================================

SMOKE = (
    "import isc_rag;"
    "from isc_rag import load_prompt, load_all;"
    "loaded, failed = load_all();"
    "print(json.dumps({"
    "'version': isc_rag.__version__,"
    "'providers': sorted(loaded),"
    "'failed': failed,"
    "'prompt_bytes': len(load_prompt('disposition')),"
    "'schema_keys': sorted(__import__('isc_rag.resources', fromlist=['x'])"
    ".load_schema('proposal').keys()),"
    "}))"
)


def main() -> None:
    banner("THE PACKAGE")
    proj = build_package()
    code_block(proj.tree())

    # ---- 1. BUILD ------------------------------------------------------
    banner("GATE 1 — build")
    result = build_project(proj.root, sdist=True)
    verdict(result.ok, f"python -m build  ->  {result.error_summary or 'ok'}")
    if not result.ok or not result.wheel:
        proj.cleanup()
        return
    show("wheel", result.wheel.name)
    show("sdist", result.sdist.name if result.sdist else "(none)")

    # ---- 2. INSPECT ----------------------------------------------------
    banner("GATE 2 — inspect the artifact")
    wheel = WheelInspector(result.wheel)
    show("summary", wheel.summary())

    section("must be present")
    for pattern, why in [
        ("py.typed", "consumers' type checkers see your annotations"),
        ("prompts/disposition.txt", "packaged prompt template"),
        ("schemas/proposal.json", "packaged JSON schema"),
        ("providers/azure.py", "the built-in provider"),
    ]:
        verdict(wheel.has(pattern), f"{pattern:<28} {why}")

    section("must be absent")
    for pattern, why in [
        ("_scratch", "scratch dir containing a credential"),
        ("conftest.py", "pytest config"),
        ("tests/", "the test suite"),
    ]:
        verdict(not wheel.has(pattern), f"{pattern:<28} {why}")

    section("metadata")
    show("Version", wheel.metadata_field("Version"))
    show("Requires-Python", wheel.metadata_field("Requires-Python"))
    show("pure python wheel", wheel.filename_parts.get("platform_tag") == "any")
    section("entry points")
    code_block(wheel.entry_points)

    # ---- 3. INSTALL INTO A CLEAN VENV ----------------------------------
    banner("GATE 3 — install into a CLEAN venv")
    venv = VenvRunner(proj.root)
    venv.create()
    install = venv.install(str(result.wheel))
    verdict(install.ok, f"pip install {result.wheel.name}")
    show("installed", venv.list_installed())

    # ---- 4. IMPORT FROM ELSEWHERE --------------------------------------
    banner("GATE 4 — import from OUTSIDE the project root")
    out = venv.run_python("import json;" + SMOKE, cwd=Path("/tmp"))
    verdict(out.ok, "import succeeded from /tmp" if out.ok else out.first_error)
    if out.ok:
        code_block(json.dumps(json.loads(out.output), indent=2))

    # ---- 5. CONSOLE SCRIPT ---------------------------------------------
    banner("GATE 5 — the console script")
    script = venv.path / "bin" / "isc-rag"
    verdict(script.exists(), "executable installed on PATH")
    if script.exists():
        run = venv.run_console_script("isc-rag")
        code_block(run.output)
        verdict(run.ok, "exit code 0")

    # ---- 6. THE SDIST REBUILDS -----------------------------------------
    banner("GATE 6 — the sdist rebuilds")
    print("""    A wheel that works proves the wheel works. It says nothing about
    whether someone can rebuild from your sdist — which is what a user on an
    unpublished platform, and some corporate audit processes, actually need.

    The check: unpack the sdist, build a wheel FROM IT, and confirm the two
    wheels contain the same files.""")
    if result.sdist:
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(result.sdist) as t:
                t.extractall(td, filter="data")
            unpacked = next(Path(td).iterdir())
            rebuilt = build_project(unpacked, sdist=False)
            verdict(rebuilt.ok,
                    f"rebuild from sdist -> {rebuilt.error_summary or 'ok'}")
            if rebuilt.ok and rebuilt.wheel:
                w2 = WheelInspector(rebuilt.wheel)
                same = set(wheel.package_files) == set(w2.package_files)
                verdict(same, "the rebuilt wheel contains the same files")
                if not same:
                    show("only in original",
                         sorted(set(wheel.package_files) - set(w2.package_files)))
                    show("only in rebuilt",
                         sorted(set(w2.package_files) - set(wheel.package_files)))

    proj.cleanup()

    banner("THE GATE, AS CI")
    code_block('''\
- run: rm -rf dist/ build/ *.egg-info
- run: python -m build                       # WITH isolation
- run: twine check dist/*
- run: check-wheel-contents dist/*.whl
- run: python -m venv /tmp/v
- run: /tmp/v/bin/pip install dist/*.whl
- run: cd /tmp && /tmp/v/bin/python -c "import isc_rag; print(isc_rag.__version__)"
- run: cd /tmp && /tmp/v/bin/isc-rag
- run: /tmp/v/bin/python -m pytest tests/ --pyargs   # against the INSTALL''')

    banner("WHAT TO DEFEND IN A REVIEW")
    print("""
  1. src/ layout, so a test run from the repo root cannot shadow the
     installed package (script 01 demonstrates the failure).
  2. ONE source of truth for the version, read at runtime via
     `importlib.metadata`, with a source-checkout fallback.
  3. Library RANGES in pyproject; the deploying application holds the lock.
  4. `py.typed` shipped, and ASSERTED to be in the wheel — otherwise every
     consumer's mypy silently sees `Any`.
  5. Packaged data read through `importlib.resources`, and asserted present
     in the wheel. Prompts and schemas are the most commonly-missing files in
     an LLM package.
  6. `_scratch/` and `conftest.py` explicitly excluded, and asserted absent.
  7. Entry points for the provider group, so a plugin can ship separately —
     with each `load()` isolated and shape-checked.
  8. The console script returns an exit code and takes no required arguments.
  9. The sdist REBUILDS to the same wheel.
 10. A changelog entry for the PROMPT change, with fingerprints, as a MINOR
     bump — because the API is unchanged and the outputs are not.
""")


if __name__ == "__main__":
    main()
