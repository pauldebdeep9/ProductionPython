"""
04 — Entry points: console scripts and plugin discovery.

WHAT AN ENTRY POINT IS
----------------------
A line of metadata in the installed distribution saying "the name X maps to
the Python object Y". Two uses, and they are quite different:

    [project.scripts]           -> a COMMAND on PATH
    [project.entry-points.*]    -> a PLUGIN another package can discover

The second is the more interesting one and the more under-used. It is how a
package registers a capability with a package it does not import and that
does not import it — the packaging-level version of the Protocol pattern.

For a model-provider registry that is exactly the shape you want: `isc-rag`
defines the interface, and `isc-rag-bedrock` (a separate package, a separate
repo, a separate team) registers an implementation without either side
importing the other.

Run:  python 04_entry_points.py
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
# The core package: defines the Protocol and a discovery function.
# ---------------------------------------------------------------------------

CORE_PYPROJECT = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.3.0"
requires-python = ">=3.11"

# A CONSOLE SCRIPT. Installs an executable named `isc-rag` onto PATH that
# calls isc_rag.cli:main.
[project.scripts]
isc-rag = "isc_rag.cli:main"

# The core package registers its OWN provider under the same group that
# third-party packages will use. That is deliberate — it means the built-in
# provider goes through exactly the same code path as an external one, so the
# plugin mechanism is exercised on every run rather than only when a plugin
# happens to be installed.
[project.entry-points."isc_rag.providers"]
azure = "isc_rag.providers.azure:AzureProvider"

[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
'''

CORE_REGISTRY = '''\
"""Provider discovery via entry points."""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

GROUP = "isc_rag.providers"


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, prompt: str) -> str: ...


def discover() -> dict[str, type]:
    """Find every registered provider.

    `entry_points(group=...)` is the modern API (Python 3.10+). The older
    `entry_points()["group"]` form is deprecated and behaves differently
    across versions — if you see it, it is pre-3.10 code.

    NOTE WHAT THIS DOES NOT DO: it does not import the providers. `ep.load()`
    is a separate, explicit step, so a broken plugin cannot break discovery
    for everyone else. See `load_all` below.
    """
    return {ep.name: ep for ep in entry_points(group=GROUP)}


def load_all() -> tuple[dict[str, type], dict[str, str]]:
    """Load every provider, isolating failures.

    THE IMPORTANT PART is the try/except around each `ep.load()`. A plugin
    from another team, another repo, another release cadence WILL eventually
    fail to import — a missing transitive dependency, an incompatible
    version. Without isolation, one bad plugin takes down the whole host
    application at startup with a traceback pointing at code you do not own.

    With it, you get a degraded feature and a clear log line.
    """
    loaded: dict[str, type] = {}
    failed: dict[str, str] = {}
    for name, ep in discover().items():
        try:
            cls = ep.load()
        except Exception as exc:                       # noqa: BLE001
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        # VALIDATE THE CONTRACT. An entry point can point at anything at all —
        # a string, a function, a class with the wrong shape. Check before
        # trusting it, or the failure surfaces much later and much further
        # from the cause.
        if not (isinstance(cls, type) and hasattr(cls, "name")):
            failed[name] = f"does not look like a Provider: {cls!r}"
            continue
        loaded[name] = cls
    return loaded, failed
'''

CORE_AZURE = '''\
class AzureProvider:
    name = "azure-openai"

    def complete(self, prompt: str) -> str:
        return f"[azure] {prompt[:24]}"
'''

CORE_CLI = '''\
"""The console-script target.

CONVENTIONS THAT MATTER:
  * `main()` takes NO required arguments. The generated wrapper calls it with
    none, so a signature requiring one is a TypeError at first invocation.
  * It RETURNS an exit code, and the wrapper passes it to sys.exit(). A
    function returning None exits 0 regardless of what happened — which is
    how a failing CLI command ends up green in CI.
  * Keep it THIN. Argument parsing here, logic in an importable module, so
    the logic is testable without a subprocess.
"""
from __future__ import annotations

import sys

from isc_rag.registry import load_all


def main() -> int:
    loaded, failed = load_all()
    print(f"isc-rag 0.3.0")
    print(f"providers: {sorted(loaded)}")
    if failed:
        for name, why in failed.items():
            print(f"  FAILED {name}: {why}", file=sys.stderr)
        return 1
    return 0
'''

# ---------------------------------------------------------------------------
# A third-party plugin package — no import relationship with the core.
# ---------------------------------------------------------------------------

PLUGIN_PYPROJECT = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag-bedrock"
version = "0.1.0"
requires-python = ">=3.11"
# NOTE: it does NOT declare isc-rag as a dependency in this demo, to make the
# decoupling obvious. A REAL plugin generally should:
#     dependencies = ["isc-rag>=0.3,<1.0"]
# ...because it depends on the Protocol's shape, and the version range is
# where you express which shape you were written against.

[project.entry-points."isc_rag.providers"]
bedrock = "isc_rag_bedrock:BedrockProvider"

[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag_bedrock"]
'''

PLUGIN_CODE = '''\
class BedrockProvider:
    name = "aws-bedrock"

    def complete(self, prompt: str) -> str:
        return f"[bedrock] {prompt[:24]}"
'''

# A deliberately broken plugin, to demonstrate failure isolation.
BROKEN_PLUGIN_PYPROJECT = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag-broken"
version = "0.1.0"
requires-python = ">=3.11"

[project.entry-points."isc_rag.providers"]
broken = "isc_rag_broken:MissingClass"

[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag_broken"]
'''

BROKEN_PLUGIN_CODE = '''\
# The entry point points at `MissingClass`, which does not exist here.
# A realistic version of this is a plugin whose import fails because a
# transitive dependency is missing.
class SomethingElse:
    pass
'''


def build_core() -> ProjectBuilder:
    b = ProjectBuilder.in_temp("core")
    b.add("pyproject.toml", CORE_PYPROJECT)
    b.add("src/isc_rag/__init__.py", '__version__ = "0.3.0"\n')
    b.add("src/isc_rag/registry.py", CORE_REGISTRY)
    b.add("src/isc_rag/cli.py", CORE_CLI)
    b.add("src/isc_rag/providers/__init__.py", "")
    b.add("src/isc_rag/providers/azure.py", CORE_AZURE)
    b.write()
    return b


def build_plugin(name: str, pyproject: str, module: str, code: str) -> ProjectBuilder:
    b = ProjectBuilder.in_temp(name)
    b.add("pyproject.toml", pyproject)
    b.add(f"src/{module}/__init__.py", code)
    b.write()
    return b


# ---------------------------------------------------------------------------
# PART 1 — console scripts
# ---------------------------------------------------------------------------

def part1_console_scripts() -> None:
    banner("PART 1 — console scripts")

    core = build_core()
    built = build_project(core.root, sdist=False)
    if not built.ok or built.wheel is None:
        show("build", f"FAILED: {built.error_summary}")
        core.cleanup()
        return

    wheel = WheelInspector(built.wheel)
    section("what the wheel records")
    code_block(wheel.entry_points)

    venv = VenvRunner(core.root)
    venv.create()
    venv.install(str(built.wheel))

    script = venv.path / "bin" / "isc-rag"
    section("after install")
    show("executable created?", script.exists())
    if script.exists():
        run = venv.run_console_script("isc-rag")
        show("running it", run.output.splitlines()[0] if run.output else "")
        show("exit code ok", run.ok)

    section("the generated wrapper")
    if script.exists():
        code_block(script.read_text(), limit=12)

    print("""
    THAT WRAPPER IS THE WHOLE MECHANISM. A tiny generated script with the
    venv's interpreter in the shebang, importing your module and calling your
    function. Nothing magic.

    Two consequences worth knowing:
      * The shebang is an ABSOLUTE PATH to the venv's python. Move or rename
        the venv and every console script breaks. That is why you recreate a
        venv rather than move it.
      * `sys.exit(main())` means your return value IS the exit code. Return
        None and you always exit 0.""")

    core.cleanup()


# ---------------------------------------------------------------------------
# PART 2 — plugin discovery
# ---------------------------------------------------------------------------

DISCOVER_CODE = (
    "from isc_rag.registry import load_all;"
    "loaded, failed = load_all();"
    "print('LOADED', sorted(loaded));"
    "print('FAILED', {k: v[:60] for k, v in failed.items()})"
)


def part2_plugins() -> None:
    banner("PART 2 — a plugin registers itself with no import relationship")

    core = build_core()
    core_wheel = build_project(core.root, sdist=False)
    plugin = build_plugin("bedrock", PLUGIN_PYPROJECT, "isc_rag_bedrock",
                          PLUGIN_CODE)
    plugin_wheel = build_project(plugin.root, sdist=False)

    if not (core_wheel.wheel and plugin_wheel.wheel):
        show("build", "FAILED")
        core.cleanup()
        plugin.cleanup()
        return

    venv = VenvRunner(core.root)
    venv.create()
    venv.install(str(core_wheel.wheel))

    section("core only")
    out = venv.run_python(DISCOVER_CODE, cwd=Path("/tmp"))
    code_block(out.output)

    section("after installing the plugin package")
    venv.install(str(plugin_wheel.wheel))
    out = venv.run_python(DISCOVER_CODE, cwd=Path("/tmp"))
    code_block(out.output)

    print("""
    THE CORE PACKAGE DID NOT CHANGE. No config file was edited, no import was
    added, no registry was updated. Installing a wheel made a new provider
    available.

    WHY THAT IS WORTH THE COMPLEXITY, specifically:
      * a plugin can live in a DIFFERENT REPO with a different release
        cadence and a different owner
      * the core has no dependency on the plugin, so it does not need to know
        the plugin exists to be released
      * `pip uninstall isc-rag-bedrock` cleanly removes the capability

    WHEN NOT TO USE IT: if all the implementations live in YOUR repo and ship
    in YOUR wheel, a plain dict is simpler, faster, statically analysable, and
    does not require an install step to change. Entry points are for crossing
    a DISTRIBUTION boundary. Inside one, they are indirection with no payoff.""")

    core.cleanup()
    plugin.cleanup()


# ---------------------------------------------------------------------------
# PART 3 — failure isolation
# ---------------------------------------------------------------------------

def part3_broken_plugin() -> None:
    banner("PART 3 — one broken plugin must not take down the host")

    core = build_core()
    core_wheel = build_project(core.root, sdist=False)
    good = build_plugin("bedrock", PLUGIN_PYPROJECT, "isc_rag_bedrock",
                        PLUGIN_CODE)
    good_wheel = build_project(good.root, sdist=False)
    bad = build_plugin("broken", BROKEN_PLUGIN_PYPROJECT, "isc_rag_broken",
                       BROKEN_PLUGIN_CODE)
    bad_wheel = build_project(bad.root, sdist=False)

    if not all([core_wheel.wheel, good_wheel.wheel, bad_wheel.wheel]):
        show("build", "FAILED")
        for p in (core, good, bad):
            p.cleanup()
        return

    venv = VenvRunner(core.root)
    venv.create()
    venv.install(str(core_wheel.wheel), str(good_wheel.wheel),
                 str(bad_wheel.wheel))

    section("discovery with a broken plugin installed")
    out = venv.run_python(DISCOVER_CODE, cwd=Path("/tmp"))
    code_block(out.output)

    section("the CLI reports it and exits non-zero")
    run = venv.run_console_script("isc-rag")
    code_block((run.stdout + run.stderr).strip())
    show("exit code non-zero", not run.ok)

    print("""
    The two working providers loaded. The broken one is reported by name,
    with the reason, and the process is still running.

    WITHOUT THE try/except IN `load_all`, an AttributeError from a package
    you did not write would propagate out of your startup path, and the
    traceback would point at `importlib.metadata` — which tells the operator
    nothing about which plugin to remove.

    THE GENERAL RULE: any code loaded from outside your distribution is
    UNTRUSTED INPUT, in the same sense as a JSON payload. Isolate the load,
    validate the shape, and report failures by name.""")

    for p in (core, good, bad):
        p.cleanup()


# ---------------------------------------------------------------------------
# PART 4 — the object reference syntax
# ---------------------------------------------------------------------------

def part4_syntax() -> None:
    banner("PART 4 — the `module:object` syntax, precisely")

    print("""
        name = "package.module:object.attribute [extras]"
                └─ import ──┘ └─ getattr chain ─┘ └─ optional ─┘

    EXAMPLES:
        cli = "isc_rag.cli:main"                  a function
        azure = "isc_rag.providers:AzureProvider"  a class
        v2 = "isc_rag.providers:Registry.v2"       a nested attribute
        heavy = "isc_rag.local:Provider [local]"   requires the `local` extra

    THE EXTRAS SUFFIX is genuinely useful and almost unknown: it declares
    that this entry point needs an extra to work. Tools can then warn rather
    than letting the import fail with a confusing ModuleNotFoundError.

    THINGS THAT GO WRONG, in order of frequency:

    1. A TYPO IN THE PATH. It is a STRING; nothing validates it at build
       time. `isc_rag.providers.azure:AzureProvider` with the class renamed
       builds a perfectly good wheel that fails at load. TEST YOUR ENTRY
       POINTS — script 09 does exactly this.

    2. IMPORTING THE WORLD. The module named here is imported during
       discovery, so a module that pulls in torch at import time makes every
       `load()` slow. Keep the entry-point module thin and do heavy imports
       inside the class.

    3. FORGETTING THE GROUP IS A NAMESPACE YOU DO NOT OWN. `console_scripts`
       is global — if two installed packages both declare `isc-rag`, one
       silently wins. Prefix your own groups with your package name
       (`isc_rag.providers`), and give console scripts a distinctive name.

    4. ASSUMING ORDER. `entry_points()` returns them in no guaranteed order,
       and two packages can register the same NAME in the same group. If
       ordering or precedence matters, add an explicit priority attribute to
       the loaded object and sort on it — do not rely on discovery order.""")


# ---------------------------------------------------------------------------
# PART 5 — performance
# ---------------------------------------------------------------------------

def part5_performance() -> None:
    banner("PART 5 — discovery cost, and when it matters")

    core = build_core()
    core_wheel = build_project(core.root, sdist=False)
    if core_wheel.wheel:
        venv = VenvRunner(core.root)
        venv.create()
        venv.install(str(core_wheel.wheel))
        timing = venv.run_python(
            "import time;"
            "t0=time.perf_counter();"
            "from importlib.metadata import entry_points;"
            "eps=list(entry_points(group='isc_rag.providers'));"
            "t1=time.perf_counter();"
            "print(f'{len(eps)} entry point(s) discovered in "
            "{(t1-t0)*1000:.1f}ms')",
            cwd=Path("/tmp"))
        section("measured in a small venv")
        code_block(timing.output)
    core.cleanup()

    print("""
    Fast here because the venv contains three packages. In a real environment
    with 200 installed distributions, `entry_points()` scans every
    `.dist-info` directory, and the cost is measured in tens of milliseconds.

    WHERE THAT MATTERS: a CLI, where it is a visible fraction of startup, and
    a serverless cold start. Where it does not: a long-running service, which
    pays it once.

    THE MITIGATIONS, in order:
      * call `entry_points()` ONCE and cache the result. `functools.cache` on
        your `discover()` is usually enough.
      * do it at STARTUP, not per request.
      * do NOT `load()` everything eagerly — discovery is cheap, importing is
        not. Load a provider when it is first used.

    The last point is the one that actually matters for an ML package. A
    provider whose module imports torch costs seconds, not milliseconds, and
    eagerly loading all providers means paying for every backend you did not
    use.""")


def main() -> None:
    part1_console_scripts()
    part2_plugins()
    part3_broken_plugin()
    part4_syntax()
    part5_performance()

    banner("SUMMARY")
    print("""
  * `[project.scripts]` generates a wrapper with an ABSOLUTE shebang into the
    venv. Moving a venv breaks them.
  * `main()` takes no required args and RETURNS an exit code.
  * `[project.entry-points."group"]` lets a separate distribution register a
    capability with no import relationship in either direction.
  * Use it to cross a DISTRIBUTION boundary. Inside one repo, a dict is
    better.
  * Isolate every `ep.load()` and validate the loaded object — a plugin is
    untrusted input.
  * The reference is an unvalidated STRING. Test that your entry points
    actually resolve.
  * Discovery is cheap; loading is not. Cache discovery, load lazily.
""")


if __name__ == "__main__":
    main()
