"""
06 — Versioning, releases, and private feeds.

TWO SEPARATE PROBLEMS
---------------------
    MECHANICS   where the version number lives, and how it gets into the
                wheel without being written down three times
    POLICY      what a version number PROMISES, and what you owe users when
                you break it

The second is the one that matters and the one usually skipped. A version
number is a compatibility claim; if it does not mean anything, your users
pin exact versions forever and you can never ship anything.

Run:  python 06_versioning.py
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
# PART 1 — one source of truth
# ---------------------------------------------------------------------------

STATIC_PYPROJECT = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
version = "0.3.0"
requires-python = ">=3.11"

[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
'''

DYNAMIC_PYPROJECT = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
requires-python = ">=3.11"
# The version is not written here at all.
dynamic = ["version"]

[tool.hatch.version]
# ONE source of truth: the package's own __init__.py.
path = "src/isc_rag/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/isc_rag"]
'''

RUNTIME_VERSION = '''\
"""Reading the version AT RUNTIME.

THE PATTERN WORTH ADOPTING — do not hardcode it here either:

    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("isc-rag")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"    # running from a source checkout

This reads the INSTALLED distribution's metadata, so it can never disagree
with what pip thinks is installed — which is the failure this whole section
is about.

The `try/except` matters: running from a source tree with no install raises
PackageNotFoundError, and a hard failure there breaks `python -m` invocations
and some test setups.

NOTE the argument is the DISTRIBUTION name ("isc-rag"), not the import name
("isc_rag"). They differ here, as they often do.
"""
'''


def part1_single_source() -> None:
    banner("PART 1 — the version must live in exactly one place")

    print("""
    THE FAILURE: a version written in pyproject.toml, in __init__.py, and in
    a VERSION file. Someone bumps two of them. Now `pip show` disagrees with
    `pkg.__version__`, and a bug report says "0.3.1" while the installed
    wheel is 0.3.0. You cannot reproduce anything.

    THREE WORKING APPROACHES:""")

    section("A) static in pyproject, read from metadata at runtime")
    code_block(STATIC_PYPROJECT + "\n# and in __init__.py:\n"
               + RUNTIME_VERSION.split('"""')[1][:0] + '''
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("isc-rag")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"''')
    print("""      SIMPLEST, and my default recommendation. The version is in the
      file a human edits at release time, and the runtime value is read from
      the installed metadata so it cannot drift.""")

    section("B) dynamic — the backend reads it from your source")
    code_block(DYNAMIC_PYPROJECT)
    print("""      The version lives in __init__.py and hatchling extracts it at
      build time. Good when you want `__version__` to work in a source
      checkout with no install.""")

    section("C) derived from the git tag (hatch-vcs / setuptools-scm)")
    code_block('''\
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "isc-rag"
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"''')
    print("""      THE TAG IS THE VERSION. `git tag v0.3.0` and the build produces
      0.3.0; an untagged commit produces something like
      `0.3.1.dev4+g1a2b3c4`, which is genuinely useful — you can tell exactly
      which commit an artifact came from.

      THE COSTS, and they are real:
        * the build needs a git checkout WITH TAGS. A shallow CI clone
          (`fetch-depth: 1`, the GitHub Actions default) has no tags, and
          you get `0.1.dev1+unknown` silently. Set `fetch-depth: 0`.
        * building from an extracted sdist has no git at all, so the backend
          must write the version into the sdist. Both tools do; verify it.

      Worth it for a package with frequent releases. Overkill for one that
      ships quarterly.""")


# ---------------------------------------------------------------------------
# PART 2 — demonstrate the dynamic version
# ---------------------------------------------------------------------------

def part2_dynamic_demo() -> None:
    banner("PART 2 — dynamic version, built and read back")

    b = ProjectBuilder.in_temp("dynver")
    b.add("pyproject.toml", DYNAMIC_PYPROJECT)
    b.add("src/isc_rag/__init__.py",
          '__version__ = "0.4.2"\n')
    b.write()

    result = build_project(b.root, sdist=False)
    if result.wheel:
        w = WheelInspector(result.wheel)
        show("version in __init__.py", "0.4.2")
        show("version in the wheel filename", w.filename_parts.get("version"))
        show("Version in METADATA", w.metadata_field("Version"))
        verdict(w.filename_parts.get("version") == "0.4.2",
                "one source of truth, no drift possible")

        venv = VenvRunner(b.root)
        venv.create()
        venv.install(str(result.wheel))
        out = venv.run_python(
            "from importlib.metadata import version;"
            "import isc_rag;"
            "print('metadata:', version('isc-rag'),"
            " '| attribute:', isc_rag.__version__)",
            cwd=Path("/tmp"))
        show("read back after install", out.output)
    else:
        show("build", f"FAILED: {result.error_summary}")
    b.cleanup()


# ---------------------------------------------------------------------------
# PART 3 — what a version number PROMISES
# ---------------------------------------------------------------------------

def part3_semver() -> None:
    banner("PART 3 — the policy half")

    print("""
    SEMVER, stated as an obligation rather than a format:

        MAJOR   I broke something you depend on.
        MINOR   I added something. Your code still works.
        PATCH   I fixed something. Your code still works.

    THE HARD QUESTION IS "WHAT COUNTS AS BREAKING", and you have to answer it
    explicitly or every change becomes an argument. For `isc-rag`:

      BREAKING (major):
        * removing or renaming a public function, class, or parameter
        * changing a default that alters behaviour (top_k 4 -> 8)
        * narrowing an accepted input type
        * changing an exception type a caller might catch
        * removing an ENTRY POINT or an EXTRA — someone's install command
          breaks
        * raising `requires-python`

      NOT BREAKING (minor):
        * adding a function, class, or KEYWORD-ONLY parameter with a default
        * widening an accepted type
        * adding an extra or an entry point
        * relaxing a dependency constraint

      AMBIGUOUS, and you must decide and write it down:
        * TIGHTENING a dependency upper bound. It can make a previously
          working install unsatisfiable, so I treat it as breaking.
        * changing LOG or telemetry field names. Breaking if anyone alerts
          on them — and someone does.
        * changing PROMPT TEXT in a shipped template. This is the
          GenAI-specific one and it is genuinely hard: the API is unchanged,
          the OUTPUTS change. Nothing in semver covers it.

    ON THAT LAST POINT, my position: if your package ships prompt templates,
    treat a prompt change as at least a MINOR version and say so in the
    changelog with the fingerprint. Users evaluating your outputs need to
    correlate a quality change with a version, and "we changed the prompt in
    a patch release" makes that impossible.

    0.x IS NOT AN EXCUSE. Semver says anything goes below 1.0, and in
    practice people depend on your 0.x anyway. Either commit to the
    discipline or ship 1.0.""")


# ---------------------------------------------------------------------------
# PART 4 — deprecation
# ---------------------------------------------------------------------------

def part4_deprecation() -> None:
    banner("PART 4 — deprecating without breaking")

    section("the mechanism")
    code_block('''\
import warnings

def retrieve(query: str, k: int | None = None, *,
             top_k: int | None = None) -> list[Chunk]:
    """Retrieve chunks.

    .. deprecated:: 0.4.0
       `k` is deprecated in favour of `top_k`; it will be removed in 1.0.
    """
    if k is not None:
        warnings.warn(
            "retrieve(k=...) is deprecated and will be removed in isc-rag "
            "1.0; use retrieve(top_k=...) instead.",
            DeprecationWarning,
            stacklevel=2,          # <- point at the CALLER, not at this line
        )
        if top_k is not None:
            raise TypeError("pass either k or top_k, not both")
        top_k = k
    ...''')

    print("""
    THREE DETAILS THAT MATTER:

    1. `stacklevel=2`. Without it the warning points at YOUR file, so the
       user sees a warning about a line they cannot edit. With it, it points
       at their call site. This is the single most common mistake in
       deprecation warnings.

    2. `DeprecationWarning` is HIDDEN BY DEFAULT outside __main__. Your users
       will not see it unless they run with `-W error::DeprecationWarning`
       or pytest (which shows them). That is by design — it is aimed at
       developers, not end users — but it means a deprecation warning alone
       is not sufficient notice. Put it in the CHANGELOG too.

    3. NAME THE REMOVAL VERSION. "will be removed in a future release" is
       not actionable. "removed in 1.0" lets someone plan.

    THE POLICY, written down:
       deprecate in 0.4.0  ->  warn for at least two MINOR releases
                           ->  remove in 1.0.0, listed in the changelog

    AND TEST IT, which almost nobody does:

        def test_k_is_deprecated_but_still_works():
            with pytest.warns(DeprecationWarning, match="use retrieve"):
                result = retrieve("q", k=4)
            assert len(result) <= 4

    That test fails when you remove the parameter, which is the reminder to
    also remove the test and note it in the changelog.""")


# ---------------------------------------------------------------------------
# PART 5 — private feeds
# ---------------------------------------------------------------------------

def part5_private_feeds() -> None:
    banner("PART 5 — private indexes")

    print("""
    An internal package is not on PyPI. It lives in Azure Artifacts, an
    internal devpi, CodeArtifact, or a Nexus.

    CONFIGURATION, in decreasing order of preference:

      1. pyproject / uv, committed
             [[tool.uv.index]]
             name = "isc-internal"
             url = "https://pkgs.dev.azure.com/isc/_packaging/isc-py/pypi/simple/"

      2. pip.conf / PIP_INDEX_URL, per environment
             [global]
             index-url = https://pkgs.dev.azure.com/.../simple/
             extra-index-url = https://pypi.org/simple/

      3. On the command line. Fine for a one-off, terrible in a Dockerfile
         where it becomes an undocumented requirement.

    AUTHENTICATION — never a PAT in a URL:
      * Azure Artifacts: `artifacts-keyring`, which uses your Azure login.
        In CI, a federated/workload identity rather than a stored token.
      * A PAT in `index-url` ends up in pip's log output, in your shell
        history, and in the Docker build cache. It is the URL-credential
        problem from the config tutorial, in a new setting.

    ─────────────────────────────────────────────────────────────────────
    THE DEPENDENCY-CONFUSION ATTACK, which is the important part
    ─────────────────────────────────────────────────────────────────────

    `--extra-index-url` does NOT mean "look here second". pip queries ALL
    configured indexes and picks the HIGHEST VERSION it finds anywhere.

    So if your internal `isc-rag` is at 0.3.0 and someone uploads a package
    named `isc-rag` version 99.0.0 to public PyPI, pip installs THEIRS. On
    every machine, in every build, silently. This has been used successfully
    against large companies.

    THE MITIGATIONS, and you want more than one:

      a) `--index-url` pointing at an internal index that PROXIES PyPI, so
         there is exactly ONE index and your feed's own package always wins.
         This is the right answer and what Azure Artifacts upstream sources
         are for.

      b) `--require-hashes` with a lockfile. A substituted package fails the
         hash check. This is the strongest control and it is what script 03
         part 6 argued for.

      c) Reserve your names on public PyPI. Cheap, and it removes the attack
         for those specific names.

      d) NEVER use a bare `--extra-index-url` for an internal feed in a
         production build. It is the configuration the attack depends on.

    THE VERSION-NUMBER TELL: an internal package that suddenly resolves to a
    version far ahead of anything you released is the signature. Pin your
    internal packages in the lock and this cannot happen silently.""")

    section("a Dockerfile line that is safe")
    code_block('''\
# ONE index, which proxies PyPI upstream. Hashes enforced. No PAT in the URL —
# artifacts-keyring picks up the workload identity.
RUN --mount=type=cache,target=/root/.cache/pip \\
    pip install --require-hashes --no-deps \\
        --index-url https://pkgs.dev.azure.com/isc/_packaging/isc-py/pypi/simple/ \\
        -r requirements.lock.txt''')


def main() -> None:
    part1_single_source()
    part2_dynamic_demo()
    part3_semver()
    part4_deprecation()
    part5_private_feeds()

    banner("SUMMARY")
    print("""
  * ONE source of truth for the version. Read it at runtime with
    `importlib.metadata.version("dist-name")`, not a hardcoded string.
  * git-tag versioning needs `fetch-depth: 0` or you silently get
    `0.1.dev1+unknown`.
  * Write down what counts as BREAKING, including the ambiguous cases:
    tightened upper bounds, renamed log fields, and changed prompt text.
  * A shipped prompt change is at least a MINOR bump, with the fingerprint in
    the changelog — users correlating quality to version need it.
  * `stacklevel=2` on every deprecation warning, and name the removal
    version. DeprecationWarning is hidden by default, so the changelog is the
    real notice.
  * `--extra-index-url` picks the HIGHEST version across ALL indexes. That is
    the dependency-confusion attack. Use one proxying index plus hashes.
""")


if __name__ == "__main__":
    main()
