# Packaging — Production Python, GenAI Focus

Eight files, ~3,600 lines. Runs offline; needs `build`, `hatchling`, and
optionally `uv`.

**Verified state:** 7/7 scripts run clean · every claim is demonstrated by
actually building wheels, installing them into throwaway venvs, and inspecting
the artifacts · the capstone runs a six-gate release pipeline end to end,
including an sdist rebuild round-trip · no leftover temp directories.

---

## Files

| File | Topic | The thing worth taking away |
|---|---|---|
| `pkg_lab.py` | Harness | Builds real wheels, installs into real venvs, unzips and inspects |
| `01_layout.py` | src vs flat | The failure **demonstrated**, not asserted |
| `02_pyproject.py` | PEP 621 | Extras are just requirements with an `extra ==` marker |
| `03_dependencies.py` | Constraints, locks | Libraries get ranges; applications get lockfiles |
| `04_entry_points.py` | Scripts and plugins | A plugin registers with **no import relationship** |
| `05_building.py` | Wheels, sdists, data files | Prompts and schemas are the most commonly-missing files |
| `06_versioning.py` | Versions, deprecation, feeds | `--extra-index-url` is the dependency-confusion attack |
| `07_capstone.py` | Everything, end to end | A six-gate release pipeline |

Run any file: `python 01_layout.py`. Each builds into a temp dir and cleans up.

---

## Suggested path

**Two hours** — 01, 03, 05. The layout argument, the library/application
split, and what actually ends up in your wheel.

**If you read one file, read 01.** Everyone has heard "use src layout."
Almost nobody has seen the failure it prevents.

**Shipping an internal package** — 06 part 5 on private feeds and dependency
confusion, then 07's gate.

---

## The demonstration that carries the tutorial

Script 01 builds the **same deliberately-broken wheel** from both layouts —
one missing its `providers` subpackage — installs each, and runs the same
import twice:

```
--- FLAT layout, broken wheel ---
  wheel ships the providers subpackage?   False
  import run FROM THE PROJECT ROOT        OK 0.1.0 azure-openai      ← green!
  import run from elsewhere               FAILED: Traceback ...

--- SRC layout, broken wheel ---
  wheel ships the providers subpackage?   False
  import run FROM THE PROJECT ROOT        FAILED: Traceback ...      ← correct
  import run from elsewhere               FAILED: Traceback ...
```

The flat layout reports success against a genuinely broken artifact, because
the cwd is on `sys.path` and the source tree is sitting right there. A test
suite run from the project root would be green.

That is the entire argument for `src/`. Not tidiness — it removes the
possibility of testing something other than what you ship.

**The honest caveat, in part 4:** an editable install weakens the guarantee,
because `isc_rag` becomes importable from anywhere regardless of wheel
contents. The real control is the CI gate: build the wheel, install into a
clean venv, run tests from a directory that is not the repo.

---

## A piece of folklore I had to retract

I wrote that `python_version >= '3.9'` is FALSE on 3.10 because "3.10" sorts
before "3.9" lexically. Then I checked:

```python
>>> Marker("python_version >= '3.9'").evaluate({"python_version": "3.10"})
True
>>> "3.10" >= "3.9"          # raw Python string comparison
False
```

PEP 508 specifies a **version** comparison when both sides are valid PEP 440
versions. The trap was real in the setuptools era and has been repeated ever
since. String comparison still applies to `in`/`not in` and to non-version
values like `platform_release` on a kernel string.

The durable lesson isn't the rule — it's that packaging folklore has a long
half-life, and `packaging.markers` settles it in ten seconds.

Two smaller corrections: a TOML table may be declared only **once**, so
appending a second `[tool.hatch.build.targets.wheel]` is a build failure
rather than a merge (my script 05 hit this and silently produced no output
until I made build failures visible). And hatchling **flattens** a
self-referencing `all` extra into its constituents at build time — which is
backend-specific, so read the METADATA rather than assuming.

---

## Measured results

**Entry-point plugins work with no import relationship** (`04`). Installing a
separate wheel changed `LOADED ['azure']` to `LOADED ['azure', 'bedrock']`
with no change to the core package.

**Failure isolation works** — a plugin whose entry point points at a
nonexistent class is reported by name while the other two load:
`FAILED {'broken': "AttributeError: module 'isc_rag_broken' has no attribute 'MissingClass'"}`

**Wheels ship too little by default; sdists ship too much** (`05`). The wheel
correctly omitted `tests/` and `README.md`; the sdist shipped `notes.md`,
labelled "should NOT ship", because hatchling's sdist default is
everything-not-gitignored.

**Exclusions verified** — `_scratch/local.py` containing
`KEY = "sk-LOCAL-DO-NOT-SHIP"` was kept out of the wheel while prompts still
shipped.

**The capstone's six gates all pass**, including the sdist round-trip: unpack
the tarball, rebuild a wheel from it, and confirm identical file lists.

---

## Decision tables

### Constraint style

| You are writing | Declare | Why |
|---|---|---|
| A **library** | `>=X.Y,<NEXT_MAJOR` | It must compose with other libraries |
| An **application** | A committed lockfile | Nothing depends on it |

`~=0.27.2` allows only patch; `~=0.27` allows minor. One dot, two very
different promises — which is why the explicit range reads better.

The argument *against* upper bounds is real: `<3.0` blocks every downstream
app on your release cadence. Cap anyway, and treat raising it as routine
maintenance.

### Where the version lives

| Approach | Use when |
|---|---|
| Static in `pyproject`, read via `importlib.metadata` | **Default** |
| `dynamic`, backend reads `_version.py` | You want `__version__` in a source checkout |
| From the git tag (`hatch-vcs`) | Frequent releases — needs `fetch-depth: 0` |

### Entry points or a dict

Entry points cross a **distribution** boundary — a plugin in another repo,
another team, another release cadence. Inside one repo, a dict is simpler,
faster, statically analysable, and doesn't need an install step to change.

---

## Review checklist

**Layout**
- [ ] `src/` layout
- [ ] Tests **outside** the package
- [ ] `py.typed` present **and asserted to be in the wheel**
- [ ] `_internal/` for non-public modules

**Metadata**
- [ ] `requires-python` set to what you **test**, not what you hope
- [ ] Ranges in a library, never pins
- [ ] Extras for optional *functionality*; dev deps in `[dependency-groups]`
- [ ] Optional imports guarded with an error naming the install command
- [ ] `Private :: Do Not Upload` on internal packages

**Artifacts**
- [ ] Data files (prompts, schemas) verified present in the wheel
- [ ] `importlib.resources`, never `Path(__file__).parent`
- [ ] Data directories have `__init__.py`
- [ ] `_scratch/`, `conftest.py`, notebooks, fixtures excluded
- [ ] Build **with isolation** in CI — that proves `build-system.requires`

**Versioning**
- [ ] One source of truth
- [ ] Written-down definition of "breaking", including the ambiguous cases
- [ ] `stacklevel=2` on deprecation warnings; removal version named
- [ ] Upload **then** tag

**Private feeds**
- [ ] One index that **proxies** PyPI, not `--extra-index-url`
- [ ] `--require-hashes` against a lockfile
- [ ] No PAT in a URL — `artifacts-keyring` or workload identity
- [ ] Internal names reserved on public PyPI

---

## Dependency confusion, because it's the security item

`--extra-index-url` does **not** mean "look here second." pip queries *all*
configured indexes and picks the **highest version found anywhere**. If your
internal `isc-rag` is 0.3.0 and someone uploads `isc-rag` 99.0.0 to public
PyPI, pip installs theirs — silently, on every machine, in every build. This
has been used successfully against large companies.

Mitigations, and you want more than one: a single internal index that proxies
PyPI upstream; `--require-hashes` with a lockfile; reserve your names on
public PyPI; never a bare `--extra-index-url` in a production build.

---

## The GenAI-specific parts

**Prompts and schemas are data files, and data files are not included by
default in every backend.** "It works in development but the installed
package can't find my prompt templates" is the most common packaging bug in
an LLM codebase, and it's invisible until someone installs the wheel —
because `Path(__file__).parent` reads the source tree during development.

**A shipped prompt change is at least a MINOR bump.** The API is unchanged;
the outputs are not. Nothing in semver covers this, so you have to decide and
write it down. The capstone's changelog does it with the fingerprint:

```
### Changed
- **PROMPT** `disposition` template updated (fingerprint 36206a02 -> 9b41ce70).
  Output wording changes; re-run your evaluations.
```

Users correlating a quality change to a version need that, and a patch-release
prompt change makes it impossible.

**Discovery is cheap; loading is not.** A provider whose module imports torch
costs seconds. Cache discovery, load lazily.

---

## Connections to your existing work

**`isc-docint`'s prompt templates and golden sets are exactly the data-file
trap.** If they live under the package and you ever install it as a wheel,
verify they ship — script 07's gate 2 is the assertion. If they live outside
the package, that is also fine, but then the loading path needs to be
explicit config rather than a relative path that happens to work.

**The provider Protocol from the typing tutorial has a packaging-level twin.**
Entry points are how a `isc-rag-bedrock` package registers an implementation
without either side importing the other — the same dependency inversion, one
layer up. Worth knowing before you need it; not worth adopting while every
provider ships in your wheel.

**Your prompt fingerprints belong in the changelog.** The observability
tutorial argued for logging `prompt.fingerprint` per request; this one argues
for recording it per release. Together they let you answer "which prompt
version produced this answer, and what release introduced it" — currently two
separate archaeology projects.

**Retrofit order:** `src/` layout and the CI gate first (they cost an
afternoon and prevent a whole class of defect), then `py.typed` with the
wheel assertion, then a lockfile with hashes for whatever deploys. The entry
points only matter once a provider ships separately.

---

That's seven topics: async concurrency, failure handling, typing,
configuration and secrets, observability, pytest, and packaging — the
complete Tier 1 list.
