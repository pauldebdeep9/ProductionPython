"""
pkg_lab.py — harness for the packaging tutorial.

WHY A HARNESS
-------------
Packaging is usually taught as prose about files you are told to trust. That
does not stick, because the interesting parts are all FAILURE MODES that only
appear when you actually build and install something:

  * a flat layout that tests the source tree instead of the installed package
  * a wheel missing a data file that worked fine in development
  * an entry point that resolves in the repo and not after install
  * a dependency constraint that is satisfiable today and not next month

So this module builds REAL packages into temp directories, builds REAL wheels
with `python -m build`, installs them into REAL virtualenvs, and inspects the
results. Every claim in the numbered scripts is demonstrated rather than
asserted.

WHAT IT PROVIDES
----------------
  ProjectBuilder   writes a project tree from a dict of files
  build_wheel      runs the real build backend, returns the artifact paths
  WheelInspector   opens the .whl (it is a zip) and reports its contents
  VenvRunner       creates an isolated venv, installs, and runs code in it
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ===========================================================================
# 1. BUILDING A PROJECT TREE
# ===========================================================================

@dataclass
class ProjectBuilder:
    """Writes a project tree from a {relative_path: contents} mapping.

    Kept deliberately dumb. The point is that the numbered scripts can show
    an ENTIRE project as a Python dict, so the layout under discussion is
    visible in one place rather than spread across a directory listing.
    """

    root: Path
    files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def in_temp(cls, name: str = "proj") -> "ProjectBuilder":
        root = Path(tempfile.mkdtemp(prefix=f"pkglab-{name}-"))
        return cls(root=root)

    def add(self, relpath: str, contents: str) -> "ProjectBuilder":
        self.files[relpath] = contents
        return self

    def write(self) -> Path:
        for rel, contents in self.files.items():
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
        return self.root

    def tree(self, max_depth: int = 4) -> str:
        """A readable listing, so the layout is visible in the output."""
        lines: list[str] = []
        base_depth = len(self.root.parts)
        for path in sorted(self.root.rglob("*")):
            if any(p in {"__pycache__", ".git", "build", ".venv"}
                   for p in path.parts):
                continue
            depth = len(path.parts) - base_depth
            if depth > max_depth:
                continue
            indent = "  " * (depth - 1)
            suffix = "/" if path.is_dir() else ""
            lines.append(f"{indent}{path.name}{suffix}")
        return "\n".join(lines)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# ===========================================================================
# 2. BUILDING ARTIFACTS
# ===========================================================================

@dataclass
class BuildResult:
    ok: bool
    wheel: Path | None
    sdist: Path | None
    stdout: str
    stderr: str

    @property
    def error_summary(self) -> str:
        """Build errors are verbose. Pull out the line that matters."""
        text = self.stderr or self.stdout
        for marker in ("error:", "ERROR", "Traceback", "ValueError",
                       "configuration error"):
            for line in text.splitlines():
                if marker in line:
                    return line.strip()[:160]
        return text.strip().splitlines()[-1][:160] if text.strip() else ""


def build_project(root: Path, *, sdist: bool = True,
                  timeout: int = 180) -> BuildResult:
    """Run the REAL build frontend against a real backend.

    `--no-isolation` is used deliberately: building in an isolated environment
    downloads the backend from PyPI every time, which this container cannot
    do and which would make the tutorial slow anyway. In CI you want
    isolation ON — it is what proves your `build-system.requires` is complete.
    That distinction is itself worth knowing, so it is stated here rather
    than hidden.
    """
    args = [sys.executable, "-m", "build", "--no-isolation", "--outdir",
            str(root / "dist")]
    if not sdist:
        args.append("--wheel")
    proc = subprocess.run(args, cwd=root, capture_output=True, text=True,
                          timeout=timeout)
    dist = root / "dist"
    wheels = sorted(dist.glob("*.whl")) if dist.exists() else []
    sdists = sorted(dist.glob("*.tar.gz")) if dist.exists() else []
    return BuildResult(
        ok=proc.returncode == 0,
        wheel=wheels[-1] if wheels else None,
        sdist=sdists[-1] if sdists else None,
        stdout=proc.stdout, stderr=proc.stderr,
    )


# ===========================================================================
# 3. INSPECTING A WHEEL
# ===========================================================================

class WheelInspector:
    """A wheel is a ZIP with a naming convention. Nothing more.

    Knowing that is genuinely useful: when a package "does not include my
    templates", you do not theorise — you unzip it and look. This class is
    that, with the metadata parsed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        with zipfile.ZipFile(path) as z:
            self.names = sorted(z.namelist())
            self._metadata = self._read(z, "METADATA")
            self._record = self._read(z, "RECORD")
            self._entry_points = self._read(z, "entry_points.txt")
            self._wheel = self._read(z, "WHEEL")

    @staticmethod
    def _read(z: zipfile.ZipFile, suffix: str) -> str:
        for name in z.namelist():
            if name.endswith(f".dist-info/{suffix}"):
                return z.read(name).decode("utf-8", errors="replace")
        return ""

    # -- filename ---------------------------------------------------------

    @property
    def filename_parts(self) -> dict[str, str]:
        """`name-version-python-abi-platform.whl` — every part is meaningful.

        `py3-none-any` is a PURE PYTHON wheel: any Python 3, no ABI, any
        platform. That is what you want for a library like this one.

        `cp312-cp312-manylinux_2_17_x86_64` is a COMPILED wheel, valid only
        for CPython 3.12 on that platform. Seeing that on a package you
        thought was pure Python means something is compiling.
        """
        stem = self.path.stem
        parts = stem.split("-")
        keys = ["name", "version", "python_tag", "abi_tag", "platform_tag"]
        if len(parts) == 6:                    # optional build tag
            keys = ["name", "version", "build", "python_tag", "abi_tag",
                    "platform_tag"]
        return dict(zip(keys, parts, strict=False))

    # -- metadata ---------------------------------------------------------

    def metadata_field(self, field: str) -> list[str]:
        out: list[str] = []
        for line in self._metadata.splitlines():
            if line.startswith(f"{field}:"):
                out.append(line.split(":", 1)[1].strip())
        return out

    @property
    def requires_dist(self) -> list[str]:
        return self.metadata_field("Requires-Dist")

    @property
    def entry_points(self) -> str:
        return self._entry_points

    @property
    def package_files(self) -> list[str]:
        """Everything except the .dist-info directory — i.e. what actually
        lands in site-packages."""
        return [n for n in self.names if ".dist-info/" not in n]

    def has(self, pattern: str) -> bool:
        return any(pattern in n for n in self.names)

    def summary(self) -> str:
        p = self.filename_parts
        return (f"{p.get('name')} {p.get('version')} "
                f"[{p.get('python_tag')}-{p.get('abi_tag')}-"
                f"{p.get('platform_tag')}]  "
                f"{len(self.package_files)} package files")


# ===========================================================================
# 4. ISOLATED INSTALL AND RUN
# ===========================================================================

@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout or self.stderr).strip()

    @property
    def first_error(self) -> str:
        for line in (self.stderr or "").splitlines():
            if any(k in line for k in ("Error", "error:", "Traceback")):
                return line.strip()[:140]
        return (self.stderr or "").strip().splitlines()[-1][:140] \
            if (self.stderr or "").strip() else ""


class VenvRunner:
    """A throwaway virtualenv, so an install is genuinely isolated.

    THE POINT: `pip install -e .` into the ambient interpreter tells you very
    little, because your source tree is on `sys.path` anyway. Installing into
    a clean venv is the only way to find out what your package ACTUALLY
    ships — which is exactly the src-layout argument in script 01.

    Uses `uv venv` / `uv pip` when available because it is dramatically
    faster; the semantics are the same as `python -m venv` + `pip`.
    """

    def __init__(self, root: Path, name: str = ".venv") -> None:
        self.path = root / name
        self.root = root
        self._uv = shutil.which("uv")

    def create(self, *, system_site: bool = False) -> RunResult:
        """`system_site=True` exposes the ambient site-packages to the venv.

        Used here ONLY so an editable install can find the already-installed
        build backend without network access. In a real project you would let
        pip fetch `hatchling` from your index — do NOT enable this in CI, or
        your venv is no longer isolated and a missing dependency will pass
        because the ambient interpreter happens to have it.
        """
        if self._uv:
            args = [self._uv, "venv", str(self.path)]
            if system_site:
                args.append("--system-site-packages")
        else:
            args = [sys.executable, "-m", "venv", str(self.path)]
            if system_site:
                args.append("--system-site-packages")
        p = subprocess.run(args, capture_output=True, text=True, timeout=180)
        return RunResult(p.returncode == 0, p.stdout, p.stderr)

    @property
    def python(self) -> Path:
        return self.path / "bin" / "python"

    def install(self, *targets: str, timeout: int = 300,
                offline: bool = True, no_build_isolation: bool = False
                ) -> RunResult:
        """Install into the venv.

        `offline=True` by default because this container has no PyPI access
        and, more importantly, because an install that needs the network is
        an install that can fail in CI for reasons unrelated to your code.
        """
        if self._uv:
            args = [self._uv, "pip", "install", "--python", str(self.python)]
            if offline:
                args.append("--offline")
            if no_build_isolation:
                args.append("--no-build-isolation")
            args.extend(targets)
        else:
            args = [str(self.python), "-m", "pip", "install", "-q"]
            if offline:
                args.append("--no-index")
            if no_build_isolation:
                args.append("--no-build-isolation")
            args.extend(targets)
        p = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout, cwd=self.root)
        return RunResult(p.returncode == 0, p.stdout, p.stderr)

    def run_python(self, code: str, *, cwd: Path | None = None,
                   timeout: int = 60) -> RunResult:
        """Run code IN the venv, optionally from a given directory.

        `cwd` matters enormously and is the crux of script 01: running from
        the project root puts that root on `sys.path`, so a flat layout
        imports the SOURCE TREE rather than the installed package.
        """
        p = subprocess.run([str(self.python), "-c", code],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(cwd) if cwd else None)
        return RunResult(p.returncode == 0, p.stdout, p.stderr)

    def run_console_script(self, name: str, *args: str,
                           timeout: int = 60) -> RunResult:
        p = subprocess.run([str(self.path / "bin" / name), *args],
                           capture_output=True, text=True, timeout=timeout)
        return RunResult(p.returncode == 0, p.stdout, p.stderr)

    def list_installed(self) -> list[str]:
        if self._uv:
            p = subprocess.run([self._uv, "pip", "list", "--python",
                                str(self.python), "--format", "json"],
                               capture_output=True, text=True, timeout=120)
        else:
            p = subprocess.run([str(self.python), "-m", "pip", "list",
                                "--format", "json"],
                               capture_output=True, text=True, timeout=120)
        try:
            return [f"{d['name']}=={d['version']}" for d in json.loads(p.stdout)]
        except Exception:  # noqa: BLE001
            return []


# ===========================================================================
# 5. OUTPUT HELPERS
# ===========================================================================

def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def section(title: str) -> None:
    print(f"\n  --- {title} ---")


def show(label: str, value: Any) -> None:
    print(f"    {label:<44} {value}")


def verdict(ok: bool, text: str) -> None:
    print(f"    {'PASS' if ok else '*** FAIL ***':<14} {text}")


def code_block(text: str, limit: int = 0) -> None:
    lines = text.splitlines()
    if limit:
        lines = lines[:limit]
    for line in lines:
        print(f"      {line}")
