"""
01 — Configuration layering: sources, precedence, and the traps.

THE PROBLEM
-----------
The same service must run on a laptop, in CI, in a dev tenant, in staging, and
in production — with different endpoints, different quotas, different log
levels, and completely different credential mechanisms. And the difference
between environments must not require a code change or a rebuild.

That is the whole job. The standard solution is LAYERED configuration with a
defined precedence order, and the design questions are:

    1. What are the layers, and in what order?
    2. What belongs in each layer?
    3. What happens when a layer is missing or malformed?
    4. How do you tell which layer a value actually came from at 3am?

Question 4 is the one people skip and then desperately need.

Run:  python 01_config_layers.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from config_lab import banner, clean_env, env, section, show

# ---------------------------------------------------------------------------
# PART 1 — the precedence order
# ---------------------------------------------------------------------------

class Layer(Enum):
    """Ordered lowest to highest priority. Later layers override earlier."""

    DEFAULT = 1        # in code. Safe values that work in the least
                       # privileged environment.
    CONFIG_FILE = 2    # checked in. Per-environment, NO SECRETS.
    ENV_FILE = 3       # .env, local only, GITIGNORED.
    ENVIRONMENT = 4    # actual env vars. How platforms inject config.
    CLI_ARGS = 5       # explicit operator override. Highest for a reason.


@dataclass
class TrackedValue:
    """A config value that remembers WHERE IT CAME FROM.

    This is the feature that pays for itself the first time production
    behaves unlike staging. Without provenance you are reduced to guessing
    which of five layers won, usually by adding print statements to a running
    service.
    """

    value: Any
    layer: Layer
    source: str        # the specific file, variable name, or flag

    def __repr__(self) -> str:
        return f"{self.value!r} (from {self.layer.name}: {self.source})"


class LayeredConfig:
    """Merges sources in precedence order, retaining provenance."""

    def __init__(self) -> None:
        self._values: dict[str, TrackedValue] = {}
        self._history: dict[str, list[TrackedValue]] = {}

    def apply(self, values: dict[str, Any], layer: Layer, source: str) -> None:
        for key, value in values.items():
            tv = TrackedValue(value, layer, source)
            self._history.setdefault(key, []).append(tv)
            existing = self._values.get(key)
            # Strictly-greater, so a later source at the SAME layer does not
            # silently override an earlier one. If two config files both set a
            # key, that is a conflict worth surfacing, not resolving by
            # accident of load order.
            if existing is None or layer.value >= existing.layer.value:
                self._values[key] = tv

    def get(self, key: str, default: Any = None) -> Any:
        tv = self._values.get(key)
        return tv.value if tv is not None else default

    def explain(self, key: str) -> str:
        """`explain("log_level")` -> the winner and everything it beat.

        Ship this as a CLI subcommand (`myservice config explain LOG_LEVEL`)
        or a diagnostic endpoint. It converts a 40-minute investigation into
        a 10-second one.
        """
        winner = self._values.get(key)
        if winner is None:
            return f"{key}: not set in any layer"
        lines = [f"{key} = {winner.value!r}",
                 f"  winner: {winner.layer.name} ({winner.source})"]
        for tv in self._history.get(key, []):
            if tv is not winner:
                lines.append(f"  overridden: {tv.layer.name} ({tv.source}) "
                             f"= {tv.value!r}")
        return "\n".join(lines)


def part1_precedence() -> None:
    banner("PART 1 — layers, precedence, and provenance")

    cfg = LayeredConfig()

    cfg.apply({"log_level": "INFO", "top_k": 5, "timeout_s": 30.0,
               "deployment": "gpt-4o-mini"},
              Layer.DEFAULT, "settings.py defaults")

    cfg.apply({"log_level": "WARNING", "top_k": 8},
              Layer.CONFIG_FILE, "config/production.toml")

    cfg.apply({"top_k": 12}, Layer.ENVIRONMENT, "ISC_TOP_K")

    cfg.apply({"log_level": "DEBUG"}, Layer.CLI_ARGS, "--log-level")

    for key in ("log_level", "top_k", "timeout_s", "deployment"):
        print()
        for line in cfg.explain(key).splitlines():
            print(f"    {line}")

    print("""
    WHY CLI ARGS SIT HIGHEST: an operator debugging a live incident must be
    able to override anything without editing a file or redeploying. That is
    the only layer a human touches under pressure, so it wins.

    WHY DEFAULTS SIT LOWEST AND MUST BE SAFE: whatever a default is, it is
    what runs when every other layer fails to load. `debug=False`,
    `allow_unfiltered_search=False`, `max_spend_usd=<small>`. A default that
    is only safe because "we always set it in production" is a default that
    will one day run in production unset.""")


# ---------------------------------------------------------------------------
# PART 2 — what belongs in which layer
# ---------------------------------------------------------------------------

def part2_placement() -> None:
    banner("PART 2 — what goes where")

    print("""
    ┌──────────────────────────────────────────────────────────────────────┐
    │ CODE DEFAULTS — checked in, safe, environment-agnostic                │
    │   timeouts, retry counts, chunk sizes, feature flags OFF,             │
    │   log format, safe fallbacks                                          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ CONFIG FILE — checked in, per-environment, NEVER SECRET                │
    │   endpoints, deployment names, index names, quotas, tenant ids,       │
    │   which features are on in which environment                          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ ENVIRONMENT VARIABLES — injected by the platform                      │
    │   anything that differs per instance, anything from Key Vault,        │
    │   anything the platform owns (PORT, WEBSITE_*, K_SERVICE)             │
    ├──────────────────────────────────────────────────────────────────────┤
    │ SECRET STORE — never in any file, never in a repo                     │
    │   API keys, connection strings, client secrets, signing keys          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ NOT CONFIG AT ALL — this is the one people get wrong                  │
    │   business logic thresholds that need review and testing,             │
    │   anything whose change should require a PR                           │
    └──────────────────────────────────────────────────────────────────────┘

    THE LAST ROW DESERVES AN ARGUMENT. "Make it configurable" feels safe and
    is often the wrong call. A value in config can be changed by anyone with
    portal access, without review, without a test run, without a deploy
    record. For a genuine operational knob (timeout, concurrency, log level)
    that is exactly what you want. For a threshold that changes what the
    system DECIDES — the confidence floor at which an invoice proposal is
    auto-approved — it is a governance hole.

    THE TEST: if this value changed by 20% and nobody noticed for a week,
    what happens? Slightly slower -> config. Different business outcomes ->
    code, with a test, behind a PR.

    A useful middle ground: keep it in config, but validate a permitted RANGE
    in code and refuse to start outside it. You get operational flexibility
    with a hard bound that requires a deploy to move.""")


# ---------------------------------------------------------------------------
# PART 3 — environment variable conventions
# ---------------------------------------------------------------------------

def part3_env_conventions() -> None:
    banner("PART 3 — environment variable conventions")

    print("""
    PREFIX EVERYTHING. `ISC_AI__AZURE__ENDPOINT`, not `ENDPOINT`.

    Three reasons, all of which bite:
      * Collision. `HOST`, `PORT`, `TIMEOUT`, `DEBUG`, and `ENV` are set by
        platforms, base images, and other tools. Unprefixed names get
        clobbered by things you do not control.
      * Discoverability. `env | grep ISC_AI__` shows your entire
        configuration surface at a glance. That is the first command you want
        during an incident.
      * Blast radius. A prefixed variable cannot accidentally reconfigure a
        sidecar or a library that happens to read the same name.

    NESTING with a double underscore:
        ISC_AI__AZURE__OPENAI__ENDPOINT
        ISC_AI__AZURE__OPENAI__DEPLOYMENT
        ISC_AI__RETRIEVAL__TOP_K

    Double, not single, because single underscores are ambiguous:
    `ISC_TOP_K` could be top.k or top_k. Doubles remove the guesswork and
    pydantic-settings supports them directly (script 02).

    TYPES: env vars are ALWAYS strings. Every non-string value needs parsing,
    and the parsing is where the bugs live.""")

    section("the boolean trap, demonstrated")
    for raw in ("true", "True", "1", "yes", "on", "false", "False", "0",
                "no", "off", "", "FALSE"):
        naive = bool(raw)
        print(f"      {raw!r:<9} bool(x)={naive!s:<6} "
              f"{'<-- WRONG' if naive and raw.lower() in ('false','0','no','off') else ''}")

    print("""
      `bool("false")` is True. So is `bool("0")` and `bool("no")`. Any code
      doing `DEBUG = bool(os.environ.get("DEBUG"))` has a flag that can only
      ever be turned ON.

      Use a parser that knows the vocabulary — pydantic-settings does, and
      accepts true/True/1/yes/on and their negatives. Never `bool()`.""")

    section("empty string vs unset")
    with env(ISC_DEMO_VALUE=""):
        show('os.environ.get("ISC_DEMO_VALUE")', repr(os.environ.get("ISC_DEMO_VALUE")))
        show('os.environ.get(..., "fallback")', repr(os.environ.get("ISC_DEMO_VALUE", "fallback")))
    print("""
      An empty string is SET. `.get(key, default)` returns "" and not the
      default, so an accidentally-empty variable silently overrides your
      fallback with nothing.

      This happens constantly in practice: a Key Vault reference that fails to
      resolve, a CI secret that is not defined for fork PRs, or a Helm value
      templated from a missing key. All three produce "" rather than unset.

      DEFEND WITH VALIDATION, not with `.get()` defaults — `min_length=1` on
      the field, so an empty value fails at startup with a clear message
      instead of producing a client pointed at "".""")

    section("the multi-value trap")
    with env(ISC_ALLOWED_GROUPS="isc-all,isc-readers"):
        raw = os.environ["ISC_ALLOWED_GROUPS"]
        show("raw", repr(raw))
        show("split(',')", raw.split(","))
        show("with a trailing comma", ["isc-all", "isc-readers", ""])
    print("""
      A trailing comma yields an empty-string element, which for a GROUPS list
      means an empty group id in your permission filter. Strip and drop empty
      elements, or use a JSON array in the variable and parse it properly.""")


# ---------------------------------------------------------------------------
# PART 4 — config files and environment overlays
# ---------------------------------------------------------------------------

def part4_overlays() -> None:
    banner("PART 4 — base + overlay, not five parallel files")

    base = {
        "retrieval": {"top_k": 5, "rerank": True, "score_threshold": 0.0},
        "generation": {"deployment": "gpt-4o-mini", "max_tokens": 512,
                       "temperature": 0.0},
        "limits": {"max_concurrency": 8, "rpm": 60, "daily_spend_usd": 10.0},
        "features": {"streaming": False, "tool_calling": False},
        "log_level": "INFO",
    }

    overlays = {
        "local": {"log_level": "DEBUG",
                  "limits": {"rpm": 10, "daily_spend_usd": 1.0}},
        "staging": {"limits": {"max_concurrency": 16, "rpm": 300},
                    "features": {"streaming": True}},
        "production": {"log_level": "WARNING",
                       "retrieval": {"top_k": 8},
                       "limits": {"max_concurrency": 64, "rpm": 3000,
                                  "daily_spend_usd": 500.0},
                       "features": {"streaming": True, "tool_calling": True}},
    }

    def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        """Recursive merge. THE COMMON BUG is a shallow merge:

            {**base, **overlay}

        With the production overlay above, a shallow merge replaces the whole
        `limits` dict — silently dropping nothing here, but in a real overlay
        that sets only `rpm`, it would delete `max_concurrency` and
        `daily_spend_usd`, leaving them unset. That is how a spend cap
        disappears in production and nobody notices until the bill.
        """
        out = dict(base)
        for k, v in overlay.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    for name, overlay in overlays.items():
        merged = deep_merge(base, overlay)
        section(name)
        show("log_level", merged["log_level"])
        show("limits", merged["limits"])
        show("features", merged["features"])

    section("shallow merge, for comparison")
    shallow = {**base, **{"limits": {"rpm": 3000}}}
    show("limits after a shallow merge", shallow["limits"])
    print("      max_concurrency and daily_spend_usd are GONE. If the code")
    print("      then does `limits.get('daily_spend_usd', float('inf'))`,")
    print("      the production spend cap is silently infinite.")

    print("""
    WHY BASE+OVERLAY BEATS PARALLEL FILES: with production.toml, staging.toml,
    and local.toml as complete independent files, they drift. A setting added
    to two of them and forgotten in the third is the classic
    "works in staging, breaks in production" bug — and the diff between two
    500-line files does not make it obvious.

    With an overlay, the file IS the diff. Reviewing production.toml means
    reading eight lines, and every one of them is a deliberate deviation.""")


# ---------------------------------------------------------------------------
# PART 5 — the traps
# ---------------------------------------------------------------------------

def part5_traps() -> None:
    banner("PART 5 — configuration traps worth naming in review")

    print("""
    1. READING CONFIG AT IMPORT TIME
         DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]   # module level

       Breaks tests (cannot set env before import), breaks tooling (a
       `--help` invocation now requires production config), and produces a
       KeyError during import that surfaces as a confusing stack trace with
       no useful message. Load config in a function, once, at startup.

    2. CONFIG AS A MUTABLE GLOBAL
       Anything can change it at runtime, so behaviour depends on import
       order and on whatever ran first. Make it FROZEN and pass it explicitly,
       or hold it in a single application-state object.

    3. `os.environ.get()` SCATTERED THROUGH THE CODEBASE
       There is now no single place that lists what the service needs. Nobody
       can answer "what does this need to run?" without grepping, and a typo
       in a variable name silently produces a default. One settings class,
       one place.

    4. DIFFERENT CONFIG SHAPES PER ENVIRONMENT
       If production reads a secret from Key Vault and local reads a
       hardcoded string through a DIFFERENT code path, that path is untested
       until it runs in production. Keep the SHAPE identical and vary only the
       SOURCE — script 05.

    5. SECRETS IN THE CONFIG FILE, "just for local dev"
       It will be committed. Not by you, by someone in a hurry six months
       from now. Use a gitignored .env file and check in a .env.example with
       the keys and dummy values.

    6. NO PROVENANCE
       See PART 1. When production disagrees with staging you need to know
       which layer won, not to reason about which layer SHOULD have won.

    7. VALIDATING LAZILY
       A malformed value discovered on the first request that touches it,
       possibly hours after deployment, possibly only in one code path.
       Validate everything at startup — script 03.""")

    section("trap 1, made concrete")
    with clean_env("ISC_DEMO_"):
        try:
            _ = os.environ["ISC_DEMO_REQUIRED"]
        except KeyError as e:
            show("os.environ[missing] at import time", f"KeyError: {e}")
        print("      A KeyError during import, with no indication of WHICH")
        print("      service needs it, WHY, or what a valid value looks like.")
        print("      Compare with a settings class, which reports every")
        print("      missing field at once with a description for each.")


def main() -> None:
    part1_precedence()
    part2_placement()
    part3_env_conventions()
    part4_overlays()
    part5_traps()

    banner("SUMMARY")
    print("""
  * Five layers, lowest to highest: defaults, config file, .env, environment,
    CLI. CLI wins because a human under pressure must be able to override.
  * Defaults must be SAFE — they are what runs when everything else fails.
  * Prefix every variable; use __ for nesting; never `bool()` on a string.
  * An empty string is SET, so `.get(key, default)` will not save you.
    Validate `min_length=1` instead.
  * Base + overlay, not parallel per-environment files. Deep merge, never
    shallow — a shallow merge silently deletes sibling keys like spend caps.
  * Track PROVENANCE and ship an `explain` command. It is the difference
    between a 10-second and a 40-minute incident.
  * "Make it configurable" is not free: config changes without review. If a
    value changes what the system DECIDES, it belongs in code.
""")


if __name__ == "__main__":
    main()
