"""
04 — Secrets: the seven surfaces they leak through.

THE PREMISE
-----------
"Don't hardcode secrets" is solved. Nobody commits an API key on purpose any
more, and secret scanners catch the ones who do.

The leaks that actually reach production logs are all the same shape: SOME
CODE PATH FORMATTED AN OBJECT. An exception message, a repr in a traceback, a
structured log line, a span attribute, a URL. None of them looks like
mishandling a secret; every one of them is.

So the defence cannot be discipline — there are hundreds of format calls and
you will not audit them all. The defence has to be STRUCTURAL: give the secret
a type that refuses to render itself, so every one of those paths is safe by
default and the only way to expose the value is to ask for it explicitly.

This script demonstrates all seven surfaces with a detector that scans for the
actual secret value, then shows the structural fix for each.

Run:  python 04_secrets.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, SecretStr

from config_lab import (
    FAKE_API_KEY,
    FAKE_CLIENT_SECRET,
    FAKE_DB_PASSWORD,
    LeakDetector,
    banner,
    section,
    verdict,
)

DETECTOR = LeakDetector()
DETECTOR.register("AZURE_OPENAI_KEY", FAKE_API_KEY)
DETECTOR.register("DB_PASSWORD", FAKE_DB_PASSWORD)
DETECTOR.register("CLIENT_SECRET", FAKE_CLIENT_SECRET)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format="%(levelname)s %(message)s", force=True)
log = logging.getLogger("isc")
log.propagate = True


# ---------------------------------------------------------------------------
# The two config objects: naive and defended
# ---------------------------------------------------------------------------

@dataclass
class NaiveConfig:
    """Plain strings. Every leak surface is open."""

    endpoint: str
    deployment: str
    api_key: str
    db_password: str


class DefendedConfig(BaseModel):
    """SecretStr wraps the value so that repr, str, log formatting, JSON
    serialisation, and traceback rendering all produce `**********`.

    THE ONE WAY OUT is `.get_secret_value()`, which is greppable. That single
    property converts "audit every format call" into "audit every
    get_secret_value call" — from hundreds of sites to a handful.
    """

    endpoint: str
    deployment: str
    api_key: SecretStr
    db_password: SecretStr


NAIVE = NaiveConfig(
    endpoint="https://isc-aoai-prod.openai.azure.com/",
    deployment="gpt-4o-mini-prod",
    api_key=FAKE_API_KEY,
    db_password=FAKE_DB_PASSWORD,
)
DEFENDED = DefendedConfig(
    endpoint="https://isc-aoai-prod.openai.azure.com/",
    deployment="gpt-4o-mini-prod",
    api_key=SecretStr(FAKE_API_KEY),
    db_password=SecretStr(FAKE_DB_PASSWORD),
)


# ---------------------------------------------------------------------------
# SURFACE 1 — repr and str
# ---------------------------------------------------------------------------

def surface1_repr() -> None:
    banner("SURFACE 1 — repr() and str()")

    section("naive")
    leaks = DETECTOR.scan_object(NAIVE, "repr(NaiveConfig)")
    print(f"      repr: {NAIVE!r}"[:130])
    verdict(not leaks, f"{len(leaks)} leak(s)")

    section("defended")
    leaks = DETECTOR.scan_object(DEFENDED, "repr(DefendedConfig)")
    print(f"      repr: {DEFENDED!r}"[:130])
    verdict(not leaks, f"{len(leaks)} leak(s)")

    print("""
    WHY THIS SURFACE MATTERS MOST: you almost never call repr() deliberately.
    It is called FOR you — by the debugger, by pytest on an assertion failure,
    by `print()` on a container, by Jupyter, and by traceback rendering. Every
    one of those is a place a plain-string config object spills.""")


# ---------------------------------------------------------------------------
# SURFACE 2 — exception messages
# ---------------------------------------------------------------------------

def surface2_exceptions() -> None:
    banner("SURFACE 2 — exception messages and tracebacks")

    section("naive: the classic connection-string-in-the-error")

    def connect_naive(cfg: NaiveConfig) -> None:
        conn = (f"Server=isc-sql.database.windows.net;Database=isc;"
                f"User Id=svc;Password={cfg.db_password};")
        raise ConnectionError(f"failed to connect using {conn}")

    try:
        connect_naive(NAIVE)
    except ConnectionError as e:
        leaks = DETECTOR.scan_exception(e)
        print(f"      {str(e)[:110]}")
        verdict(not leaks, f"{len(leaks)} leak(s) in the traceback")

    section("defended: never interpolate the secret into the message")

    def connect_defended(cfg: DefendedConfig) -> None:
        # Build the connection string, use it, and NEVER put it in a message.
        # The error names the SERVER and the USER — everything an engineer
        # needs to debug — and nothing an attacker needs.
        raise ConnectionError(
            "failed to connect to isc-sql.database.windows.net as 'svc' "
            "(password from DEFENDED.db_password)"
        )

    try:
        connect_defended(DEFENDED)
    except ConnectionError as e:
        leaks = DETECTOR.scan_exception(e)
        print(f"      {e}")
        verdict(not leaks, f"{len(leaks)} leak(s) in the traceback")

    print("""
    THE SUBTLE ONE: some libraries put the failing INPUT in the exception.
    `int("abc")` includes "abc". A driver that fails to parse a connection
    string may include the whole string. You do not control that, which is
    another reason to keep the secret inside a SecretStr until the last
    possible moment and to hand it to the library as a separate parameter
    rather than embedded in a URL or DSN where a parse error can echo it.""")


# ---------------------------------------------------------------------------
# SURFACE 3 — logging
# ---------------------------------------------------------------------------

def surface3_logging() -> None:
    banner("SURFACE 3 — log lines, including the ones you did not write")

    section("naive: 'log the config at startup, it helps debugging'")
    with DETECTOR.capture_logs("isc"):
        log.info("starting with config: %s", NAIVE)
        log.debug("connecting to %s with key %s", NAIVE.endpoint, NAIVE.api_key)
    naive_leaks = [x for x in DETECTOR.leaks if x.surface == "log"]
    verdict(not naive_leaks, f"{len(naive_leaks)} leak(s) in captured logs")
    DETECTOR.reset()

    section("defended: the same log statements, safe by construction")
    with DETECTOR.capture_logs("isc"):
        log.info("starting with config: %s", DEFENDED)
        log.debug("connecting to %s with key %s", DEFENDED.endpoint,
                  DEFENDED.api_key)
    leaks = [x for x in DETECTOR.leaks if x.surface == "log"]
    verdict(not leaks, f"{len(leaks)} leak(s) in captured logs")

    print("""
    THE SECOND LINE IS THE POINT. `log.debug(..., DEFENDED.api_key)` is
    careless code — someone deliberately logged a field named `api_key`. And
    it is STILL SAFE, because SecretStr's __str__ returns '**********'.

    That is what structural defence means: the careless path is safe, so you
    do not have to be careful in hundreds of places.

    THE REMAINING HOLE, and it is real: `.get_secret_value()` returns a plain
    str, and from that moment the value is naked again. Grep for it — every
    call should be immediately adjacent to the client that consumes it, never
    assigned to a variable that lives on.""")


# ---------------------------------------------------------------------------
# SURFACE 4 — serialisation
# ---------------------------------------------------------------------------

def surface4_serialisation() -> None:
    banner("SURFACE 4 — JSON dumps, diagnostics endpoints, cache writes")

    section("naive")
    from dataclasses import asdict
    payload = json.dumps(asdict(NAIVE))
    leaks = DETECTOR.scan(payload, "json.dumps(naive)")
    print(f"      {payload[:110]}")
    verdict(not leaks, f"{len(leaks)} leak(s)")

    section("defended: model_dump_json()")
    payload = DEFENDED.model_dump_json()
    leaks = DETECTOR.scan(payload, "model_dump_json")
    print(f"      {payload[:110]}")
    verdict(not leaks, f"{len(leaks)} leak(s)")

    section("THE TRAP: model_dump(mode='json') is NOT automatically safe")
    dumped = DEFENDED.model_dump()
    print(f"      model_dump() gives: {dumped}")
    print("      -> the values are SecretStr objects, safe to print...")
    unsafe = {k: (v.get_secret_value() if isinstance(v, SecretStr) else v)
              for k, v in dumped.items()}
    leaks = DETECTOR.scan(json.dumps(unsafe), "manual unwrap")
    verdict(not leaks, f"{len(leaks)} leak(s) after a manual unwrap")
    DETECTOR.reset()

    print("""
    The unwrap above is contrived, but the real version is not: someone needs
    the config as a plain dict for a template, a subprocess env, or a cache
    key, writes exactly that comprehension, and every downstream consumer now
    holds naked secrets.

    IF YOU NEED A PLAIN DICT, make a REDACTED one explicitly and give it a
    name that says so — `config.redacted_dict()`. Then the unsafe version has
    to be written deliberately and is visible in review.""")


# ---------------------------------------------------------------------------
# SURFACE 5 — URLs and query strings
# ---------------------------------------------------------------------------

def surface5_urls() -> None:
    banner("SURFACE 5 — secrets in URLs")

    section("naive: an API key as a query parameter")
    url = (f"https://isc-search.search.windows.net/indexes/isc-docs-v3/docs"
           f"?api-key={urllib.parse.quote(FAKE_API_KEY)}&search=invoice")
    leaks = DETECTOR.scan(url, "url")
    print(f"      {url[:110]}")
    verdict(not leaks, f"{len(leaks)} leak(s) — note the detector catches "
                       f"the URL-ENCODED form")

    print("""
    A secret in a URL is written down in more places than anywhere else:
      * the web server access log, on both ends
      * every proxy, gateway, and CDN in between
      * browser history and the Referer header
      * your own APM's "http.url" span attribute
      * a screenshot in a bug report

    None of those is under your log-redaction control. Even over TLS, the URL
    is logged at both endpoints in cleartext.

    ALWAYS use a header: `Authorization: Bearer ...` or `api-key: ...`.
    Headers are not logged by default anywhere in that list.""")

    section("defended: header, and the detector confirms the URL is clean")
    url2 = ("https://isc-search.search.windows.net/indexes/isc-docs-v3/docs"
            "?search=invoice")
    headers = {"api-key": DEFENDED.api_key}   # SecretStr, unwrapped at send
    leaks = DETECTOR.scan(url2 + repr(headers), "url+headers repr")
    print(f"      url:     {url2}")
    print(f"      headers: {headers}")
    verdict(not leaks, f"{len(leaks)} leak(s)")


# ---------------------------------------------------------------------------
# SURFACE 6 — subprocess arguments
# ---------------------------------------------------------------------------

def surface6_subprocess() -> None:
    banner("SURFACE 6 — subprocess argv is world-readable")

    print("""
    subprocess.run(["some-cli", "--api-key", key])

    On Linux, /proc/<pid>/cmdline is readable by any process running as the
    same user, and often more broadly. `ps aux` shows it. Container runtimes
    log it. A sidecar can read it.

    THE FIX: pass secrets through the ENVIRONMENT or through STDIN, never
    argv. Environment is per-process and not visible in `ps`; stdin is better
    still because it leaves no trace at all.""")

    section("demonstrating that argv is visible")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; print('argv seen by the child:', sys.argv[1:])",
         "--api-key", "SECRET-IN-ARGV"],
        capture_output=True, text=True,
    )
    print(f"      {proc.stdout.strip()}")
    print("      ...and the same string is in /proc/<pid>/cmdline while it runs.")

    section("the fix: environment")
    proc2 = subprocess.run(
        [sys.executable, "-c",
         "import os; k=os.environ.get('CHILD_KEY','');"
         "print('child got a key of length', len(k))"],
        capture_output=True, text=True,
        env={"CHILD_KEY": DEFENDED.api_key.get_secret_value(), "PATH": "/usr/bin"},
    )
    print(f"      {proc2.stdout.strip()}")
    print("      argv contains no secret; the value is in the child's env only.")


# ---------------------------------------------------------------------------
# SURFACE 7 — telemetry
# ---------------------------------------------------------------------------

@dataclass
class FakeSpan:
    """Models an OpenTelemetry span. Attributes are serialised and shipped."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def export(self) -> str:
        return json.dumps({"name": self.name, "attributes": self.attributes},
                          default=str)


def surface7_telemetry() -> None:
    banner("SURFACE 7 — span attributes and APM tags")

    section("naive: attaching the whole config to a span")
    from dataclasses import asdict
    span = FakeSpan("rag.request")
    span.set_attribute("config", asdict(NAIVE))
    leaks = DETECTOR.scan(span.export(), "span")
    print(f"      {span.export()[:110]}")
    verdict(not leaks, f"{len(leaks)} leak(s)")

    section("defended")
    span2 = FakeSpan("rag.request")
    span2.set_attribute("endpoint", DEFENDED.endpoint)
    span2.set_attribute("deployment", DEFENDED.deployment)
    leaks = DETECTOR.scan(span2.export(), "span")
    print(f"      {span2.export()[:110]}")
    verdict(not leaks, f"{len(leaks)} leak(s)")

    print("""
    TELEMETRY IS A SEPARATE TRUST BOUNDARY, and this is the argument worth
    making in a design review. The people who can read your Application
    Insights workspace are not the same set as the people who can read your
    Key Vault — and usually a much larger set, because query access to
    telemetry is handed out freely for debugging.

    So a secret in a span has not just been logged; it has been COPIED into a
    system with a different access model and a different retention period.

    The same argument covers prompt text and retrieved document content, which
    is why `log_level=DEBUG in production` was a startup-fatal rule in script
    02: it moves document content into the telemetry workspace.""")


# ---------------------------------------------------------------------------
# The redaction helper, and why a denylist is not enough
# ---------------------------------------------------------------------------

def redacted_dict(model: BaseModel) -> dict[str, Any]:
    """Produce a dict safe to log, using the TYPE rather than the key name."""
    out: dict[str, Any] = {}
    for name, value in model:
        if isinstance(value, SecretStr):
            out[name] = "***REDACTED***"
        elif isinstance(value, BaseModel):
            out[name] = redacted_dict(value)
        else:
            out[name] = value
    return out


SUSPICIOUS = ("key", "secret", "password", "token", "credential")


def denylist_redact(d: dict[str, Any]) -> dict[str, Any]:
    """The tempting alternative. Shown so its failure is visible."""
    return {k: ("***" if any(s in k.lower() for s in SUSPICIOUS) else v)
            for k, v in d.items()}


def part8_redaction_strategy() -> None:
    banner("STRUCTURAL vs DENYLIST redaction")

    section("type-based redaction (structural)")
    print(f"      {redacted_dict(DEFENDED)}")

    section("name-based redaction (denylist) — where it fails")
    tricky = {
        "api_key": FAKE_API_KEY,                         # caught
        "sas_url": f"https://blob/?sig={FAKE_API_KEY}",  # MISSED
        "conn": f"Password={FAKE_DB_PASSWORD};",         # MISSED
        "bootstrap_servers": f"user:{FAKE_CLIENT_SECRET}@kafka:9092",  # MISSED
    }
    result = denylist_redact(tricky)
    DETECTOR.reset()
    leaks = DETECTOR.scan(json.dumps(result), "denylist output")
    for k, v in result.items():
        print(f"      {k:<20} {str(v)[:56]}")
    verdict(not leaks, f"{len(leaks)} leak(s) survived the denylist")

    print("""
    Three of four leaked. `sas_url`, `conn`, and `bootstrap_servers` contain
    secret material and none of their NAMES contains a suspicious word.

    A denylist protects against the fields you thought of. Types protect
    against the fields you did not — which is the entire population of fields
    added by someone else, later, in a hurry.

    THE RULE: `SecretStr` (or an equivalent wrapper) at the point of
    DECLARATION. Redaction then follows from the type, and a new secret field
    is protected the moment it is declared rather than the moment someone
    remembers to add it to a list.""")


def main() -> None:
    surface1_repr()
    surface2_exceptions()
    surface3_logging()
    surface4_serialisation()
    surface5_urls()
    surface6_subprocess()
    surface7_telemetry()
    part8_redaction_strategy()

    banner("FINAL LEAK REPORT")
    print(f"    Total leaks recorded across this run: {len(DETECTOR.leaks)}")
    print("    (all of them deliberate demonstrations from the 'naive' paths)")

    banner("SUMMARY")
    print("""
  * Seven surfaces: repr, exception messages, logs, serialisation, URLs,
    subprocess argv, telemetry. All are "some code formatted an object".
  * Defend STRUCTURALLY with SecretStr, not with discipline. The careless
    path must be the safe path.
  * `.get_secret_value()` is the one hole, and it is greppable. Keep each call
    adjacent to the client that consumes it.
  * Never put a secret in a URL — access logs, proxies, Referer, APM span
    attributes, and screenshots all capture it. Use a header.
  * Never put a secret in argv. Use the environment or stdin.
  * Telemetry is a SEPARATE TRUST BOUNDARY with a larger reader set. This
    covers prompts and retrieved content as well as credentials.
  * Redact by TYPE, never by key name. A denylist only protects the fields
    someone remembered.
  * Register your test secrets with a scanner and assert zero leaks — see
    script 09. It is the only way "we don't log secrets" becomes testable.
""")


if __name__ == "__main__":
    main()
