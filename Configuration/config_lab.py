"""
config_lab.py — shared harness for the configuration and secrets tutorial.

WHAT THIS PROVIDES
------------------
  * LeakDetector — the centrepiece. Scans log output, reprs, tracebacks, and
    serialised JSON for known secret material. Most "don't log secrets" advice
    is untestable; this makes it a red test.
  * A fake Azure credential chain with realistic behaviour: ordered fallback,
    token caching, expiry, and per-scope tokens.
  * A fake Key Vault with versioning, so rotation can be demonstrated.
  * Environment isolation helpers, because config tests that leak env vars
    into each other are worse than no config tests.

THE THREAT MODEL, stated up front
---------------------------------
"Don't hardcode secrets" is the easy part and almost nobody gets it wrong any
more. The leaks that actually happen in production:

    1. An exception message containing a connection string.
    2. A dataclass or Pydantic repr printed in a traceback.
    3. A config object dumped to a structured log at DEBUG level.
    4. A secret placed in a URL query parameter, which lands in access logs,
       proxy logs, and browser history.
    5. A subprocess invoked with a secret in argv, visible in `ps`.
    6. Telemetry: a secret attached as a span attribute or an APM tag.
    7. A secret in an error returned to the CLIENT.

Every one of those is a code path that formats an object. So the defence is
structural: make the secret's type refuse to format itself.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# ===========================================================================
# 1. THE LEAK DETECTOR
# ===========================================================================

@dataclass
class Leak:
    """One detected exposure of secret material."""

    surface: str          # where it leaked: "log", "repr", "traceback", ...
    secret_name: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.surface}: {self.secret_name} in {self.excerpt!r}"


class LeakDetector:
    """Registers known secret VALUES and scans arbitrary text for them.

    WHY THIS IS THE RIGHT TOOL: you cannot test "no secrets are logged" by
    reading code. Every `logger.info(f"...{config}")` is a potential leak and
    there are hundreds of them. What you CAN do is register the actual secret
    values used in tests, then scan everything the process emits.

    In production the same idea runs as a log-pipeline scrubber, but at that
    point the secret has already left the process. Catching it in tests is
    strictly better — and it is a red test rather than an alert.

    LIMITS, stated honestly:
      * Only finds secrets it was TOLD about. It cannot detect a leak of a
        value it does not know.
      * A short or low-entropy secret ("password") produces false positives.
        Test secrets should be long and distinctive, like the ones below.
      * It scans text. A secret written to a binary format or encoded
        (base64, URL-encoding) evades it unless you register the encoded form
        too — `register` handles the common encodings for you.
    """

    def __init__(self) -> None:
        self._secrets: dict[str, list[str]] = {}
        self.leaks: list[Leak] = []

    def register(self, name: str, value: str) -> None:
        """Register a secret and its common encodings.

        The encodings matter: a secret in a URL is percent-encoded, and one in
        a JSON body may be escaped. Scanning only for the raw value misses
        both.
        """
        import base64
        import urllib.parse

        forms = [
            value,
            urllib.parse.quote(value, safe=""),
            base64.b64encode(value.encode()).decode(),
            json.dumps(value)[1:-1],          # JSON-escaped, without quotes
        ]
        self._secrets[name] = [f for f in forms if f]

    def scan(self, text: str, surface: str) -> list[Leak]:
        found: list[Leak] = []
        for name, forms in self._secrets.items():
            for form in forms:
                idx = text.find(form)
                if idx >= 0:
                    start = max(0, idx - 24)
                    excerpt = text[start:idx + len(form) + 16]
                    leak = Leak(surface, name, excerpt)
                    found.append(leak)
                    self.leaks.append(leak)
                    break
        return found

    def scan_object(self, obj: object, surface: str = "repr") -> list[Leak]:
        """Scan repr(), str(), and a JSON dump if the object supports it."""
        found = self.scan(repr(obj), surface)
        found += self.scan(str(obj), f"{surface}/str")
        dump = getattr(obj, "model_dump_json", None)
        if callable(dump):
            with contextlib.suppress(Exception):
                found += self.scan(dump(), f"{surface}/model_dump_json")
        return found

    def scan_exception(self, exc: BaseException) -> list[Leak]:
        """Scan the full formatted traceback, which includes local variables
        under some formatters and always includes the exception message."""
        text = "".join(traceback.format_exception(type(exc), exc,
                                                  exc.__traceback__))
        return self.scan(text, "traceback")

    @contextmanager
    def capture_logs(self, logger_name: str = "") -> Iterator[io.StringIO]:
        """Capture everything written to a logger and scan it on exit.

        This is what turns "we are careful about logging" into a test.
        """
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger = logging.getLogger(logger_name)
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            yield stream
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
            self.scan(stream.getvalue(), "log")

    def report(self) -> str:
        if not self.leaks:
            return "no leaks detected"
        lines = [f"{len(self.leaks)} LEAK(S) DETECTED:"]
        lines += [f"    - {leak}" for leak in self.leaks]
        return "\n".join(lines)

    def reset(self) -> None:
        self.leaks.clear()


# Distinctive test values. Long and unmistakable so the detector cannot
# produce false positives, and so a leak is obvious when you see it.
FAKE_API_KEY = "sk-TESTKEY-9f2a41c7b3e84d16a5f0-DO-NOT-SHIP"
FAKE_DB_PASSWORD = "Pw-TESTDB-3e91f4a72c0b-DO-NOT-SHIP"
FAKE_CLIENT_SECRET = "cs-TESTCLIENT-71bd0e5a9f34-DO-NOT-SHIP"


# ===========================================================================
# 2. FAKE AZURE CREDENTIAL CHAIN
# ===========================================================================

@dataclass(frozen=True)
class AccessToken:
    """Mirrors azure.core.credentials.AccessToken.

    Note `expires_on` is an absolute epoch timestamp, not a duration. That is
    the real shape and it matters: code that computes `now + expires_in` from
    a duration drifts if the token was minted before your clock ticked.
    """

    token: str
    expires_on: float

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.expires_on - time.time())

    def __repr__(self) -> str:
        # A token IS a bearer credential. Never render it.
        return f"AccessToken(expires_in={self.seconds_remaining:.0f}s)"


class CredentialUnavailable(Exception):
    """One link in the chain cannot produce a token. Try the next."""


class CredentialFailed(Exception):
    """A link was ABLE to try and was REJECTED. This is terminal — a
    misconfigured identity should not silently fall through to the next
    credential, because that produces a service running as the wrong
    principal, which is worse than a hard failure."""


class FakeCredentialLink:
    """One credential source in the chain."""

    def __init__(self, name: str, *, available: bool = True,
                 lifetime: float = 3600.0, latency: float = 0.0) -> None:
        self.name = name
        self.available = available
        self.lifetime = lifetime
        self.latency = latency
        self.calls = 0

    def get_token(self, *scopes: str) -> AccessToken:
        self.calls += 1
        if self.latency:
            time.sleep(self.latency)
        if not self.available:
            raise CredentialUnavailable(f"{self.name}: not available here")
        return AccessToken(
            token=f"eyJ-FAKE-{self.name}-{'.'.join(scopes)}-token",
            expires_on=time.time() + self.lifetime,
        )


class FakeDefaultAzureCredential:
    """Mirrors DefaultAzureCredential: an ordered chain with token caching.

    THE REAL ORDER (as of the current SDK) is roughly:
        EnvironmentCredential
        WorkloadIdentityCredential
        ManagedIdentityCredential
        SharedTokenCacheCredential
        AzureCliCredential
        AzurePowerShellCredential
        AzureDeveloperCliCredential
        InteractiveBrowserCredential  (disabled by default)

    Two properties worth internalising:
      1. It CACHES tokens per scope. Constructing a new credential per request
         defeats that cache and hammers the IMDS endpoint, which throttles.
      2. It falls through on CredentialUnavailable but NOT on a real auth
         failure — see the CredentialFailed distinction above.
    """

    def __init__(self, links: list[FakeCredentialLink]) -> None:
        self.links = links
        self._cache: dict[tuple[str, ...], AccessToken] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.chain_attempts: list[str] = []

    def get_token(self, *scopes: str, refresh_margin: float = 300.0) -> AccessToken:
        key = tuple(scopes)
        cached = self._cache.get(key)
        # Refresh EARLY, before actual expiry. A token that expires between
        # your check and the downstream call produces a 401 you have to retry.
        if cached is not None and cached.seconds_remaining > refresh_margin:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        errors: list[str] = []
        for link in self.links:
            try:
                token = link.get_token(*scopes)
            except CredentialUnavailable as e:
                self.chain_attempts.append(f"{link.name}: unavailable")
                errors.append(str(e))
                continue
            self.chain_attempts.append(f"{link.name}: OK")
            self._cache[key] = token
            return token
        raise CredentialFailed(
            "no credential in the chain produced a token: " + "; ".join(errors)
        )


# ===========================================================================
# 3. FAKE KEY VAULT
# ===========================================================================

@dataclass
class SecretVersion:
    value: str
    version: str
    created: float
    enabled: bool = True


class FakeKeyVault:
    """A secret store with VERSIONING, so rotation can be demonstrated.

    Real Key Vault semantics that matter here:
      * `get_secret(name)` returns the CURRENT version.
      * Old versions remain retrievable by id until purged — which is what
        makes a dual-secret rotation window possible.
      * Reads are network calls and are RATE LIMITED. Fetching a secret on
        every request is a real production failure mode; cache with a TTL.
    """

    def __init__(self, name: str = "kv-isc-prod") -> None:
        self.name = name
        self._secrets: dict[str, list[SecretVersion]] = {}
        self.get_calls = 0
        self.throttle_after: int | None = None

    def set_secret(self, name: str, value: str) -> SecretVersion:
        versions = self._secrets.setdefault(name, [])
        v = SecretVersion(value=value, version=f"v{len(versions) + 1}",
                          created=time.time())
        versions.append(v)
        return v

    def get_secret(self, name: str, version: str | None = None) -> SecretVersion:
        self.get_calls += 1
        if self.throttle_after is not None and self.get_calls > self.throttle_after:
            raise RuntimeError("429 Key Vault request rate exceeded")
        versions = self._secrets.get(name)
        if not versions:
            raise KeyError(f"secret {name!r} not found in {self.name}")
        if version is None:
            live = [v for v in versions if v.enabled]
            if not live:
                raise KeyError(f"secret {name!r} has no enabled version")
            return live[-1]
        for v in versions:
            if v.version == version:
                return v
        raise KeyError(f"secret {name!r} version {version!r} not found")

    def disable(self, name: str, version: str) -> None:
        for v in self._secrets.get(name, []):
            if v.version == version:
                v.enabled = False

    def uri(self, name: str, version: str | None = None) -> str:
        """The Key Vault reference form used in App Service / Container Apps:
            @Microsoft.KeyVault(SecretUri=https://kv/secrets/name/version)
        The platform resolves it and injects the VALUE as an env var, so your
        code never talks to Key Vault at all — see script 05.
        """
        suffix = f"/{version}" if version else ""
        return f"https://{self.name}.vault.azure.net/secrets/{name}{suffix}"


# ===========================================================================
# 4. ENVIRONMENT ISOLATION
# ===========================================================================

@contextmanager
def env(**overrides: str | None) -> Iterator[None]:
    """Temporarily set/unset environment variables, restoring on exit.

    WHY THIS MATTERS FOR TESTS: os.environ is process-global. A test that sets
    AZURE_OPENAI_ENDPOINT and does not clean up will change the behaviour of
    every test that runs after it, in an order-dependent way that only shows
    up in CI. Use this, or pytest's `monkeypatch.setenv`, always.

    Passing None removes the variable, which is distinct from setting it to
    "" — a distinction that matters more than it looks (see script 01).
    """
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def clean_env(prefix: str) -> Iterator[None]:
    """Remove every variable with a prefix, restoring afterwards.

    Essential when testing a settings class: a developer's real
    AZURE_OPENAI_* variables would otherwise bleed into the test and make it
    pass on their machine and fail in CI (or vice versa).
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith(prefix)}
    try:
        for k in saved:
            os.environ.pop(k, None)
        yield
    finally:
        os.environ.update(saved)


# ===========================================================================
# 5. OUTPUT HELPERS
# ===========================================================================

def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def section(title: str) -> None:
    print(f"\n  --- {title} ---")


def show(label: str, value: object) -> None:
    print(f"    {label:<46} {value}")


def verdict(ok: bool, text: str) -> None:
    print(f"    {'PASS' if ok else '*** LEAK ***':<12} {text}")
