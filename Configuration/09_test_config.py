"""
09 — Testing configuration and secrets.

Run:  pytest 09_test_config.py -v

WHAT IS WORTH TESTING
---------------------
  1. THAT BAD CONFIG IS REJECTED. Not that good config works — that is the
     happy path everyone tests. The value is in proving each guard fires.
  2. THAT SECRETS DO NOT LEAK, using a scanner over real emitted output.
     This is the only way "we are careful about logging" becomes a control.
  3. THAT DEFAULTS ARE SAFE. Defaults are what runs when every other layer
     fails, so assert on them directly.
  4. THAT PRECEDENCE HOLDS. Layer ordering is easy to break in a refactor and
     invisible until an override silently stops working.

WHAT IS NOT WORTH TESTING
-------------------------
  * That pydantic parses "8" as 8. That is pydantic's test suite.
  * That your settings class has the fields you gave it.

THE THING THAT MAKES OR BREAKS THIS FILE
----------------------------------------
ENVIRONMENT ISOLATION. os.environ is process-global. One test that forgets to
clean up changes the behaviour of every test after it, in an order-dependent
way that only reproduces in CI. Every test below either uses `monkeypatch` or
the `clean_env` context manager, and there is a test that asserts the
isolation itself works.
"""

from __future__ import annotations

# Import the capstone's settings by loading it as a module.
import importlib.util
import json
import logging
import os
import pathlib
import sys
from decimal import Decimal

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from config_lab import (
    FAKE_API_KEY,
    FAKE_DB_PASSWORD,
    FakeCredentialLink,
    FakeDefaultAzureCredential,
    FakeKeyVault,
    LeakDetector,
    clean_env,
    env,
)

_spec = importlib.util.spec_from_file_location(
    "capstone", pathlib.Path(__file__).parent / "08_capstone.py")
assert _spec and _spec.loader
capstone = importlib.util.module_from_spec(_spec)
sys.modules["capstone"] = _spec.loader and capstone  # type: ignore[assignment]
sys.modules["capstone"] = capstone
_spec.loader.exec_module(capstone)

Settings = capstone.Settings
DISPOSITION_PROMPT = capstone.DISPOSITION_PROMPT


GOOD_ENV = {
    "ISC__ENVIRONMENT": "production",
    "ISC__LOG_LEVEL": "INFO",
    "ISC__ALLOWED_GROUPS": "isc-all",
    "ISC__OPENAI__ENDPOINT": "https://isc-aoai-prod.openai.azure.com/",
    "ISC__OPENAI__DEPLOYMENT": "gpt-4o-mini-prod",
    "ISC__SEARCH__ENDPOINT": "https://isc-search.search.windows.net/",
    "ISC__SEARCH__INDEX_NAME": "isc-docs-v3",
}


@pytest.fixture
def clean_isc_env():
    """Remove every ISC__ variable for the duration of a test.

    WITHOUT THIS, a developer with real ISC__ variables exported gets
    different results than CI — and the tests that pass on their machine are
    testing their config, not the code.
    """
    with clean_env("ISC__"):
        yield


@pytest.fixture
def detector() -> LeakDetector:
    d = LeakDetector()
    d.register("API_KEY", FAKE_API_KEY)
    d.register("DB_PASSWORD", FAKE_DB_PASSWORD)
    return d


# ===========================================================================
# 1. THE ISOLATION ITSELF
# ===========================================================================

def test_env_helper_restores_state() -> None:
    """Test the test harness. If isolation is broken, every other result in
    this file is untrustworthy, so assert it directly rather than assuming."""
    os.environ["ISC_TEST_MARKER"] = "original"
    with env(ISC_TEST_MARKER="overridden", ISC_TEST_NEW="added"):
        assert os.environ["ISC_TEST_MARKER"] == "overridden"
        assert os.environ["ISC_TEST_NEW"] == "added"
    assert os.environ["ISC_TEST_MARKER"] == "original"
    assert "ISC_TEST_NEW" not in os.environ
    del os.environ["ISC_TEST_MARKER"]


def test_env_helper_distinguishes_unset_from_empty() -> None:
    """The trap from script 01: "" is SET, and `.get(k, default)` returns it
    rather than the default."""
    with env(ISC_TEST_EMPTY=""):
        assert os.environ.get("ISC_TEST_EMPTY") == ""
        assert os.environ.get("ISC_TEST_EMPTY", "fallback") == ""
    with env(ISC_TEST_EMPTY=None):
        assert os.environ.get("ISC_TEST_EMPTY", "fallback") == "fallback"


# ===========================================================================
# 2. BAD CONFIG IS REJECTED — the highest-value tests here
# ===========================================================================

@pytest.mark.parametrize(
    "overrides,expected_fragment",
    [
        ({"ISC__OPENAI__API_KEY": FAKE_API_KEY}, "must not be set in production"),
        ({"ISC__LOG_LEVEL": "DEBUG"}, "DEBUG logging in production"),
        ({"ISC__ALLOWED_GROUPS": ""}, "allowed_groups must be non-empty"),
        ({"ISC__OPENAI__API_KEY": "@Microsoft.KeyVault(SecretUri=x)",
          "ISC__ENVIRONMENT": "staging"}, "did not resolve"),
        ({"ISC__OPENAI__MODEL": "gpt-4o"}, "exceeding the cap"),
        ({"ISC__SEARCH__TOP_K": "20"}, "exceeding 80%"),
        ({"ISC__OPENAI__ENDPOINT": "https://api.openai.com/v1"},
         "not an Azure OpenAI endpoint"),
        ({"ISC__OPENAI__DEPLOYMENT": ""}, "at least 1 character"),
        ({"ISC__SEARCH__INDEX_NAME": "Invalid_Name"}, "match pattern"),
        ({"ISC__SEARCH__TOPK": "8"}, "Extra inputs are not permitted"),
    ],
    ids=["prod_api_key", "prod_debug_log", "prod_no_groups",
         "unresolved_kv_ref", "cost_cap_inconsistent", "context_overflow",
         "wrong_endpoint", "empty_deployment", "bad_index_name",
         "typo_in_var_name"],
)
def test_bad_config_is_rejected(clean_isc_env, overrides: dict[str, str],
                                expected_fragment: str) -> None:
    """Each row is a production incident that cannot now happen.

    ASSERTING ON THE MESSAGE FRAGMENT, not just that something raised, is
    what makes this meaningful — it proves the RIGHT guard fired. A test that
    only checks `pytest.raises(ValidationError)` passes when the config is
    rejected for an unrelated reason, which is how a guard silently stops
    working.
    """
    with env(**{**GOOD_ENV, **overrides}):
        with pytest.raises(ValidationError) as exc:
            Settings()
        assert expected_fragment in str(exc.value), (
            f"expected {expected_fragment!r} in the error, got: {exc.value}"
        )


def test_all_errors_reported_at_once(clean_isc_env) -> None:
    """The argument from script 02: one report, not one problem per deploy."""
    with env(
        ISC__ENVIRONMENT="production",
        ISC__LOG_LEVEL="TRACE",
        ISC__OPENAI__ENDPOINT="https://api.openai.com/v1",
        ISC__OPENAI__DEPLOYMENT="",
        ISC__SEARCH__ENDPOINT="not-a-url",
        ISC__SEARCH__INDEX_NAME="BAD",
    ):
        with pytest.raises(ValidationError) as exc:
            Settings()
        assert exc.value.error_count() >= 4, (
            "validation should report every problem, not stop at the first"
        )


def test_good_config_is_accepted(clean_isc_env) -> None:
    """The negative control for the parametrised test above. Without it, a
    settings class that rejected EVERYTHING would pass all ten rows."""
    with env(**GOOD_ENV):
        s = Settings()
        assert s.environment.value == "production"
        assert s.uses_managed_identity is True


# ===========================================================================
# 3. SECRETS DO NOT LEAK
# ===========================================================================

class ConfigWithSecrets(BaseModel):
    endpoint: str
    api_key: SecretStr
    db_password: SecretStr


@pytest.fixture
def secret_config() -> ConfigWithSecrets:
    return ConfigWithSecrets(
        endpoint="https://isc-aoai-prod.openai.azure.com/",
        api_key=SecretStr(FAKE_API_KEY),
        db_password=SecretStr(FAKE_DB_PASSWORD),
    )


@pytest.mark.parametrize(
    "surface",
    ["repr", "str", "json", "log", "traceback", "span"],
)
def test_secret_does_not_leak(detector: LeakDetector,
                              secret_config: ConfigWithSecrets,
                              surface: str) -> None:
    """One test per leak surface from script 04.

    Parametrising by SURFACE means a failure names which one broke, which is
    the difference between "a secret leaked somewhere" and "the span
    exporter started including config".
    """
    log_obj = logging.getLogger("test.leak")

    if surface == "repr":
        detector.scan(repr(secret_config), surface)
    elif surface == "str":
        detector.scan(str(secret_config), surface)
    elif surface == "json":
        detector.scan(secret_config.model_dump_json(), surface)
    elif surface == "log":
        with detector.capture_logs("test.leak"):
            log_obj.info("config: %s", secret_config)
            log_obj.debug("key: %s", secret_config.api_key)
    elif surface == "traceback":
        try:
            raise ConnectionError(f"failed to reach {secret_config.endpoint}")
        except ConnectionError as e:
            detector.scan_exception(e)
    elif surface == "span":
        span = {"name": "req", "attributes": {
            "endpoint": secret_config.endpoint,
            "has_key": secret_config.api_key is not None}}
        detector.scan(json.dumps(span), surface)

    assert not detector.leaks, detector.report()


def test_leak_detector_would_catch_a_real_leak(detector: LeakDetector) -> None:
    """THE NEGATIVE CONTROL, and the most important test in this file.

    Without it, the six tests above could pass because the detector is
    broken, because the secrets were never registered, or because the
    fixture's values do not match. Proving the detector fires on a genuine
    leak is what makes the passing tests mean anything.
    """
    naked = f"connecting with api_key={FAKE_API_KEY}"
    leaks = detector.scan(naked, "deliberate")
    assert leaks, "the detector failed to catch a plain-text secret"
    assert leaks[0].secret_name == "API_KEY"


def test_detector_catches_encoded_forms(detector: LeakDetector) -> None:
    """A secret in a URL is percent-encoded and a naive scan misses it."""
    import urllib.parse
    url = f"https://x/?api-key={urllib.parse.quote(FAKE_API_KEY, safe='')}"
    assert detector.scan(url, "url"), "encoded secret evaded the scan"


def test_settings_diagnostics_are_redacted(clean_isc_env,
                                           detector: LeakDetector) -> None:
    """The diagnostics endpoint is a deliberate dump of config. It must be
    safe by construction, because its whole purpose is to be read."""
    with env(**{**GOOD_ENV, "ISC__ENVIRONMENT": "staging",
                "ISC__OPENAI__API_KEY": FAKE_API_KEY}):
        s = Settings()
        detector.scan(json.dumps(s.redacted()), "redacted()")
        detector.scan(s.model_dump_json(), "model_dump_json")
        detector.scan(repr(s), "repr")
    assert not detector.leaks, detector.report()


def test_denylist_redaction_is_insufficient(detector: LeakDetector) -> None:
    """Pins the finding from script 04: name-based redaction misses fields
    whose names contain no suspicious word.

    This test asserts a WEAKNESS, which is unusual and deliberate — it exists
    so that if someone replaces type-based redaction with a denylist, a test
    documents why that is a downgrade.
    """
    suspicious = ("key", "secret", "password", "token")
    payload = {"sas_url": f"https://blob/?sig={FAKE_API_KEY}"}
    redacted = {k: ("***" if any(s in k.lower() for s in suspicious) else v)
                for k, v in payload.items()}
    assert detector.scan(json.dumps(redacted), "denylist"), (
        "expected the denylist to MISS this; if it now catches it, the test "
        "is stale"
    )


# ===========================================================================
# 4. DEFAULTS ARE SAFE
# ===========================================================================

def test_defaults_are_safe(clean_isc_env) -> None:
    """Defaults are what runs when every other layer fails to load, so assert
    on them explicitly rather than trusting they were chosen carefully."""
    minimal = {
        "ISC__OPENAI__ENDPOINT": "https://x.openai.azure.com/",
        "ISC__OPENAI__DEPLOYMENT": "d",
        "ISC__SEARCH__ENDPOINT": "https://y.search.windows.net/",
        "ISC__SEARCH__INDEX_NAME": "idx",
    }
    with env(**minimal):
        s = Settings()
        assert s.environment.value == "local", "must not default to production"
        assert s.log_level != "DEBUG", "must not default to DEBUG"
        assert s.allowed_groups == [], "must not default to a permissive list"
        assert s.limits.daily_spend_cap_usd <= Decimal("100"), (
            "the default spend cap must be conservative"
        )
        assert s.openai.api_key is None, "must not default to a key"


# ===========================================================================
# 5. PRECEDENCE
# ===========================================================================

def test_env_overrides_default(clean_isc_env) -> None:
    with env(**GOOD_ENV):
        assert Settings().search.top_k == 5
    with env(**GOOD_ENV, ISC__SEARCH__TOP_K="3"):
        assert Settings().search.top_k == 3


def test_nested_delimiter_maps_correctly(clean_isc_env) -> None:
    """A refactor that changes env_nested_delimiter silently stops every
    nested variable from applying — with no error, because the settings just
    keep their defaults."""
    with env(**GOOD_ENV, ISC__LIMITS__MAX_CONCURRENCY="42"):
        assert Settings().limits.max_concurrency == 42


# ===========================================================================
# 6. CREDENTIALS
# ===========================================================================

def test_credential_is_cached_across_calls() -> None:
    """The 100x amplification from script 05, as an assertion."""
    cred = FakeDefaultAzureCredential(
        [FakeCredentialLink("ManagedIdentityCredential", available=True)])
    for _ in range(50):
        cred.get_token("https://cognitiveservices.azure.com/.default")
    assert cred.links[0].calls == 1, (
        f"expected 1 identity-endpoint call, got {cred.links[0].calls}"
    )
    assert cred.cache_hits == 49


def test_tokens_are_cached_per_scope() -> None:
    cred = FakeDefaultAzureCredential(
        [FakeCredentialLink("ManagedIdentityCredential", available=True)])
    cred.get_token("scope-a")
    cred.get_token("scope-b")
    cred.get_token("scope-a")
    assert cred.links[0].calls == 2, "a token for one scope is not valid for another"


def test_access_token_repr_hides_the_token() -> None:
    cred = FakeDefaultAzureCredential(
        [FakeCredentialLink("ManagedIdentityCredential", available=True)])
    token = cred.get_token("scope")
    assert "eyJ" not in repr(token), "a bearer token must never be rendered"
    assert "expires_in" in repr(token)


# ===========================================================================
# 7. PROMPTS AS CONFIG
# ===========================================================================

def test_missing_prompt_variable_fails_loudly() -> None:
    """The silent-failure guard from script 07: an unrendered {context}
    produces a fluent, ungrounded answer with no error anywhere."""
    with pytest.raises(ValueError, match="missing"):
        DISPOSITION_PROMPT.render(question="why?")


def test_prompt_fingerprint_changes_with_the_text() -> None:
    """The fingerprint is what makes "which prompt produced this answer?"
    answerable. Assert it is actually sensitive to the content."""
    original = DISPOSITION_PROMPT.fingerprint
    edited = capstone.Prompt(
        DISPOSITION_PROMPT.name, DISPOSITION_PROMPT.version,
        DISPOSITION_PROMPT.template + " Be concise.",
        DISPOSITION_PROMPT.required)
    assert edited.fingerprint != original


# ===========================================================================
# 8. CONFIG DRIFT
# ===========================================================================

def test_fingerprint_is_stable_and_sensitive(clean_isc_env) -> None:
    """Stable across identical config, different when anything changes.
    Both halves matter: an unstable fingerprint is noise, an insensitive one
    is useless."""
    with env(**GOOD_ENV):
        a, b = Settings().fingerprint, Settings().fingerprint
    assert a == b, "fingerprint must be stable for identical config"

    with env(**GOOD_ENV, ISC__SEARCH__TOP_K="3"):
        c = Settings().fingerprint
    assert c != a, "fingerprint must change when config changes"


def test_fingerprint_does_not_depend_on_secret_values(clean_isc_env) -> None:
    """A fingerprint computed over redacted config is safe to log AND stays
    stable across a secret rotation — so a rotation does not look like a
    config change in your drift dashboard."""
    base = {**GOOD_ENV, "ISC__ENVIRONMENT": "staging"}
    with env(**base, ISC__OPENAI__API_KEY="key-one"):
        f1 = Settings().fingerprint
    with env(**base, ISC__OPENAI__API_KEY="key-two"):
        f2 = Settings().fingerprint
    assert f1 == f2, (
        "the fingerprint changed when only a secret VALUE changed; it is "
        "being computed over unredacted config"
    )


# ===========================================================================
# 9. ROTATION
# ===========================================================================

@pytest.mark.asyncio
async def test_cache_invalidation_recovers_from_rotation() -> None:
    """Script 06: recovery must be immediate, not TTL-bound."""
    vault = FakeKeyVault()
    vault.set_secret("k", "v1")
    cache = capstone.SecretCache(vault, ttl=1000.0)   # effectively never expires

    assert await cache.get("k") == "v1"
    vault.set_secret("k", "v2")
    assert await cache.get("k") == "v1", "still cached, as expected"

    cache.invalidate()
    assert await cache.get("k") == "v2", "invalidation must force a re-read"


@pytest.mark.asyncio
async def test_secret_refresh_is_single_flight() -> None:
    """50 concurrent cold-start misses must collapse to one vault call."""
    import asyncio
    vault = FakeKeyVault()
    vault.set_secret("k", "v1")
    cache = capstone.SecretCache(vault, ttl=30.0)
    await asyncio.gather(*(cache.get("k") for _ in range(50)))
    assert cache.fetches == 1, (
        f"expected 1 vault fetch, got {cache.fetches} — a stampede against a "
        f"rate-limited vault"
    )
