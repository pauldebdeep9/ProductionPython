# Production Python Core

A compact, hands-on repository covering the highest-ROI engineering patterns for
turning Python and AI experiments into reliable services. It prioritizes practical
familiarity and reusable mental models over exhaustive framework coverage.

This is a learning and reference repository—not a production framework, a complete
backend curriculum, or a reusable Python library. Each numbered file is a small,
independently runnable tutorial. Topic 8 integrates the ideas into one
production-shaped, offline AI service.

## Eight questions to remember

1. How does configuration enter the application? — `01_config_secrets.py`
2. Where is untrusted data validated? — `02_validation_typing.py`
3. How is slow I/O handled efficiently? — `03_async_concurrency.py`
4. What happens when dependencies fail? — `04_failures_retries.py`
5. How can production behavior be observed? — `05_observability.py`
6. How is important behavior tested? — `06_testing_patterns.py`
7. How does Python become an HTTP service? — `07_service_api.py`
8. How do all these concepts work together? — `08_ai_service_capstone.py`

## Topic map

| File | Core topic | Main ideas |
| --- | --- | --- |
| `01_config_secrets.py` | Configuration | Environment variables, settings, secrets, fail-fast validation |
| `02_validation_typing.py` | Validation and typing | Pydantic boundaries, `Protocol`, typed result variants |
| `03_async_concurrency.py` | Async I/O | `async`/`await`, `TaskGroup`, semaphore, timeout, cancellation |
| `04_failures_retries.py` | Resilience | Failure classification, retry, backoff, timeout, fallback |
| `05_observability.py` | Observability | Structured logs, correlation, latency, metrics, traces |
| `06_testing_patterns.py` | Testing | pytest, fakes, fixtures, contracts, AI evaluation distinction |
| `07_service_api.py` | HTTP service | FastAPI, lifespan, health/readiness, safe error mapping |
| `08_ai_service_capstone.py` | Integration | A production-shaped AI service combining Topics 1–7 |

Supporting files have deliberately narrow roles:

| File | Purpose |
| --- | --- |
| `test_core.py` | Consolidated high-value repository contracts |
| `pyproject.toml` | Python metadata plus runtime/development dependencies |
| `.env.example` | Safe configuration-key template |
| `Dockerfile` | Capstone runtime-image recipe |
| `.dockerignore` | Files excluded from the Docker build context |

## Recommended review sequence

Read the topics in order:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

- **01–02:** establish configuration, trust boundaries, and correctness.
- **03–04:** handle external I/O, resource limits, and failure behavior.
- **05–06:** make behavior observable and verify its important contracts.
- **07:** expose those contracts through a managed HTTP-service boundary.
- **08:** combine the complete model in one application.

For a quick revisit after several months, use:

```text
01 → 03 → 04 → 05 → 07 → 08
```

Return to Topics 2 and 6 when reviewing schemas, typing, or test design.

## Local environment

The repository assumes the existing development environment:

```bash
conda activate Sai2608
python --version
```

The repository does not create or own this environment. `Sai2608` is the local
developer environment; Docker creates a separate application runtime. Installing a
package locally does not change an image that has already been built.

## Running the tutorials

Every numbered module runs independently. For example:

```bash
python 01_config_secrets.py
python 03_async_concurrency.py
python 08_ai_service_capstone.py
```

Run any other topic by substituting its filename. These default executions use
deterministic local fakes and make no real provider calls.

Topic 6 doubles as a runnable explanation and a pytest-collectible example:

```bash
python 06_testing_patterns.py
python -m pytest -q 06_testing_patterns.py
```

## Repository validation

Run the consolidated contracts:

```bash
python -m pytest -q test_core.py
```

Run Topic 6's tutorial tests separately, or run everything together:

```bash
python -m pytest -q 06_testing_patterns.py
python -m pytest -q test_core.py 06_testing_patterns.py
```

The current scale is 25 consolidated cases plus 12 Topic 6 cases: 37 passing tests
when collected together. `python -m pytest` deliberately uses pytest from the
active Python environment.

## Configuration and secrets

`.env.example` lists the supported configuration keys. It is a safe template;
`.env` and environment-specific variants are local configuration and are ignored
by Git and Docker. Never commit real credentials.

The examples demonstrate environment-driven settings, typed validation, safe
secret display, and fail-fast startup. Deployment configuration should normally be
injected by the deployment platform. The current capstone uses a fake model and
requires no real LLM key.

## Integrated capstone

`08_ai_service_capstone.py` is a manufacturing-operations knowledge assistant. Its
small in-memory knowledge base contains `production-log-17`,
`maintenance-plan-03`, and `production-plan-08`.

```text
HTTP request
      ↓
Pydantic request validation
      ↓
retrieval
      ↓
bounded model gateway
      ↓
timeout and retry policy
      ↓
structured model-output validation
      ↓
safe HTTP response
```

The surrounding service also demonstrates structured logging, request IDs, basic
metrics, application lifespan, and distinct health/readiness endpoints.

Run the console tutorial:

```bash
python 08_ai_service_capstone.py
```

Start the local API, which safely defaults to `127.0.0.1:8000`:

```bash
python 08_ai_service_capstone.py --serve
```

- API documentation: `http://127.0.0.1:8000/docs`
- `GET /health` — process liveness
- `GET /ready` — dependency readiness
- `POST /answer` — validated question/answer boundary

Example request:

```bash
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question":"Why is Line 2 delayed?"}' \
  http://127.0.0.1:8000/answer
```

Two deterministic questions expose important failure policies:

- `simulate transient failure` → retry a temporary infrastructure failure.
- `simulate bad output` → reject invalid model structure with a safe HTTP 502.

Malformed model output is a schema failure, not automatically another transport
failure to retry.

## Packaging model

The numbered files are standalone applications rather than importable library
modules. The project therefore uses metadata-only packaging and intentionally has
no `src/` hierarchy or `__init__.py`. This command installs project metadata and
runtime dependencies without packaging the tutorial files:

```bash
python -m pip install .
```

Dependency responsibilities remain separate:

| Runtime | Development/testing |
| --- | --- |
| FastAPI | pytest |
| Pydantic Settings | HTTPX |
| Uvicorn | — |

Ruff, mypy, and Pyright are existing environment tools rather than declared
project dependencies.

`pyproject.toml` describes Python project metadata and dependencies. `Dockerfile`
describes construction of the operating-system and application runtime. Neither
replaces the other.

The declared dependency ranges are bounded but are not a fully locked dependency
graph. This repository teaches runtime construction, not supply-chain locking.

## Docker

```text
Dockerfile   = image recipe
docker build = create an image
Docker image = packaged immutable runtime artifact
docker run   = create a running container from that image
```

Build and run the capstone:

```bash
docker build -t production-python-core .
docker run --rm -p 8000:8000 production-python-core
```

Then probe it from another terminal:

```bash
curl http://127.0.0.1:8000/health
```

The local Python server defaults to `127.0.0.1:8000`. Inside the image,
`APP_HOST=0.0.0.0` and `APP_PORT=8000` allow Docker's published port to reach
Uvicorn. `EXPOSE 8000` documents the container port; `-p 8000:8000` performs the
actual host-to-container mapping. A running Docker daemon is required to build or
run the provided, statically validated Dockerfile.

```text
Git repository
      ↓ docker build
required metadata and capstone source copied into image
      ↓ docker run
container executes the copied application
```

The running container does not need access to the Git repository. One image can
start multiple independent containers. Supply runtime settings with Docker's `-e`
option; real deployments should inject secrets through their platform rather than
embedding credentials in the image.

## Key production principles

- Validate untrusted data at system boundaries.
- Keep configuration external, typed, and fail-fast; never log secrets.
- Async improves I/O concurrency, not CPU-bound Python performance.
- Bound downstream concurrency to protect providers and the service itself.
- Retry transient failures only, and always bound attempts and time.
- Treat malformed model output separately from infrastructure failure.
- Emit structured, correlated events without raw sensitive payloads.
- Health and readiness answer different operational questions.
- Use deterministic fakes for software tests; avoid real provider calls.
- AI quality evaluations complement rather than replace software tests.
- A container runs files copied into its image, not the source Git repository.

## Intentionally out of scope

Real LLM integrations, full RAG/vector databases, LangChain/LlamaIndex,
authentication, persistence, queues/workers, observability backends, Kubernetes,
cloud deployment, and CI/CD are omitted intentionally. Keeping those concerns out
preserves the small, reusable production-engineering mental model.
