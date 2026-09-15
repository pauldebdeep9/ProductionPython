# The commands that shape the developer loop. Worth committing — an
# undocumented pytest invocation gets typed differently by every person.

.PHONY: test test-fast test-all test-eval test-security cov mutants

test-fast:            ## every save: unit only, <1s
	pytest -m unit -q

test:                 ## pre-push: everything except slow evals
	pytest -m "not slow" -q

test-all:             ## CI: everything, including evals
	pytest -q

test-eval:            ## nightly: quality evals, verbose so k/n is visible
	pytest -m eval -v -s

test-security:        ## these must NEVER be skipped
	pytest -m security -q --strict-markers

cov:                  ## per-file, term-missing — find UNTOUCHED branches
	pytest --cov=rag --cov-report=term-missing -q

live:                 ## costs money; nightly only
	pytest -m eval --run-live -v -s

mutants:              ## the metric that actually measures test strength
	@echo "pip install mutmut && mutmut run --paths-to-mutate rag/"
