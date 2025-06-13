.PHONY: tests

-include .env
export

package-build:
	rm -rf dist/*
	uv build --no-sources

package-publish:
	uv publish --token ${UV_PUBLISH_TOKEN}

package: package-build package-publish

####
# project commands
#
# `uv add` to add new package
# `uv add --dev` to add new package as a dev dependency
####
# commands to run inside docker container
linting:
	uv run ruff check src/sik_llm_eval/
	uv run ruff check tests/

unittests:
	# pytest tests
	uv run coverage run -m pytest --durations=0 tests
	uv run coverage html

doctests:
	# python -m doctest sik_llm_eval/evals.py

tests: linting unittests doctests

####
# examples
####
run_nvidia_ragbench:
	# Run Nvidia RAGBench example
	python -m examples.rag_evals_via_nvidia_chatrag_bench
