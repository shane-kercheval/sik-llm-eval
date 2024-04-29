.PHONY: tests

####
# docker commands
####
docker_build:
	# build the docker container used to run tests and build package
	docker compose -f docker-compose.yml build

docker_run: docker_build
	# run the docker container
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	# rebuild docker container
	docker compose -f docker-compose.yml build --no-cache

####
# project commands
####
# commands to run inside docker container
linting:
	ruff check llm_eval/
	ruff check tests/

unittests:
	# pytest tests
	coverage run -m pytest --durations=0 tests
	coverage html

doctests:
	# python -m doctest llm_eval/evals.py

tests: linting unittests doctests

####
# dev commands
####
# commands for development
CONDA_ENV ?= ./env
env := $(CONDA_ENV)

setup_env:  ## Setup local environment
	@if [ -z "$${CONDA_SHLVL:+x}" ]; then echo "Conda is not installed." && exit 1; fi
	conda env $(shell [ -d $(env) ] && echo update || echo create) -p $(env) --file environment.yml