
"""Classes and functions to evaluate LLMs."""

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from textwrap import dedent
from typing import Callable
from collections.abc import Iterator
from llm_eval.candidates import Candidate, CandidateResponse, is_async_candidate
from llm_eval.checks import (
    Check,
    CheckResult,
    PassFailResult,
    ResponseModel,
)
from llm_eval.internal_utilities import DictionaryEqualsMixin, SerializationMixin


class Eval(SerializationMixin, DictionaryEqualsMixin):
    """
    An Eval defines single test-case/scenario that the user is interested in evaluating. It
    typically includes both the input for the LLM or agent, and the tests (called "checks") that
    define the evaluation criteria. The `input` is optional.

    When using the Eval object directly, by passing the response (from the LLM/agent) to the
    object's `__call__` method, the `input` is not directly used. The Eval object will run the
    checks against the response and will return an EvalResult object.

    However, when running evaluations using the EvalHarness, the Eval object is usually not called
    directly by the user. Instead, the EvalHarness will pass the `input` defined in the Eval object
    to the Candidate (which is a light wrapper around the LLM/Agent) and the harness will pass the
    response from the Candidate back to the Eval object.

    NOTE: This decouples the Candidate from the Eval object, allowing Evals to be used against
    pre-generated responses. This also allows a future implementation of the EvalHarness to run
    Evals against pre-generated responses, which will be useful for rerunning previous evaluations
    against new Checks or, for example, rerunning evaluations after bug fixes in a Check.
    """

    def __init__(
            self,
            checks: list[Check | dict | Callable[[object], CheckResult]],
            input: str | dict | list | object | None = None,  # noqa: A002
            ideal_response: str | list | None = None,
            metadata: dict | None = None,
        ) -> None:
        """
        Initializes the Eval.

        Args:
            checks:
                A list of Check objects, dictionaries, or callables. If a dictionary is passed in,
                the Check subclasses need to be registered via `Check.register(...)`, and the
                dictionary needs a `check_type` key with the registration value.
            input:
                The input that was sent to the LLM/Agent. This can be a string, dictionary,
                list, or any other object that the LLM/Agent expects.

                This argument is optional since an Eval can be called directly with a response. The
                `input` argument can be used A) to store the input for reference, or B) input will
                be passed to the Candidate object if used in the EvalHarness.
            ideal_response:
                The ideal response or list of ideal responses that the Candidate/LLM should return.
                This is optional and not currently used in the evaluation process, but it can be
                used in user-defined checks and candidates.
            metadata:
                Metadata associated with the Eval.
        """
        self.input = input
        self.ideal_response = ideal_response
        self.metadata = deepcopy(metadata) or {}
        self.checks = Eval._create_checks(checks)

    @staticmethod
    def _create_checks(
            checks: list[Check | dict | Callable[[object], CheckResult]],
        ) -> list[Check]:
        """Converts a list of checks to Check objects."""
        checks = checks or []
        if not isinstance(checks, list):
            checks = [checks]
        checks_created = []
        for check in checks:
            if isinstance(check, dict):
                assert 'check_type' in check, "Check dictionary must contain a 'check_type' key"
                checks_created.append(Check.from_dict(check))
            elif isinstance(check, (Callable, Check)):
                checks_created.append(deepcopy(check))
            else:
                raise TypeError("Checks must be either a Check, dictionary, or callable.")
        return checks_created

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Eval."""
        value = {}
        if self.metadata:
            value['metadata'] = deepcopy(self.metadata)
        if self.input:
            value['input'] = deepcopy(self.input)
        value['checks'] = [
            c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in self.checks
        ]
        if self.ideal_response:
            value['ideal_response'] = deepcopy(self.ideal_response)
        return value

    @classmethod
    def from_dict(cls: 'Eval', value: dict) -> 'Eval':
        """Create a new instance from a dictionary."""
        return cls(**value)

    def __call__(
            self,
            response: object,
            metadata: dict | None = None,
            candidate: Candidate | dict | None = None,
            ) -> 'EvalResult':
        """
        Evaluates the model against the prompts and tests.

        Args:
            response: The response from the model.
            metadata: Metadata associated with the response.
            candidate:
                The candidate object that was evaluated. This is an optional arugment and the
                candidate object is simply added to the EvalResult object for reference and, for
                exmaple, will be converted to a dictionary and saved with the EvalResult if the
                EvalResult is saved to a file.
        """
        response_model = ResponseModel(
            input=self.input,
            ideal_response=self.ideal_response,
            response=response,
            metadata=metadata,
        )
        check_results = [
            check.run_on_model(response_model) if isinstance(check, Check) else check(response_model)  # noqa: E501
            for check in self.checks
        ]
        return EvalResult(
            eval=self,
            candidate=candidate,
            response=response,
            metadata=metadata,
            check_results=check_results,
        )

    def __str__(self) -> str:
        """Returns a string representation of the Eval."""
        metadata = '' if not self.metadata else f'            metadata={self.metadata},\n{" " * 12}'  # noqa
        return dedent(f"""
        Eval(
            {metadata}input={self.input},
        )
        """).strip()


class EvalResult(SerializationMixin, DictionaryEqualsMixin):
    """
    An EvalResult is the result of evaluating a specific Candidate/LLM against a specific
    Eval.
    """

    def __init__(
        self,
        eval: Eval | dict,  # noqa: A002
        candidate: Candidate | dict | Callable,
        response: str | object,
        metadata: dict,
        check_results: list[CheckResult | dict],
        timestamp: str | None = None,
        ) -> None:
        """
        Initializes the EvalResult.

        Args:
            eval:
                The Eval object that was evaluated.
            candidate:
                The Candidate object that was evaluated. If the Candidate is a dictionary, the
                Candidate subclasses need to be registered via `Candidate.register(...)`.
                The dictionary needs a `candidate_type` key with the registration value.
            response:
                The response from the Candidate (e.g. LLM/agent).
            metadata:
                Metadata associated with the response (e.g. the response.metadata from the
                CandidateResponse object).
            timestamp:
                The timestamp when the Eval was completed.
            check_results:
                A list of CheckResult objects.
        """
        self.eval = eval if isinstance(eval, Eval) else deepcopy(eval)
        if candidate is None:
            self.candidate = None
        elif isinstance(candidate, Candidate):
            self.candidate = candidate
        elif isinstance(candidate, dict):
            self.candidate = deepcopy(candidate)
        else:
            self.candidate = str(candidate)
        self.response = response
        self.metadata = metadata
        self.timestamp = timestamp if timestamp else datetime.now(timezone.utc).isoformat()
        results = check_results or []
        results_created = []
        # convert dictionaries to CheckResults
        for r in results:
            if isinstance(r, dict):
                assert 'result_type' in r, \
                    "CheckResult dictionary must contain a 'result_type' key"
                results_created.append(CheckResult.from_dict(r))
            elif isinstance(r, CheckResult):
                results_created.append(r)
            elif isinstance(r, bool):
                # if a boolean is passed in, convert it to a CheckResult
                results_created.append(PassFailResult(value=r))
            else:
                raise TypeError("results must be a CheckResult, dictionary, or bool")
        self.check_results = results_created

    @property
    def num_checks(self) -> int:
        """Returns the number of checks."""
        return len(self.check_results)

    @property
    def num_successful_checks(self) -> int:
        """Returns the number of successful checks."""
        return sum(r.success for r in self.check_results if r.success)

    @property
    def perc_successful_checks(self) -> float | None:
        """Returns the percentage of passing checks. If there are checks, returns None."""
        return self.num_successful_checks / self.num_checks if self.num_checks else None

    def to_dict(self) -> dict:
        """Return a dictionary representation of the EvalResult."""
        if self.candidate is None:
            candidate = None
        elif isinstance(self.candidate, Candidate):
            candidate = self.candidate.to_dict()
        elif isinstance(self.candidate, dict):
            candidate = self.candidate
        else:
            candidate = str(self.candidate)
        return {
            'eval': self.eval if isinstance(self.eval, dict) else self.eval.to_dict(),
            'candidate': candidate,
            'response': deepcopy(self.response),
            'metadata': deepcopy(self.metadata),
            'timestamp': self.timestamp,
            'check_results': [r.to_dict() for r in self.check_results],
        }

    @classmethod
    def from_dict(cls: 'Eval', value: dict) -> 'Eval':
        """Create a new instance from a dictionary."""
        return cls(**value)


@dataclass
class EvalRunResult:
    """Encapsulates the results of running an Eval."""

    eval: Eval  # The Eval object that was evaluated
    # The eval object is also included in EvalResult if both the response generation and eval
    # execution are successful. This is included here for reference in case there is an error.
    eval_result: EvalResult | None  # None if there was an error during eval execution
    response_error: Exception | None  # Error during response generation, if any
    eval_error: Exception | None  # Error during eval execution, if any


class CandidateRunResults:
    """Encapsulates the results of running one or more Evals against a single Candidate."""

    def __init__(self, candidate: Candidate, run_results: list[EvalRunResult]) -> None:
        """
        Initializes the CandidateRunResults.

        Args:
            candidate:
                The Candidate object that was evaluated.
            run_results:
                A list of EvalRunResult objects.
        """
        self.candidate = candidate
        self._num_errors = len([result for result in run_results if result.response_error or result.eval_error])  # noqa: E501
        self._response_errors = [result.response_error for result in run_results]
        self._eval_results = [result.eval_result for result in run_results]
        self._eval_errors = [result.eval_error for result in run_results]

    def __iter__(self) -> Iterator[tuple[EvalResult | None, Exception | None, Exception | None]]:
        """
        Iterates over the results, yielding tuples of (EvalResult, Exception (response eror), and
        Exception (eval_error)) for each index in the results lists.
        """
        return zip(self._eval_results, self._response_errors, self._eval_errors)

    @property
    def num_errors(self) -> int:
        """Returns the number of errors during response generation or eval execution."""
        return self._num_errors

    @property
    def eval_results(self) -> list[EvalResult | None]:
        """
        Returns a list of EvalResults. If there was an error during eval execution, the
        corresponding EvalResult will be None and will not be included in the list.
        """
        return self._eval_results

    @property
    def response_errors(self) -> list[Exception | None]:
        """Returns a list of exceptions for response generation errors."""
        return self._response_errors

    @property
    def eval_errors(self) -> list[Exception | None]:
        """Returns a list of exceptions for eval execution errors."""
        return self._eval_errors


class Mode(Enum):
    """
    Enumeration for different modes of execution.

    SYNC: Runs the corresponding tasks synchronously.
    ASYNC: Runs the tasks asynchronously using the event loop.
    PARALLEL: Runs the tasks in parallel using ProcessPoolExecutor across all CPU cores.
    """

    SYNC = auto()
    ASYNC = auto()
    PARALLEL = auto()


class EvalHarness:
    """
    An EvalHarness provides a interface for evaluating multiple Evals against multiple
    Candidates/LLMs.

    Candidates must be registered via Candidate.register if they are passed in as dictionaries
    (since the Candidates will be instantiated from the dictionary).

    The EvalHarness is responsible for generating the response from the Candidate (from the input
    defined in the Eval object) and running the Checks defined in the Eval against the response.
    The EvalHarness will return a list of CandidateRunResults objects, which encapsulate the
    results of running the Evals against the Candidates.

    Example usage:

        ```python
        from llm_eval import EvalHarness

        harness = EvalHarness(
            response_mode=Mode.ASYNC,
            eval_mode=Mode.PARALLEL,
            num_samples=5,  # optional
        )
        harness.add_evals_from_files('xxx/evals/')
        harness.add_candidates_from_files('xxx/candidates/')
        results = harness()
    ```

    """

    def __init__(
            self,
            evals: list[Eval | dict ] | Eval | dict | None = None,
            candidates: list[Candidate | Callable | dict] | Candidate | dict | None = None,
            num_samples: int = 1,
            response_mode: Mode = Mode.SYNC,
            eval_mode: Mode = Mode.SYNC,
            num_cpus: int | None = None,
            async_batch_size: int = 50,  # only applicable when using async mode
            ) -> None:
        """
        Initializes the EvalHarness. The user can either pass in Eval and Candidate objects in the
        constructor or call
            - `add_eval_from_file(...)`, which takes a path to a YAML/JSON file.
            - `add_evals_from_files(...)`, which takes a path to a directory of YAML/JSON files.
            - `add_candidate_from_file(...)`, which takes a path to a YAML/JSON file.
            - `add_candidates_from_files(...)`, which takes a path to a directory of YAML/JSON files.

        The methods above can be called multiple times to add additional Eval and Candidate
        objects.

        Example:

        ```
        harness = EvalHarness(
            num_cpus=None,
            async_batch_size=50,
            num_samples=5,  # optional; runs 5 samples for each Eval
            callback=print,  # optional
        )
        harness.add_eval_from_file('xxx/eval.yaml')
        harness.add_candidate_from_file('xxx/candidate.yaml')
        results = harness()
        ```

        Args:
            evals:
                A list of Eval objects or dictionaries. Alternatively, `add_eval...` methods can
                be called to add Eval objects.
            candidates:
                A list of Candidate objects or dictionaries. Alternatively, `add_candidate...`
                methods can be called to add Candidate objects.
                Candidates.
            num_samples:
                The number of samples to run for each Eval. This is useful for running multiple
                samples for each Eval to get a better estimate of the performance metrics. Running
                multiple samples only makes sense for a non-zero temperature for the LLM. A zero
                temperature (or close) will give the same or very similar responses for each
                sample, defeating the purpose of collecting a sample.

                `num_samples > 1` effectively copies the Eval object num_samples times and runs
                each copy against the Candidate. An id should be used in the Eval metadata to
                identify unique evals in order to aggregate accordingly.
            response_mode:
                The mode for generating responses. The mode can be set to `SYNC` (default),
                `ASYNC`, or `PARALLEL. The `SYNC` mode generates responses synchronously, the
                `ASYNC` mode generates responses asynchronously using the event loop, and the
                `PARALLEL` mode generates responses in parallel using ProcessPoolExecutor across
                all CPU cores.
            eval_mode:
                The mode for running evals. The mode can be set to `SYNC` (default), `ASYNC`, or
                `PARALLEL. The `SYNC` mode runs the Evals synchronously, the `ASYNC` mode runs the
                Evals asynchronously using the event loop, and the `PARALLEL` mode runs the Evals
                in parallel using ProcessPoolExecutor across all CPU cores.
            num_cpus:
                The number of CPUs to use for parallel processing. This parameters is only used
                if response_mode or eval_mode is set to `Mode.PARALLEL`.

                If num_cpus is None or less than 1, then the corresponding tasks will use all
                available CPUs.
            async_batch_size:
                The number of tasks (e.g. response generation) to run asynchronously at a time.
                This parameter is only used if response_mode or eval_mode is set to `Mode.ASYNC`.
        """  # noqa: D412, E501
        self.response_mode = response_mode
        self.eval_mode = eval_mode
        self.async_batch_size = async_batch_size
        self.num_cpus = num_cpus if num_cpus is not None and num_cpus >= 1 else os.cpu_count()
        self.num_samples = num_samples
        self.evals = []
        self.candidates = []
        if evals:
            self.add_evals(evals)
        if candidates:
            self.add_candidates(candidates)

    def add_evals(self, evals: list[Eval | dict ] | Eval | dict) -> None:
        """
        Adds one ore more Eval object. This method can be called multiple times to add additional
        Eval objects.

        Args:
            evals:
                The Eval object to add. If the Eval is a dictionary, the Check subclasses need to
                be registered via `Check.register(...)`.
                The checks needs a `check_type` key with the registration value.
        """
        if isinstance(evals, dict):
            self.evals.append(Eval(**evals))
        elif isinstance(evals, Eval):
            self.evals.append(evals)
        elif isinstance(evals, list):
            for obj in evals:
                self.add_evals(obj)
        else:
            raise TypeError(f"incompatible type {type(evals)} for eval")

    def add_eval_from_file(self, path: str) -> None:
        """
        Adds an Eval from a YAML file. This method can be called multiple times to add additional
        Eval objects.

        The underlying Check objects must be registered with the CheckRegistry before calling this
        method and the checks need a `check_type` key with the registration value.

        Args:
            path:
                Path to the YAML file.
        """
        self.add_evals(Eval.from_file(path))

    def add_evals_from_files(self, path: str) -> None:
        """
        Adds multiple Evals from a directory of yml, yaml, or json files. This method can be called
        multiple times to add additional Eval objects.

        The underlying Check objects must be registered with the CheckRegistry before calling this
        method and the checks need a `check_type` key with the registration value.

        Args:
            path:
                Path to the directory of YAML files along shell-style wildcards. For example,
                `path/to/directory/*.yaml` (passed to `glob.glob` function).

        """
        for file_path in glob.glob(path):
            self.add_eval_from_file(file_path)

    def add_candidates(
            self,
            candidate: list[Candidate | dict | Callable[[object], object]] | Candidate | dict | Callable[[object], object],  # noqa
            ) -> None:
        """
        Adds a Candidate object. This method can be called multiple times to add additional
        Candidate objects.

        Args:
            candidate:
                The Candidate object to add. If the Candidate is a dictionary, the Candidate
                subclasses need to be registered via `Candidate.register(...)`.
                The dictionary needs a `candidate_type` key with the registration value.
        """
        if isinstance(candidate, dict):
            # loads the Candidate subclass from the registry
            candidate = Candidate.from_dict(candidate)
            if isinstance(candidate, list):
                self.candidates.extend(candidate)
            else:
                self.candidates.append(candidate)
        elif isinstance(candidate, list):
            for obj in candidate:
                self.add_candidates(obj)
        elif isinstance(candidate, (Candidate, Callable)):
            self.candidates.append(candidate)
        else:
            raise TypeError(f"incompatible type {type(candidate)} for candidate")

    def add_candidate_from_file(self, path: str) -> None:
        """
        Adds a Candidate from a YAML file. This method can be called multiple times to add
        additional Candidate objects.

        The underlying Candidate objects must be registered with the CandidateRegistry before
        calling this method and the candidate needs a `candidate_type` key with the registration
        value.

        Args:
            path:
                Path to the YAML file.
        """
        self.add_candidates(Candidate.from_file(path))

    def add_candidates_from_files(self, path: str) -> None:
        """
        Adds multiple Candidates from a directory of YAML files. This method can be called
        multiple times to add additional Candidate objects.

        The underlying Candidate objects must be registered with the CandidateRegistry before
        calling this method and the candidate needs a `candidate_type` key with the registration
        value.

        Args:
            path:
                Path to the directory of YAML files.
        """
        for file_path in glob.glob(path):
            self.add_candidate_from_file(file_path)

    @staticmethod
    def _generate_single_response(
            candidate: Candidate,
            eval: Eval,  # noqa: A002
        ) -> tuple[CandidateResponse, Exception | None]:
        """
        Generates a response for a given candidate and eval, returning a tuple containing the
        response and any exception generated during the response generation. If no exception is
        generated, the exception will be None.
        """
        try:
            candidate = deepcopy(candidate)
            if is_async_candidate(candidate):
                candidate_response = candidate(eval.input)
            else:
                candidate_response = candidate(eval.input)
            return candidate_response, None
        except Exception as e:
            return None, e

    @staticmethod
    async def _generate_single_response_async(
            semaphore: asyncio.Semaphore,
            candidate: Candidate,
            eval: Eval,  # noqa: A002
        ) -> tuple[CandidateResponse, Exception | None]:
        """
        Generates a response for a given candidate and eval, returning a tuple containing the
        response and any exception generated during the response generation. If no exception is
        generated, the exception will be None.
        """
        async with semaphore:
            try:
                candidate = deepcopy(candidate)
                if is_async_candidate(candidate):
                    candidate_response = await candidate(eval.input)
                else:
                    # we need to run the synchronous candidate in a thread pool to avoid blocking
                    loop = asyncio.get_running_loop()
                    candidate_response = await loop.run_in_executor(
                        None,
                        lambda: candidate(eval.input),
                    )
                return candidate_response, None
            except Exception as e:
                return None, e

    @staticmethod
    def _run_single_eval(
            eval: Eval,  # noqa: A002
            candidate: Candidate,
            candidate_response: CandidateResponse,
        ) -> tuple[EvalResult, Exception | None]:
        """
        Runs an Eval object against a CandidateResponse object, returning a tuple containing the
        EvalResult and any exception generated during the eval execution. If no exception is
        generated, the exception will be None.
        """
        if candidate_response:
            response = candidate_response.response
            metadata = {
                'response_metadata': candidate_response.metadata,
                'response_timestamp': candidate_response.timestamp,
            }
        else:
            response = None
            metadata = {}
        try:
            eval_result = eval(
                response=response,
                metadata=metadata,
                candidate=candidate,
            )
            return eval_result, None
        except Exception as e:
            return None, e

    @staticmethod
    async def _run_single_eval_async(
            semaphore: asyncio.Semaphore,
            eval: Eval,  # noqa: A002
            candidate: Candidate,
            candidate_response: CandidateResponse,
        ) -> tuple[EvalResult, Exception | None]:
        """Helper function to run evals asynchronously."""
        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                EvalHarness._run_single_eval,
                eval,
                candidate,
                candidate_response,
            )

    async def _generate_all_responses(
            self,
            eval_copies: list[list[Eval]],
        ) -> list[list[tuple[CandidateResponse, Exception | None]]]:
        """
        Generates responses for all Evals and Candidates.

        Returns a list of lists of tuples containing the CandidateResponse and any exception. The
        outer list is indexed by candidate and the inner list is indexed by eval.
        """
        responses = []
        semaphore = asyncio.Semaphore(self.async_batch_size)
        if self.response_mode == Mode.SYNC:
            # Synchronous generation
            for candidate, candidate_evals in zip(self.candidates, eval_copies):
                candidate_responses = []
                for eval_ in candidate_evals:
                    candidate_response, response_error = await EvalHarness._generate_single_response_async(  # noqa: E501
                        semaphore=semaphore,
                        candidate=candidate,
                        eval=eval_,
                    )
                    candidate_responses.append((candidate_response, response_error))
                responses.append(candidate_responses)
        elif self.response_mode == Mode.ASYNC or any(is_async_candidate(candidate) for candidate in self.candidates):  # noqa: E501
            # Asynchronous generation using the event loop
            generate_tasks = [
                self._generate_single_response_async(
                    semaphore=semaphore,
                    candidate=candidate,
                    eval=eval_,
                )
                for candidate, candidate_evals in zip(self.candidates, eval_copies)
                for eval_ in candidate_evals
            ]
            flat_responses = await asyncio.gather(*generate_tasks)
            # Reshape responses by candidate
            index = 0
            for candidate_evals in eval_copies:
                num_evals = len(candidate_evals)
                responses.append(flat_responses[index:index + num_evals])
                index += num_evals
        elif self.response_mode == Mode.PARALLEL:
            # Parallel generation using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
                arg_pairs = [
                    (candidate, eval_)
                    for candidate, candidate_evals in zip(self.candidates, eval_copies)
                    for eval_ in candidate_evals
                ]
                flat_responses = list(executor.map(EvalHarness._generate_single_response, *zip(*arg_pairs)))  # noqa: E501
            # Reshape responses by candidate
            index = 0
            for candidate_evals in eval_copies:
                num_evals = len(candidate_evals)
                responses.append(flat_responses[index:index + num_evals])
                index += num_evals
        else:
            raise ValueError("mode must be `Mode.SYNC`, `Mode.ASYNC`, or `Mode.PARALLEL`")
        return responses

    async def _run_all_evals(
            self,
            responses: list[list[tuple[CandidateResponse, Exception | None]]],
            eval_copies: list[list[Eval]],
        ) -> list[CandidateRunResults]:
        """
        Runs all Evals against all of the responses generated for each Candidate via
        `_generate_responses`.
        """
        results = []
        if self.eval_mode == Mode.SYNC:
            for candidate, candidate_responses, candidate_evals in zip(self.candidates, responses, eval_copies):  # noqa: E501
                candidate_results = []
                for eval_, (candidate_response, response_error) in zip(candidate_evals, candidate_responses):  # noqa: E501
                    eval_result, eval_error = EvalHarness._run_single_eval(
                        eval=eval_,
                        candidate=candidate,
                        candidate_response=candidate_response,
                    )
                    candidate_results.append(EvalRunResult(
                        eval=eval_,
                        eval_result=eval_result,
                        response_error=response_error,
                        eval_error=eval_error,
                    ))
                results.append(CandidateRunResults(
                    candidate=candidate,
                    run_results=candidate_results,
                ))
        elif self.eval_mode == Mode.ASYNC:
            semaphore = asyncio.Semaphore(self.async_batch_size)
            eval_tasks = [
                self._run_single_eval_async(
                    semaphore=semaphore,
                    eval=eval_,
                    candidate=candidate,
                    candidate_response=candidate_response,
                )
                for candidate, candidate_responses, candidate_evals in zip(self.candidates, responses, eval_copies)  # noqa: E501
                for eval_, (candidate_response, _) in zip(candidate_evals, candidate_responses)
            ]
            flat_eval_results = await asyncio.gather(*eval_tasks)
            # Reshape results by candidate
            index = 0
            for candidate, candidate_responses, candidate_evals in zip(self.candidates, responses, eval_copies):  # noqa: E501
                candidate_results = []
                for eval_, (candidate_response, response_error) in zip(candidate_evals, candidate_responses):  # noqa: E501
                    eval_result, eval_error = flat_eval_results[index]
                    candidate_results.append(EvalRunResult(
                        eval=eval_,
                        eval_result=eval_result,
                        response_error=response_error,
                        eval_error=eval_error,
                    ))
                    index += 1
                results.append(CandidateRunResults(
                    candidate=candidate,
                    run_results=candidate_results,
                ))
        elif self.eval_mode == Mode.PARALLEL:
            with ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
                arg_pairs = [
                    (eval_, candidate, candidate_response)
                    for candidate, candidate_responses, candidate_evals in zip(self.candidates, responses, eval_copies)  # noqa: E501
                    for eval_, (candidate_response, _) in zip(candidate_evals, candidate_responses)
                ]
                flat_eval_results = list(executor.map(EvalHarness._run_single_eval, *zip(*arg_pairs)))  # noqa: E501
            # Reshape results by candidate
            index = 0
            for candidate, candidate_responses, candidate_evals in zip(self.candidates, responses, eval_copies):  # noqa: E501
                candidate_results = []
                for eval_, (candidate_response, response_error) in zip(candidate_evals, candidate_responses):  # noqa: E501
                    eval_result, eval_error = flat_eval_results[index]
                    candidate_results.append(EvalRunResult(
                        eval=eval_,
                        eval_result=eval_result,
                        response_error=response_error,
                        eval_error=eval_error,
                    ))
                    index += 1
                results.append(CandidateRunResults(
                    candidate=candidate,
                    run_results=candidate_results,
                ))
        else:
            raise ValueError("mode must be `Mode.SYNC`, `Mode.ASYNC`, or `Mode.PARALLEL`")
        return results

    async def _execute(self) -> list[CandidateRunResults]:
        """
        Creates all necessary eval copies up front and executes the harness.
        Each candidate gets its own set of eval copies, and each eval is copied num_samples times.
        """
        # Create copies of evals up front - one set per candidate, num_samples copies of each eval
        eval_copies = [
            [
                deepcopy(eval_)
                for eval_ in self.evals
                for _ in range(self.num_samples)  # creates num_samples copies of each eval
            ]
            for _ in self.candidates
        ]
        responses = await self._generate_all_responses(eval_copies)
        return await self._run_all_evals(responses, eval_copies)

    def __call__(self) -> list[CandidateRunResults]:
        """
        Executes the EvalHarness. The response_mode and eval_mode can be set to "sync" (default),
        "async", or "parallel". The "sync" mode runs the corresponding tasks (generating responses
        or running evals) synchronously, the "async" mode runs the tasks asynchronously using the
        event loop, and the "parallel" mode runs the tasks in parallel using ProcessPoolExecutor
        across all CPU cores.
        """
        return asyncio.run(self._execute())
