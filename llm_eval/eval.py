"""Classes and functions to evaluate LLMs."""

import asyncio
import glob
import os
import time
import yaml
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, Callable
from llm_eval.candidates import Candidate, CandidateResponse, is_async_candidate
from llm_eval.checks import (
    Check,
    CheckResult,
    PassFailResult,
    ResponseData,
)
from llm_eval.internal_utilities import DictionaryEqualsMixin


class Eval(DictionaryEqualsMixin):
    """
    An Eval is single test-case/scenario that the user is interested in evaluating. It typically
    consists of sending an input to a Candidate (e.g. 'messages' to ChatGpt) and checking the
    response against one or more "checks" (e.g. checking if the response mathces/contains a
    specific phrase, or has generated python code).

    An Eval is a callable object that is executed by calling it with a Candidate object, or a
    dictionary representing a Candidate that has been registered via `Candidate.register(...)`.
    """

    def __init__(
            self,
            input: str | dict | list | Any,  # noqa: A002, ANN401
            checks: list[Check | dict | Callable[[Any], CheckResult]] | None = None,
            ideal_response: str | list | None = None,
            metadata: dict | None = None) -> None:
        """
        Initializes the Eval.

        Args:
            input:
                The input to send to the Candidate/LLM (i.e. the input to the __call__ of the
                corresponding Candidate object). This can be a string, dictionary, list, or
                any other type that the Candidate/LLM can accept.
            checks:
                A list of Check objects, dictionaries, or callables. If a dictionary is passed in,
                the Check subclasses need to be registered via `Check.register(...)`, and the
                dictionary needs a `check_type` key with the registration value.
            ideal_response:
                The ideal response or list of ideal responses that the Candidate/LLM should return.
                This is optional and not currently used in the evaluation process, but it can be
                used in user-defined checks and candidates.
            metadata:
                Metadata associated with the Eval.
        """
        self._has_executed = False
        self._candidate = None
        self._response = None
        self._duration = None

        self.input = input
        self.ideal_response = ideal_response
        self.metadata = deepcopy(metadata) or {}
        checks = checks or []
        if not isinstance(checks, list):
            checks = [checks]
        checks_created = []
        for check in checks:
            if isinstance(check, dict):
                assert 'check_type' in check, "Check dictionary must contain a 'check_type' key"
                checks_created.append(Check.from_dict(check))
            elif isinstance(check, (Callable, Check)):
                checks_created.append(check)
            else:
                raise TypeError("Checks must be either a Check, dictionary, or callable.")
        self.checks = checks_created

    def to_dict(self) -> dict:
        """Return a dictionary representation of the PromptTest."""
        value = {}
        value['input'] = self.input
        if self.checks:
            value['checks'] = [
                c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in self.checks
            ]
        if self.ideal_response:
            value['ideal_response'] = self.ideal_response
        if self.metadata:
            value['metadata'] = self.metadata
        return value

    def to_yaml(self, file_path: str) -> None:
        """Saves the Eval to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'Eval':  # noqa: ANN102
        """
        Creates an Eval object from a YAML file.

        Any custom checks must be registered with the CheckRegistry before calling this
        method.

        For example:

        ```
        from llm_eval.checks import register_check

        @register_check('my_custom_check)
        class MyCheck(EvalCheck):
            ...
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def _to_candidate(self, candidate: Candidate | Callable | dict) -> Candidate:
        """Converts the candidate parameter to a Candidate object."""
        if isinstance(candidate, dict):
            candidate = Candidate.from_dict(candidate)
        assert isinstance(candidate, (Candidate, Callable)), \
            "candidate must be a Candidate or a callable"
        return candidate

    def _generate_response(self, candidate: Candidate | Callable | dict) -> None:
        """
        _generate_response is responsible for generating responses from the Candidate/LLM. It is a
        separate function from _execute_checks so that we can async the generation of responses
        and then execute the checks (which are heavier on the CPU and shouldn't be async).

        This method has side effects of setting self._response, self._duration, and
        self._candidate, which are used by _execute_checks. This is bad practice but we need to do
        this to support calling _generate_response async and then executing the checks afterwards.
        """
        self._candidate = self._to_candidate(candidate)
        self._response = None
        start = time.time()
        try:
            if is_async_candidate(self._candidate):
                self._response = asyncio.run(self._candidate(self.input))
            else:
                self._response = self._candidate(self.input)
        finally:
            end = time.time()
            self._duration = end - start

    async def _async_generate_response(self, candidate: Candidate | Callable | dict) -> None:
        """Async version of _generate_response. See function for details."""
        self._candidate = self._to_candidate(candidate)
        start = time.time()
        self._response = None
        try:
            self._response = await self._candidate(self.input)
        finally:
            end = time.time()
            self._duration = end - start

    def _execute_checks(self) -> 'EvalResult':
        """
        Executes the checks against the responses and returns an EvalResult object. This method
        should only be called after _generate_response has been called. This method is separate
        from _generate_response so that we can async the generation of responses and then execute
        the checks (which are heavier on the CPU and shouldn't be async).
        """
        assert self._candidate
        assert self._duration is not None
        check_results = [
            check(ResponseData(
                input=self.input,
                ideal_response=self.ideal_response,
                response=self._response.response,
                response_metadata=self._response.metadata,
            ))
            for check in self.checks
        ]
        return EvalResult(
            eval_obj=self,
            candidate_obj=self._candidate,
            response=self._response.response,
            response_metadata=self._response.metadata,
            total_time_seconds=self._duration,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            check_results=check_results,
        )

    def __call__(
            self,
            candidate: Candidate | Callable[[Any], CandidateResponse] | dict,
        ) -> 'EvalResult':
        """
        Evaluates the model against the prompts and tests.

        Args:
            candidate:
                The Candidate object to evaluate. If the Candidate is a dictionary, the Candidate
                subclasses need to be registered via `Candidate.register(...)`, and the dictionary
                needs a `candidate_type` key with the registration value.

                If the Candidate is a callable, it should be a function that takes the `input` as
                an argument and returns a CandidateResponse object.
        """
        if self._has_executed:
            raise RuntimeError("Eval has already been executed; create a new Eval object")
        self._has_executed = True
        self._generate_response(candidate)
        # _generate_response has side effects of setting self._response, self._duration, and
        # self._candidate that _execute_check relies on;
        # this is bad practice but we need to do this to support calling _generate_response async
        # and then executing the checks afterwards
        results = self._execute_checks()
        # these fields should be reset so we don't accidentally use them again; they should not be
        # accessed directly; they are only used to store information between running
        # _generate_response and _execute_checks
        self._candidate = None
        self._response = None
        self._duration = None
        return results

    def __str__(self) -> str:
        """Returns a string representation of the Eval."""
        metadata = '' if not self.metadata else f'            metadata={self.metadata},\n{" " * 12}'  # noqa
        return dedent(f"""
        Eval(
            {metadata}input={self.input},
        )
        """).strip()

    def clone(self) -> 'Eval':
        """
        Returns a copy of the Candidate with the same state but with a different instance of the
        underlying model (e.g. same parameters but reset history/context).
        """
        return Eval(
            input=deepcopy(self.input),
            checks=deepcopy(self.checks),
            ideal_response=deepcopy(self.ideal_response),
            metadata=deepcopy(self.metadata),
        )


class EvalResult(DictionaryEqualsMixin):
    """
    An EvalResult is the result of evaluating a specific Candidate/LLM against a specific
    Eval.
    """

    def __init__(
        self,
        eval_obj: Eval | dict,
        candidate_obj: Candidate | dict | Callable,
        response: str | object,
        response_metadata: dict,
        total_time_seconds: float,
        timestamp: str,
        check_results: list[CheckResult | dict]) -> None:
        """
        Initializes the EvalResult.

        Args:
            eval_obj:
                The Eval object that was evaluated.
            candidate_obj:
                The Candidate object that was evaluated. If the Candidate is a dictionary, the
                Candidate subclasses need to be registered via `Candidate.register(...)`.
                The dictionary needs a `candidate_type` key with the registration value.
            response:
                The response from the Candidate (e.g. LLM/agent).
            response_metadata:
                Metadata associated with the response (e.g. the response.metadata from the
                CandidateResponse object).
            total_time_seconds:
                The total time (in seconds) it took to run the Eval.
            timestamp:
                The timestamp when the Eval was completed.
            check_results:
                A list of CheckResult objects.
        """
        self.eval_obj = eval_obj if isinstance(eval_obj, Eval) else Eval(**deepcopy(eval_obj))
        if isinstance(candidate_obj, Candidate):
            self.candidate_obj = candidate_obj
        elif isinstance(candidate_obj, dict):
            if 'candidate_type' in candidate_obj:
                # loads the Candidate subclass from the registry
                self.candidate_obj = Candidate.from_dict(candidate_obj)
            else:
                self.candidate_obj = Candidate(**deepcopy(candidate_obj))
        else:
            self.candidate_obj = str(candidate_obj)
        self.response = response
        self.response_metadata = response_metadata
        self.total_time_seconds = total_time_seconds
        self.timestamp = timestamp
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
        if isinstance(self.candidate_obj, Candidate):
            candidate_obj = self.candidate_obj.to_dict()
        else:
            candidate_obj = str(self.candidate_obj)
        return {
            'eval_obj': self.eval_obj.to_dict(),
            'candidate_obj': candidate_obj,
            'response': deepcopy(self.response),
            'response_metadata': deepcopy(self.response_metadata),
            'total_time_seconds': self.total_time_seconds,
            'timestamp': self.timestamp,
            'check_results': [r.to_dict() for r in self.check_results],
        }

    def to_yaml(self, file_path: str) -> None:
        """Saves the EvalResult to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'EvalResult':  # noqa: ANN102
        """Creates an EvalResult object from a YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return EvalResult(**config)

    def to_json(self, file_path: str) -> None:
        """Saves the EvalResult to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> 'EvalResult':  # noqa: ANN102
        """Creates an EvalResult object from a JSON file."""
        with open(path) as f:
            config = json.load(f)
        return EvalResult(**config)


class ResponseError(RuntimeError):
    """
    Exception raised for errors that occur when generating responses from a candidate using the
    EvalHarness. These errors can be captured or ignored using an error_callback function on the
    EvalHarness.

    Args:
        eval_obj (Eval): The Eval object that was being evaluated.
        candidate_obj (Candidate): The Candidate object that was being evaluated.
        exception (Exception): The exception that was raised during the evaluation.
    """

    def __init__(self, exception: Exception, eval_obj: Eval, candidate_obj: Candidate) -> None:
        """
        Initializes the ResponseException with the raised exception, and references to the Eval and
        Candidate objects involved in the error.

        Args:
            exception (Exception): The original exception that was raised.
            eval_obj (Eval): The Eval object involved in the error.
            candidate_obj (Candidate): The Candidate object involved in the error.
        """
        message = f"Error in evaluating {eval_obj.metadata.get('id', 'unknown')} for candidate " \
            f"{candidate_obj.metadata.get('id', 'unknown')}: {exception}"
        super().__init__(message)
        self.eval_obj = eval_obj
        self.candidate_obj = candidate_obj
        self.exception = exception

    def __reduce__(self):
        """
        __reduce__ helps Python understand how to pickle and unpickle an instance of ResponseError,
        ensuring that the necessary data is serialized correctly across processes.
        """
        # Return the class and the tuple of args needed to reconstruct it
        return (self.__class__, (self.exception, self.eval_obj, self.candidate_obj))


class EvalHarness:
    """
    An EvalHarness provides a interface for evaluating multiple Evals against multiple
    Candidates/LLMs.

    Candidates must be registered via Candidate.register if they are passed in as dictionaries.

    Checks used by EvalHarness must be a CloneableCheck because Evals (and corresponding Check
    objects) are cloned across multiple candidates.

    The EvalHarness is responsible for calling each Eval object with each Candidate and returning a
    collection of EvalResults.

    By default, the EvalHarness will run each Candidate (and the corresponding Evals) in parallel
    on different CPUs. The number of CPUs can be set via the `num_cpus` parameter. If `num_cpus` is
    set to 1, the EvalHarness will run each Candidate (and the corresponding Evals) sequentially.

    Additionally, for each Candidate, the EvalHarness will run all Evals in asynchronous batches.
    The user must be careful to set the `async_batch_size` parameter to a value that is appropriate
    for the underlying Candidate. For example, if the Candidate is a remote API that can handle a
    very large number of requests such as OpenAI, the `async_batch_size` can be set to a large
    number. If the Candidate is a local model, the `async_batch_size` should be set to a small
    number to avoid performance issues (which will effect metrics such as characters per second).

    Example usage:

        ```python
        from llm_eval import EvalHarness

        harness = EvalHarness(
            num_cpus=-1,
            async_batch_size=50,
            callback=print,  # optional
            num_samples=5,  # optional
        )
        harness.add_evals_from_yamls('xxx/evals/')
        harness.add_candidates_from_yamls('xxx/candidates/')
        results = harness()
    ```

    """

    def __init__(
            self,
            evals: list[Eval | dict ] | Eval | dict | None = None,
            candidates: list[Candidate | Callable | dict] | Candidate | dict | None = None,
            num_cpus: int | None = None,
            async_batch_size: int | None = 50,
            num_samples: int = 1,
            callback: Callable[[EvalResult], None] | None = None,
            error_callback: Callable[[Exception, Eval, Candidate], None] | None = None,
            ) -> None:
        """
        Initializes the EvalHarness. The user can either pass in Eval and Candidate objects in the
        constructor or call
            - `add_eval_from_yaml(...)`, which takes a path to a YAML file.
            - `add_evals_from_yamls(...)`, which takes a path to a directory of YAML files.
            - `add_candidate_from_yaml(...)`, which takes a path to a YAML file.
            - `add_candidates_from_yamls(...)`, which takes a path to a directory of YAML files.

        The methods above can be called multiple times to add additional Eval and Candidate
        objects.

        Example:

        ```
        harness = EvalHarness(
            num_cpus=-1,
            async_batch_size=50,
            num_samples=5,  # optional; runs 5 samples for each Eval
            callback=print,  # optional
            # optional; captures errors when generating responses from the candidate and allows
            # the EvalHarness to continue execution; without this the EvalHarness will raise an
            # exception for any errors that occur when generating responses by the Candidate
            error_callback=print,
        )
        harness.add_eval_from_yaml('xxx/eval.yaml')
        harness.add_candidate_from_yaml('xxx/candidate.yaml')
        results = harness()
        ```

        Args:
            evals:
                A list of Eval objects or dictionaries. Alternatively, `add_eval...` methods can
                be called to add Eval objects.
            candidates:
                A list of Candidate objects or dictionaries. Alternatively, `add_candidate...`
                methods can be called to add Candidate objects. The candidate either needs to
                implement the `clone()` method or should be stateless. The reason for this is that
                the EvalHarness will clone the Candidate for each Eval so that we can run each Eval
                against a unique Candidate in memory. This isn't necessary for stateless
                Candidates.
            num_cpus:
                The number of CPUs to use for parallel processing. If set to 1, the EvalHarness
                will run each Candidate (and the corresponding Evals) sequentially. If set to
                None or a number < 1, the EvalHarness will use all available CPUs. If set to a
                number > 1, the EvalHarness will use the specified number of CPUs.
            async_batch_size:
                The number of Evals to run asynchronously for each Candidate. If set to None, the
                batch size is se tot he number of evals. If set to a number, the EvalHarness
                will run the Evals in batches of the specified size.
            callback:
                A callback function that will be called for each EvalResult (each Eval/Candidate
                result) after the corresponding checks have been executed for that Eval. The
                callback function should take a single parameter, which is an EvalResult object.
                The callback can be used, for example, to save the EvalResult objects as they are
                generated (in case the EvalHarness fails to complete due to an error or the user
                wants to save the results as they are generated).
            error_callback:
                By default, if there is an error generating a response from the candidate (e.g.
                context limit, rate limit, etc.) the EvalHarness will raise an exception. If the
                user wants to capture or ignore these errors (and allow the EvalHarness to
                continue execution) the user can pass in a callback function that will be called
                when an error occurs. The callback function should take three parameters: the
                exception, the Eval, and the Candidate. If the callback is set, the EvalHarness
                will not raise an exception for any errors that occur when generating responses by
                the Candidate. Instead, the user can check the results to see if there were any
                errors. The error will be stored in the corresponding EvalResult object in a
                `harness_exception` key in the response_metadata dictionary and the checks should
                have a `False` value for the `success` property.
            num_samples:
                The number of samples to run for each Eval. This is useful for running multiple
                samples for each Eval to get a better estimate of the performance metrics. Running
                multiple samples only makes sense for a non-zero temperature for the LLM. A zero
                temperature (or close) will give the same or very similar responses for each
                sample, defeating the purpose of collecting a sample.
        """  # noqa: D412
        self.num_cpus = num_cpus
        self.async_batch_size = async_batch_size
        self.callback = callback
        self.error_callback = error_callback
        self.num_samples = num_samples
        self.evals = []
        self.candidates = []
        if evals:
            self.add_evals(evals)
        if candidates:
            self.add_candidates(candidates)

    def add_evals(self, eval_obj: list[Eval | dict ] | Eval | dict) -> None:
        """
        Adds an Eval object. This method can be called multiple times to add additional Eval
        objects.

        Args:
            eval_obj:
                The Eval object to add. If the Eval is a dictionary, the Check subclasses need to
                be registered via `Check.register(...)`.
                The checks needs a `check_type` key with the registration value.
        """
        if isinstance(eval_obj, dict):
            self.evals.append(Eval(**eval_obj))
        elif isinstance(eval_obj, Eval):
            self.evals.append(eval_obj)
        elif isinstance(eval_obj, list):
            for obj in eval_obj:
                self.add_evals(obj)
        else:
            raise TypeError(f"incompatible type {type(eval_obj)} for eval_obj")

    def add_eval_from_yaml(self, path: str) -> None:
        """
        Adds an Eval from a YAML file. This method can be called multiple times to add additional
        Eval objects.

        The underlying Check objects must be registered with the CheckRegistry before calling this
        method and the checks need a `check_type` key with the registration value.

        Args:
            path:
                Path to the YAML file.
        """
        self.add_evals(Eval.from_yaml(path))

    def add_evals_from_yamls(self, path: str) -> None:
        """
        Adds multiple Evals from a directory of YAML files. This method can be called multiple
        times to add additional Eval objects.

        The underlying Check objects must be registered with the CheckRegistry before calling this
        method and the checks need a `check_type` key with the registration value.

        Args:
            path:
                Path to the directory of YAML files along shell-style wildcards. For example,
                `path/to/directory/*.yaml` (passed to `glob.glob` function).

        """
        for file_path in glob.glob(path):
            self.add_eval_from_yaml(file_path)

    def add_candidates(
            self,
            candidate: list[Candidate | dict | Callable[[Any], Any]] | Candidate | dict | Callable[[Any], Any],  # noqa
            ) -> None:
        """
        Adds a Candidate object. This method can be called multiple times to add additional
        Candidate objects.

        Args:
            candidate:
                The Candidate object to add. If the Candidate is a dictionary, the Candidate
                subclasses need to be registered via `Candidate.register(...)`.
                The dictionary needs a `candidate_type` key with the registration value.

                The candidate either needs to implement the `clone()` method or should be
                stateless. The reason for this is that the EvalHarness will clone the Candidate for
                each Eval so that we can run each Eval against a unique Candidate in memory. This
                isn't necessary for stateless Candidates.
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

    def add_candidate_from_yaml(self, path: str) -> None:
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
        with open(path) as f:
            config = yaml.safe_load(f)
        self.add_candidates(config)

    def add_candidates_from_yamls(self, path: str) -> None:
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
            self.add_candidate_from_yaml(file_path)

    @staticmethod
    def _generate_eval_responses(candidate: Candidate, eval_obj: Eval) -> tuple[Eval, Exception]:
        """
        Generates the response(s) from the Candidate/LLM for a particular Eval. Ensure that the
        Eval objects are cloned before calling this method.
        """
        eval_obj = eval_obj.clone()
        exception = None
        try:
            eval_obj._generate_response(candidate)
        except Exception as e:
            exception = e
        return eval_obj, exception

    @staticmethod
    async def _async_generate_eval_responses(
        candidate: Candidate,
        eval_obj: Eval) -> tuple[Eval, Exception]:
        """
        Generates the response(s) from the Candidate/LLM for a particular Eval where the Candidate
        is an async function. See additional notes from non-async function
        `_generate_eval_responses`.
        """
        eval_obj = eval_obj.clone()
        exception = None
        try:
            await eval_obj._async_generate_response(candidate)
        except Exception as e:
            exception = e
        return eval_obj, exception

    @staticmethod
    async def _run_async_evals_batch(
        candidate: Candidate,
        evals: list[Eval]) -> list[tuple[Eval, Exception]]:
        """Generates responses asynchronously for candidates with async functions."""
        tasks = [
            EvalHarness._async_generate_eval_responses(candidate, eval_obj)
            for eval_obj in evals
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _run_eval_batch_asynchronously(
        candidate: Candidate,
        evals: list[Eval]) -> list[tuple[Eval, Exception]]:
        """Generates responses asynchronously for candidates with non-async functions."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tasks = [
                loop.run_in_executor(
                    None,EvalHarness._generate_eval_responses,candidate,eval_obj,
                )
                for eval_obj in evals
            ]
            return loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            loop.close()

    @staticmethod
    def _process_response(
            response_eval: Eval,
            candidate: Candidate,
            exception: Exception | None,
            callback: Callable[[EvalResult], None] | None,
            error_callback: Callable[[Exception, Eval, Candidate], None] | None,
            ) -> EvalResult:
        """
        When we get an exception when generating a response, we either need to send the error via
        error callback, or we need to raise an exception and stop the harness.

        If no error occurs, we execute the checks and send the results via regular callback.
        """
        if exception:
            # if there is an error and a callback, then we will call the callback and
            # return still continue running evals; otherwise, we will raise an exception
            if error_callback:
                error_callback(exception, response_eval, candidate)
                # we still need to generate a result object, set the exception, and have
                # the CheckResults have a `False` value for the `success` property
                # We can set the response to empty strings and run the checks to
                # accomplish this.
                response_eval._response = CandidateResponse(response='', metadata={})
                eval_result = response_eval._execute_checks()
                if eval_result.response_metadata is None:
                    eval_result.response_metadata = {}
                eval_result.response_metadata['harness_exception'] = exception
            else:
                raise ResponseError(exception, response_eval, candidate)
        else:
            eval_result = response_eval._execute_checks()
        if callback:
            callback(eval_result)
        return eval_result

    @staticmethod
    def _run_evals(
            candidate: Candidate,
            evals: list[Eval],
            async_batch_size: int | None,
            callback: Callable[[EvalResult], None] | None,
            error_callback: Callable[[Exception, Eval, Candidate], None] | None,
        ) -> list[EvalResult]:
        eval_batch_size = len(evals) if async_batch_size is None else async_batch_size
        assert eval_batch_size >= 1
        results = []

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for i in range(0, len(evals), eval_batch_size):
            eval_batch = evals[i:i + eval_batch_size]
            if is_async_candidate(candidate):
                # regardless of batch size, if the candidate is async, run asynchronously
                batch_results = loop.\
                    run_until_complete(EvalHarness._run_async_evals_batch(candidate, eval_batch))
            elif eval_batch_size > 1:
                # if we are running non-async functions in batches, then run asynchronously
                batch_results = EvalHarness._run_eval_batch_asynchronously(candidate, eval_batch)
            else:
                # run synchronously
                batch_results = [
                    EvalHarness._generate_eval_responses(candidate, eval_obj)
                    for eval_obj in eval_batch
                ]
            for response_eval, exception in batch_results:
                eval_result = EvalHarness._process_response(
                    response_eval, candidate, exception, callback, error_callback,
                )
                results.append(eval_result)
        return results

    def __call__(self, num_samples: int | None = None) -> list[list[EvalResult]]:
        """
        Evaluates the Evals against the Candidates.

        Returns a list of lists of EvalResults. Each index of the outer list corresponds to a
        particular candidate (in the order they were added to the EvalHarness). Each
        index/candidate contains a list of EvalResults (in the order they were added to the
        EvalHarness).

        So if [{candidate_1}, {candidate_2}] were added to the EvalHarness and [{eval_1},
        {eval_2}, {eval_3}] were added to the EvalHarness, the returned EvalResult object
        would be in the following order:

            [
                [
                    {eval_1 result for candidate_1},
                    {eval_2 result for candidate_1},
                    {eval_3 result for candidate_1}
                ],
                [
                    {eval_1 result for candidate_2},
                    {eval_2 result for candidate_2},
                    {eval_3 result for candidate_2}
                ],
            ]

        If a callback has been set, the callback will be called for each EvalResult after the
        corresponding checks have been executed for that Eval.

        Args:
            num_samples:
                The number of samples to run for each Eval. num_samples can also be passed to the
                constructor of EvalHarness. See notes in the constructor for more information. This
                parameter will override the value passed to the constructor.
        """
        num_cpus = self.num_cpus
        evals = self.evals
        num_samples = num_samples or self.num_samples
        if num_samples > 1:
            evals = [eval_obj.clone() for eval_obj in evals for _ in range(num_samples)]
        # currently, if num_cpus is set to >1, each candidate will be run in parallel on a
        # separate CPU; if num_cpus is set to 1, each candidate will be run sequentially;
        # if there is only one candidate, there is nothing to parallelize with this implementation;
        # future implementations could parallelize batches evals for 1 or more candidates
        if num_cpus == 1 or len(self.candidates) == 1:
            return [
                EvalHarness._run_evals(
                    candidate=candidate,
                    evals=evals,
                    async_batch_size=self.async_batch_size,
                    callback=self.callback,
                    error_callback=self.error_callback,
                )
                for candidate in self.candidates
            ]
        if num_cpus is None or num_cpus < 1:
            num_cpus = os.cpu_count()
        results = []
        # run each candidate in a separate process/CPU
        for i in range(0, len(self.candidates), num_cpus):
            candidate_batch = self.candidates[i:i + num_cpus]
            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                futures = [
                    executor.submit(
                        EvalHarness._run_evals,
                        candidate=candidate,
                        evals=evals,
                        async_batch_size=self.async_batch_size,
                        callback=self.callback,
                        error_callback=self.error_callback,
                    )
                    for candidate in candidate_batch
                ]
                batch_results = [future.result() for future in futures]
                results.extend(batch_results)
        return results
