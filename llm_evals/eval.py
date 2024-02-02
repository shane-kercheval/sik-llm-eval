"""Classes and functions to evaluate LLMs."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import glob
import os
from textwrap import dedent, indent
import time
from typing import Callable
import yaml
from llm_evals.candidates import CallableCandidate, Candidate
from llm_evals.checks import (
    Check,
    CheckResult,
    CheckType,
)
from llm_evals.utilities.internal_utilities import (
    DictionaryEqualsMixin,
    extract_code_blocks,
    extract_valid_parameters,
    get_callable_info,
    has_property,
)


class PromptTest(DictionaryEqualsMixin):
    """
    A PromptTest represents a prompt, an optional ideal response, and a list of checks to run
    against the response. There can be one or more PromptTests in an Eval. If more than one
    PromptTest is provided, the intent is evaluate a conversation and, therefore, it's expected
    that the underlying Candidate (model/client) will maintain state between multiple
    prompt/response exchanges.

    Checks are optional because even a test without checks still collects performance information
    (e.g. characters per second) as well as responses (which can be visually/subjectively compared
    against either the ideal response or the responses from other LLMs).

    A PromptTest only contains information (it is not directly callable). An Eval object is
    responsible for calling the PromptTest(s), executing the checks, and returning a EvalResult
    object.
    """

    def __init__(
            self,
            prompt: str,
            ideal_response: str | None = None,
            checks: list[Check | dict] | None = None) -> None:
        """
        Initializes the PromptTest.

        Args:
            prompt:
                The prompt to send to the LLM.
            ideal_response:
                The ideal response to compare against the LLM's response.
            checks:
                'A list of checks to run against the response. If a dictionary is provided, the
                Check subclasses need to be registered via `Check.register(...)`.
                The dictionary needs a `check_type` key with the registration value.
        """
        self.prompt = dedent(prompt)
        self.ideal_response = dedent(ideal_response) if ideal_response else None
        checks = checks or []
        checks_created = []
        for check in checks:
            if isinstance(check, dict):
                assert 'check_type' in check, "Check dictionary must contain a 'check_type' key"
                checks_created.append(Check.from_dict(deepcopy(check)))
            elif isinstance(check, Check):
                checks_created.append(check)
            else:
                raise TypeError("Checks must be either a Check instance or a dictionary")
        self.checks = checks_created

    def __str__(self) -> str:
        """Returns a string representation of the PromptTest."""
        if self.checks:
            indent_value = ' ' * 16
            checks = '[\n' + indent_value
            checks += f',\n{indent_value}'.join([str(c) for c in self.checks]) if self.checks else ''  # noqa: E501
            checks += '\n            ]'
        else:
            checks = '[]'
        if self.ideal_response:
            ideal_response = self.ideal_response.strip()
            if len(ideal_response) > 50:
                ideal_response = ideal_response[0:50] + '...'
            ideal_response = f'\n            ideal_response="{ideal_response}",'
        else:
            ideal_response = ''
        return dedent(f"""
        {self.__class__.__name__}(
            prompt='{self.prompt}',{ideal_response}
            checks={checks},
        )
        """).strip()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the PromptTest."""
        value = {'prompt': self.prompt}
        if self.ideal_response:
            value['ideal_response'] = self.ideal_response
        if self.checks:
            value['checks'] = [c.to_dict() for c in self.checks]
        return value


class Eval(DictionaryEqualsMixin):
    """
    An Eval is single scenario (i.e. one or more prompts) that the user is interested in
    evaluating. Multiple prompts can be used to test conversations (i.e. sequential
    prompt/response exchanges where the assumption is the LLM client maintains conversational
    history). Additionally, each Eval is associated with custom "checks". Examples of checks are:
    if the response matches an exact value, if it contain a particular value, if it contain code
    blocks, if those code blocks run, if the variables/functions/etc created by those code blocks
    contain the expected values/behavior.

    An Eval is a callable object that is executed by calling it with a Candidate object, or a
    dictionary representing a Candidate that has been registered via `Candidate.register(...)`.
    """

    def __init__(
            self,
            test_sequence: list[PromptTest | dict] | dict | PromptTest,
            metadata: dict | None = None) -> None:
        """
        Initializes the Eval.

        Args:
            test_sequence:
                A list of PromptTest objects (prompt/check pairs) to run against the LLM.
            metadata:
                Metadata associated with the Eval.
        """
        self.metadata = deepcopy(metadata)
        self._candidate = None
        self._responses = None
        self._duration = None

        test_sequence = test_sequence or []
        if isinstance(test_sequence, (dict, PromptTest)):
            test_sequence = [test_sequence]
        tests_created = []
        for test in test_sequence:
            if isinstance(test, dict):
                tests_created.append(PromptTest(**test))
            elif isinstance(test, PromptTest):
                tests_created.append(test)
            else:
                raise TypeError(
                    "test_sequence must be either a PromptTest instance or a dictionary",
                )
        self.test_sequence = tests_created

    def to_dict(self) -> dict:
        """Return a dictionary representation of the PromptTest."""
        value = {'test_sequence': [t.to_dict() for t in self.test_sequence]}
        if self.metadata:
            value['metadata'] = self.metadata
        return value

    @classmethod
    def from_yaml(cls, path: str) -> 'Eval':  # noqa: ANN102
        """
        Creates an Eval object from a YAML file.

        Any custom checks must be registered with the CheckRegistry before calling this
        method.

        For example:

        ```
        from llm_evals.checks import register_check

        @register_check('my_custom_check)
        class MyCheck(EvalCheck):
            ...
        """
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    @staticmethod
    def _to_candidate(candidate: Candidate | Callable | dict) -> Candidate:
        """Converts a candidate to a Candidate object."""
        if isinstance(candidate, dict):
            candidate = Candidate.from_dict(candidate)
        elif not isinstance(candidate, Candidate) and isinstance(candidate, Callable):
            # all Candidates must be callable so need to ensure it's not already a Candidate
            candidate = CallableCandidate(
                model=candidate,
                metadata={'function': get_callable_info(candidate)},
            )
        else:
            assert isinstance(candidate, Candidate), \
                "candidate must be either a Candidate, callable, or a dictionary"
        return candidate

    def _generate_responses(self, candidate: Candidate | Callable | dict) -> None:
        """
        _generate_responses is responsible for generating responses from the Candidate/LLM. It is a
        separate function from _execute_checks so that we can async the generation of responses
        and then execute the checks (which are heavier on the CPU and shouldn't be async).

        This method has side effects of setting self._responses, self._duration, and
        self._candidate, which are used by _execute_checks. This is bad practice but we need to do
        this to support calling _generate_responses async and then executing the checks afterwards.
        """
        self._candidate = self._to_candidate(candidate)
        start = time.time()
        self._responses = [self._candidate(p.prompt) for p in self.test_sequence]
        end = time.time()
        self._duration = end - start

    def _execute_checks(self) -> 'EvalResult':
        """
        Executes the checks against the responses and returns an EvalResult object. This method
        should only be called after _generate_responses has been called. This method is separate
        from _generate_responses so that we can async the generation of responses and then execute
        the checks (which are heavier on the CPU and shouldn't be async).
        """
        assert self._responses
        assert self._duration
        assert self._candidate
        results = []
        code_blocks = []
        for test, response in zip(self.test_sequence, self._responses):
            check_results = []
            code_blocks.extend(extract_code_blocks(response))
            parameters = {
                'prompt': test.prompt,
                'ideal_response': test.ideal_response,
                'response': response,
                'code_blocks': code_blocks,
            }
            for check in test.checks:
                valid_parameters = extract_valid_parameters(check.__call__, parameters)
                check_results.append(check(**valid_parameters))
            results.append(check_results)

        return EvalResult(
            eval_obj=self,
            candidate_obj=self._candidate,
            responses=self._responses,
            total_time_seconds=self._duration,
            num_code_blocks=len(code_blocks),
            cost = self._candidate.cost if has_property(self._candidate, 'cost') else None,
            results=results,
        )

    def __call__(self, candidate: Candidate | Callable | dict) -> 'EvalResult':
        """
        Evaluates the model against the prompts and tests.

        Args:
            candidate:
                The Candidate object to evaluate. If the Candidate is a dictionary, the Candidate
                subclasses need to be registered via `Candidate.register(...)`.
                The dictionary needs a `candidate_type` key with the registration value.

                If the Candidate is a callable, it will be wrapped in a CallableCandidate object
                and the metadata will be populated with the function's signature. The Candidate
                can be serialized via `to_dict()` and deserialized via `from_dict()` but will
                not be able to be called when deserialized, since the underlying function will not
                be serialized.
        """
        self._generate_responses(candidate)
        # _generate_responses has side effects of setting self._responses, self._duration, and
        # self._candidate that _execute_check relies on;
        # this is bad practice but we need to do this to support calling _generate_responses async
        # and then executing the checks afterwards
        results = self._execute_checks()
        # these fields should be reset so we don't accidentally use them again; they should not be
        # accessed directly; they are only used to store information between running
        # _generate_responses and _execute_checks
        self._candidate = None
        self._responses = None
        self._duration = None
        return results

    def __str__(self) -> str:
        """Returns a string representation of the Eval."""
        if self.test_sequence:
            test_sequence = '[\n'
            indent_value = ' ' * 16
            test_sequence += ',\n'.join([indent(str(s), indent_value) for s in self.test_sequence])
            test_sequence += '\n            ]'
        else:
            test_sequence = '[]'
        metadata = '' if not self.metadata else f'            metadata={self.metadata},\n{" " * 12}'  # noqa
        return dedent(f"""
        Eval(
            {metadata}test_sequence={test_sequence},
        )
        """).strip()

    def clone(self) -> 'Eval':
        """
        Returns a copy of the Candidate with the same state but with a different instance of the
        underlying model (e.g. same parameters but reset history/context).

        Reques
        """
        return Eval(**deepcopy(self.to_dict()))


class EvalResult(DictionaryEqualsMixin):
    """
    An EvalResult is the result of evaluating a specific Candidate/LLM against a specific
    Eval.
    """

    def __init__(
        self,
        eval_obj: Eval | dict,
        candidate_obj: Candidate | dict,
        responses: list[str],
        total_time_seconds: float,
        num_code_blocks: int,
        cost : float | None,
        results: list[list[CheckResult | dict]]) -> None:
        """
        Initializes the EvalResult.

        Args:
            eval_obj:
                The Eval object that was evaluated.
            candidate_obj:
                The Candidate object that was evaluated. If the Candidate is a dictionary, the
                Candidate subclasses need to be registered via `Candidate.register(...)`.
                The dictionary needs a `candidate_type` key with the registration value.
            responses:
                A list of responses (strings) from the LLM.
            total_time_seconds:
                The total time (in seconds) it took to run the Eval.
            num_code_blocks:
                The total number of code blocks generated across all responses.
            cost:
                The cost associated with the candidate. This is optional and only applicable to
                candidates that have a `cost` property.
            results:
                A list of lists of CheckResult objects.
        """
        self.eval_obj = eval_obj if isinstance(eval_obj, Eval) else Eval(**eval_obj)
        if isinstance(candidate_obj, Candidate):
            self.candidate_obj = candidate_obj
        elif isinstance(candidate_obj, dict):
            if 'candidate_type' in candidate_obj:
                # loads the Candidate subclass from the registry
                self.candidate_obj = Candidate.from_dict(deepcopy(candidate_obj))
            else:
                self.candidate_obj = Candidate(**candidate_obj)
        else:
            raise TypeError("candidate_obj must be either a Candidate or a dictionary")
        self.responses = responses
        if total_time_seconds <= 0:
            raise ValueError("Total time must be greater than zero")
        self.total_time_seconds = total_time_seconds
        self.num_code_blocks = num_code_blocks
        self.cost = cost
        results = results or []
        results_created = []
        # results is a list of lists of CheckResults (each list corresponds to a prompt/test)
        # convert dictionaries to CheckResults
        for tests in results:
            test_results_created = []  # maintain list of lists
            for r in tests:
                if isinstance(r, dict):
                    assert 'result_type' in r, \
                        "CheckResult dictionary must contain a 'result_type' key"
                    test_results_created.append(CheckResult.from_dict(deepcopy(r)))
                elif isinstance(r, CheckResult):
                    test_results_created.append(r)
                else:
                    raise TypeError("results must be either a CheckResult or a dictionary")
            results_created.append(test_results_created)
        self.results = results_created

    @property
    def prompts(self) -> list[str]:
        """Returns a list of prompts."""
        return [p.prompt for p in self.eval_obj.test_sequence]

    @property
    def ideal_responses(self) -> list[str | None]:
        """Returns a list of ideal responses."""
        return [p.ideal_response for p in self.eval_obj.test_sequence]

    @property
    def response_characters(self) -> int:
        """Returns the number of characters across all responses."""
        return sum(len(r) for r in self.responses)

    @property
    def characters_per_second(self) -> float:
        """Returns the number of characters per second across all responses."""
        # Adding a tiny value to prevent divide-by-zero error
        return sum(len(r) for r in self.responses) / (self.total_time_seconds + 1e-6)

    @property
    def num_checks(self) -> int:
        """Returns the number of checks."""
        return sum(len(r) for r in self.results)

    @property
    def num_successful_checks(self) -> int:
        """Returns the number of successful checks."""
        return sum(r.success for r in self.all_checks_results if r.success)

    @property
    def perc_successful_checks(self) -> float | None:
        """Returns the percentage of passing checks. If there are checks, returns None."""
        return self.num_successful_checks / self.num_checks if self.num_checks else None

    @property
    def all_checks_results(self) -> list[CheckResult]:
        """Returns a (flattened) list of all CheckResults."""
        return [r for result in self.results for r in result]

    def __str__(self) -> str:
        cost_str = f'\n{" " * 12}Cost:{" " * 22} ${self.cost:.4f}' if self.cost else ''
        candidate_name = self.candidate_obj.metadata.get('name', '')
        if candidate_name:
            candidate_name = f"Candidate:{' ' * 18}{candidate_name}\n{' ' * 12}"
        eval_name = self.eval_obj.metadata.get('name', '')
        if eval_name:
            eval_name = f"Eval:{' ' * 23}{eval_name}\n{' ' * 12}"
        return dedent(f"""
        EvalResult:
            {candidate_name}{eval_name}# of Prompts Tested:        {len(self.eval_obj.test_sequence)}{cost_str}
            Total Response Time:        {self.total_time_seconds:0.1f} seconds
            # of Response Characters:   {self.response_characters:,}
            # of Code Blocks Generated: {self.num_code_blocks}
            Characters per Second:      {self.characters_per_second:,.1f}
            # of Checks:                {self.num_checks}
            # of Successful Checks:     {self.num_successful_checks}
            % of Successful Checks:     {self.perc_successful_checks:.1%}
        """).strip()  # noqa: E501

    def to_dict(self) -> dict:
        """Return a dictionary representation of the EvalResult."""
        return {
            'eval_obj': self.eval_obj.to_dict(),
            'candidate_obj': self.candidate_obj.to_dict(),
            'responses': self.responses,
            'total_time_seconds': self.total_time_seconds,
            'num_code_blocks': self.num_code_blocks,
            'cost': self.cost,
            'results': [[r.to_dict() for r in result] for result in self.results],
        }

    def to_yaml(self, file_path: str) -> None:
        """Saves the EvalResult to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_yaml(cls, path: str) -> 'EvalResult':  # noqa: ANN102
        """Creates an EvalResult object from a YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return EvalResult(**config)


def eval_result_summarizer(result: EvalResult) -> dict:
    """Simple summarizer that returns a dictionary of summary statistics."""
    summary = {}
    if 'uuid' in result.eval_obj.metadata:
        summary['eval_uuid'] = result.eval_obj.metadata['uuid']
    if 'name' in result.eval_obj.metadata:
        summary['eval_name'] = result.eval_obj.metadata['name']
    if 'uuid' in result.candidate_obj.metadata:
        summary['candidate_uuid'] = result.candidate_obj.metadata['uuid']
    if 'name' in result.candidate_obj.metadata:
        summary['candidate_name'] = result.candidate_obj.metadata['name']
    summary['num_prompts'] = len(result.responses)
    if result.cost:
        summary['cost'] = result.cost
    summary['total_time_seconds'] = result.total_time_seconds
    summary['response_characters'] = result.response_characters
    summary['characters_per_second'] = result.characters_per_second
    summary['num_checks'] = result.num_checks
    summary['num_successful_checks'] = result.num_successful_checks
    summary['perc_successful_checks'] = result.perc_successful_checks
    summary['num_code_blocks'] = result.num_code_blocks
    code_run_checks = [
        r for r in result.all_checks_results
        if r.metadata.get('check_type', '') == CheckType.PYTHON_CODE_BLOCKS_RUN.name
    ]
    if code_run_checks:
        summary['num_code_blocks_successful'] = sum(
            r.metadata['num_code_blocks_successful'] for r in code_run_checks
        )
        summary['perc_code_blocks_successful'] = \
            summary['num_code_blocks_successful'] / summary['num_code_blocks']
        summary['num_code_block_checks'] = \
            sum(c.metadata['num_code_block_checks'] for c in code_run_checks)
        summary['num_code_block_checks_successful'] = \
            sum(c.metadata['num_code_block_checks_successful'] for c in code_run_checks)
    return summary


class EvalHarness:
    """
    An EvalHarness provides a interface for evaluating multiple Evals against multiple
    Candidates/LLMs.

    Candidates must be registered so that we can clone them. This is necessary because we need to
    clone the Candidate for each Eval so that we can run each Eval against a unique Candidate in
    memory (i.e. if the Candidate maintains state/history between prompts, we don't want to reuse
    the same candidate for each eval).

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
    """

    def __init__(
            self,
            evals: list[Eval] | list[dict] | None = None,
            candidates: list[Candidate | Callable | dict] | None = None,
            num_cpus: int | None = None,
            async_batch_size: int | None = 50,  # TODO: document None means async = # of evals
            callback: Callable | None = None) -> None:
        """
        Initializes the EvalHarness. The user can either pass in Eval and Candidate objects in the
        constructor or call
            - `add_eval_from_yaml(...)`, which takes a path to a YAML file.
            - `add_evals_from_yamls(...)`, which takes a path to a directory of YAML files.
            - `add_candidate_from_yaml(...)`, which takes a path to a YAML file.
            - `add_candidates_from_yamls(...)`, which takes a path to a directory of YAML files.

        The methods above can be called multiple times to add additional Eval and Candidate
        objects.
        """
        evals = evals or []
        eval_objects = []
        candidates = candidates or []
        candidate_objects = []

        self.num_cpus = num_cpus
        self.async_batch_size = async_batch_size
        self.callback = callback

        for eval_obj in evals:
            if isinstance(eval_obj, dict):
                eval_objects.append(Eval(**eval_obj))
            elif isinstance(eval_obj, Eval):
                eval_objects.append(eval_obj)
            else:
                raise TypeError("evals must be either an Eval instance or a dictionary")

        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate_objects.append(Candidate.from_dict(deepcopy(candidate)))
            elif isinstance(candidate, Candidate):
                candidate_objects.append(candidate)
            else:
                raise TypeError("candidates must be either a Candidate instance or a dictionary")

        self.evals = eval_objects
        self.candidates = candidate_objects

    def add_eval(self, eval_obj: Eval | dict) -> None:
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
            eval_obj = Eval(**eval_obj)
        assert isinstance(eval_obj, Eval), "eval_obj must be an Eval or a dictionary"
        self.evals.append(eval_obj)

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
        self.add_eval(Eval.from_yaml(path))

    def add_evals_from_yamls(self, path: str) -> None:
        """
        Adds multiple Evals from a directory of YAML files. This method can be called multiple
        times to add additional Eval objects.

        The underlying Check objects must be registered with the CheckRegistry before calling this
        method and the checks need a `check_type` key with the registration value.

        Args:
            path:
                Path to the directory of YAML files.
        """
        for file_path in glob.glob(path):
            self.add_eval_from_yaml(file_path)

    def add_candidate(self, candidate: Candidate | dict) -> None:
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
            # Candidate.from_dict can potentially return a list of Candidates
            candidate = Candidate.from_dict(candidate)
            if isinstance(candidate, list):
                self.candidates.extend(candidate)
                return
        assert isinstance(candidate, Candidate), "candidate must be a Candidate or a dictionary"
        self.candidates.append(candidate)

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
        self.add_candidate(config)

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
    def _generate_response(candidate: Candidate, eval_obj: Eval) -> Eval:
        """
        Generates the responses from the Candidate/LLM for a particular Eval. Ensure that the
        Candidate and Eval objects are cloned before calling this method. This is especially
        important for the Candidate object to ensure that each Eval is run against a unique
        Candidate in memory (i.e. if the Candidate maintains state/history between prompts, we
        don't want to reuse the same candidate for each eval).
        """
        eval_obj = eval_obj.clone()
        eval_obj._generate_responses(candidate.clone())
        return eval_obj

    @staticmethod
    async def _async_generate_responses(candidate: Candidate, evals: list[Eval]) -> list[Eval]:
        """Generates the responses from the Candidate/LLM for a list of Evals asynchronously."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, EvalHarness._generate_response, candidate, eval_obj)
            for eval_obj in evals
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _run_evals(
        candidate: Candidate,
        evals: list[Eval],
        async_batch_size: int | None = 50,
        callback: Callable | None = None) -> list[EvalResult]:
        """
        Evaluates the Evals against a Candidate. This method is responsible creating the batches
        of Evals and running them asynchronously. It is also responsible for executing the checks
        and returning the EvalResults.
        """
        eval_batch_size = len(evals) if async_batch_size is None else async_batch_size
        assert eval_batch_size >= 1
        results = []
        for i in range(0, len(evals), eval_batch_size):
            eval_batch = evals[i:i + eval_batch_size]
            # generate responses, potentially async
            if eval_batch_size > 1:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(
                    EvalHarness._async_generate_responses(candidate, eval_batch),
                )
            else:
                responses = [
                    EvalHarness._generate_response(candidate, eval_obj)
                    for eval_obj in eval_batch
                ]
            # Run tasks that are heavier on the CPU and shouldn't be async
            for response in responses:
                eval_result = response._execute_checks()
                results.append(eval_result)
                if callback:
                    callback(eval_result)
        return results

    def __call__(self) -> list[list[EvalResult]]:
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
        """
        num_cpus = self.num_cpus
        if num_cpus == 1:
            return [
                EvalHarness._run_evals(
                    candidate=candidate,
                    evals=self.evals,
                    async_batch_size=self.async_batch_size,
                    callback=self.callback,
                )
                for candidate in self.candidates
            ]
        if num_cpus is None or num_cpus < 1:
            num_cpus = os.cpu_count()
        results = []
        for i in range(0, len(self.candidates), num_cpus):
            candidate_batch = self.candidates[i:i + num_cpus]
            eval_list = [self.evals for _ in candidate_batch]
            batch_sizes = [self.async_batch_size for _ in candidate_batch]
            callbacks = [self.callback for _ in candidate_batch]
            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                batch_results = list(executor.map(
                    EvalHarness._run_evals,
                    candidate_batch,
                    eval_list,
                    batch_sizes,
                    callbacks,
                ))
                results.extend(batch_results)

        return results
