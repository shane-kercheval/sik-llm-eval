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
from textwrap import dedent, indent
from typing import Any, Callable
from llm_eval.candidates import Candidate, is_async_candidate
from llm_eval.checks import (
    Check,
    CheckResult,
    CheckType,
    PassFailResult,
    PythonCodeBlockTests,
    ResponseData,
)
from llm_eval.internal_utilities import (
    DictionaryEqualsMixin,
    extract_code_blocks,
    has_property,
)


# class PromptTest(DictionaryEqualsMixin):
#     """
#     A PromptTest represents a prompt, an optional ideal response, and a list of checks to run
#     against the response. There can be one or more PromptTests in an Eval. If more than one
#     PromptTest is provided, the intent is to evaluate a conversation (multiple sequential
#     prompts/responses) and, therefore, it's expected that the underlying Candidate (model/client)
#     will maintain state, if needed, between each PromptTest.

#     Although most PromptTests will contain 'checks', checks are optional because even a test
#     without checks still collects performance information (e.g. characters per second) as well as
#     responses (which can be visually/subjectively compared against either the ideal response or the
#     responses from other LLMs).

#     A PromptTest only contains information (it is not directly callable). An Eval object is
#     responsible for calling the PromptTest(s), executing the checks, and returning a EvalResult
#     object.
#     """

#     def __init__(
#             self,
#             prompt: str | dict | list,
#             ideal_response: str | None = None,
#             checks: list[Check | dict | Callable[[Any], CheckResult]] | None = None) -> None:
#         """
#         Initializes the PromptTest.

#         Note: More than one PythonCodeBlockTests check is not allowed. This is
#         because the PythonCodeBlockTests check runs all code blocks across all responses and
#         therefore should only be added once so the code blocks are not re-executed multiple times.

#         Args:
#             prompt:
#                 The prompt to send to the LLM.
#             ideal_response:
#                 The ideal response to compare against the LLM's response.
#             checks:
#                 'A list of checks to run against the response. If a dictionary is provided, the
#                 Check subclasses need to be registered via `Check.register(...)`.
#                 The dictionary needs a `check_type` key with the registration value.
#                 If a callable is provided, it will not be cloned and so should not have any state.
#                 Additionally it cannot be serialized/deserialized via to_dict()/from_dict() since
#                 the underlying callable will not have those methods.
#         """
#         self.prompt = dedent(prompt).lstrip() if isinstance(prompt, str) else prompt
#         self.ideal_response = dedent(ideal_response) if ideal_response else None
#         checks = checks or []
#         if not isinstance(checks, list):
#             checks = [checks]
#         checks_created = []
#         for check in checks:
#             if isinstance(check, dict):
#                 assert 'check_type' in check, "Check dictionary must contain a 'check_type' key"
#                 checks_created.append(Check.from_dict(check))
#             elif isinstance(check, (Callable, Check)):
#                 checks_created.append(check)
#             else:
#                 raise TypeError("Checks must be either a Check, dictionary, or callable.")
#         # Cannot add more than one PythonCodeBlockTests check
#         if len([c for c in checks_created if isinstance(c, PythonCodeBlockTests)]) > 1:
#             raise ValueError("Cannot add more than one PythonCodeBlockTests check")
#         self.checks = checks_created

#     def __str__(self) -> str:
#         """Returns a string representation of the PromptTest."""
#         if self.checks:
#             indent_value = ' ' * 16
#             checks = '[\n' + indent_value
#             checks += f',\n{indent_value}'.join([str(c) for c in self.checks]) if self.checks else ''  # noqa: E501
#             checks += '\n            ]'
#         else:
#             checks = '[]'
#         if self.ideal_response:
#             ideal_response = self.ideal_response.strip()
#             if len(ideal_response) > 50:
#                 ideal_response = ideal_response[0:50] + '...'
#             ideal_response = f'\n            ideal_response="{ideal_response}",'
#         else:
#             ideal_response = ''
#         return dedent(f"""
#         {self.__class__.__name__}(
#             prompt='{self.prompt}',{ideal_response}
#             checks={checks},
#         )
#         """).strip()

#     def to_dict(self) -> dict:
#         """
#         Return a dictionary representation of the PromptTest.

#         NOTE: if the underlying checks do not have a to_dict method, (e.g. lambda function) they
#         will be converted to a string. This means that the check cannot be deserialized via
#         from_dict.
#         """
#         value = {'prompt': self.prompt}
#         if self.ideal_response:
#             value['ideal_response'] = self.ideal_response
#         if self.checks:
#             value['checks'] = [
#                 c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in self.checks
#             ]
#         return value

#     def clone(self) -> 'PromptTest':
#         """
#         Returns a copy of the PromptTest with the same state.

#         NOTE: This method only clones checks that have a clone method. If a check does not have a
#         clone method (e.g. lambda function), it will not be cloned and it is assumed that the check
#         is stateless and multiple usage of the check will not cause side effects.
#         """
#         return PromptTest(
#             prompt=deepcopy(self.prompt),
#             ideal_response=self.ideal_response,
#             checks=[c.clone() if hasattr(c, 'clone') else c for c in self.checks],
#         )


class Eval(DictionaryEqualsMixin):
    """
    An Eval is single scenario (i.e. one or more prompts) that the user is interested in
    evaluating (via one or more checks), which is encapsulated in a PromptTest object.  Multiple
    prompts can be used to test conversations (i.e. sequential prompt/response exchanges where the
    assumption is the LLM client maintains conversational history) encapsulated in a list of
    PromptTests. Additionally, each Eval is associated with custom "checks". Examples of checks
    include: if the response matches an exact value, if it contain a particular value, if it
    contain code blocks, if those code blocks run, if the variables/functions/etc created by those
    code blocks contain the expected values/behavior.

    An Eval is a callable object that is executed by calling it with a Candidate object, or a
    dictionary representing a Candidate that has been registered via `Candidate.register(...)`.
    """

    def __init__(
            self,
            input: str | dict | list | Any,  # noqa: A002, ANN401
            checks: list[Check | dict | Callable[[Any], CheckResult]] | None = None,
            ideal_response: str | None = None,
            metadata: dict | None = None) -> None:
        """
        Initializes the Eval.

        Note: More than one PythonCodeBlockTests check cannot be added across all tests. This is
        because the PythonCodeBlockTests check runs all code blocks across all responses and
        therefore should only be added once so the code blocks are not re-executed multiple times.
        The PythonCodeBlockTests check should be added to the last PromptTest in the sequence.

        Args:
            metadata:
                Metadata associated with the Eval.
        """
        self._has_executed = False
        self._candidate = None
        self._responses = None
        self._duration = None
        self.metadata = deepcopy(metadata) or {}
        self.input = input
        self.ideal_response = dedent(ideal_response) if ideal_response else None
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
        # Cannot add more than one PythonCodeBlockTests check
        if len([c for c in checks_created if isinstance(c, PythonCodeBlockTests)]) > 1:
            raise ValueError("Cannot add more than one PythonCodeBlockTests check")
        self.checks = checks_created

    def to_dict(self) -> dict:
        """Return a dictionary representation of the PromptTest."""
        value = {}
        if self.metadata:
            value['metadata'] = self.metadata
        value['input'] = self.input
        if self.ideal_response:
            value['ideal_response'] = self.ideal_response
        if self.checks:
            value['checks'] = [
                c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in self.checks
            ]
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

        This method has side effects of setting self._responses, self._duration, and
        self._candidate, which are used by _execute_checks. This is bad practice but we need to do
        this to support calling _generate_response async and then executing the checks afterwards.
        """
        self._candidate = self._to_candidate(candidate)
        start = time.time()
        self._response = None
        try:
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
        code_blocks = []
        # for test, response in zip(self.prompt_sequence, self._responses):
        check_results = []
        if isinstance(self.response, str):
            code_blocks.extend(extract_code_blocks(self.response))
        else:
            code_blocks = []
        data = ResponseData(
            input=self.input,
            ideal_response=self.ideal_response,
            response=self.response,
            code_blocks=code_blocks,
        )
        for check in self.checks:
            check_results.append(check(data))

        return EvalResult(
            eval_obj=self,
            candidate_obj=self._candidate,
            responses=self._responses,
            total_time_seconds=self._duration,
            num_code_blocks=len(code_blocks),
            cost = self._candidate.cost if has_property(self._candidate, 'cost') else None,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            check_results=check_results,
        )

    def __call__(self, candidate: Candidate | Callable[[Any], Any] | dict) -> 'EvalResult':
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
        if self._has_executed:
            raise RuntimeError("Eval has already been executed; create a new Eval object")
        self._has_executed = True
        self._generate_response(candidate)
        # _generate_response has side effects of setting self._responses, self._duration, and
        # self._candidate that _execute_check relies on;
        # this is bad practice but we need to do this to support calling _generate_response async
        # and then executing the checks afterwards
        results = self._execute_checks()
        # these fields should be reset so we don't accidentally use them again; they should not be
        # accessed directly; they are only used to store information between running
        # _generate_response and _execute_checks
        self._candidate = None
        self._responses = None
        self._duration = None
        return results

    def __str__(self) -> str:
        """Returns a string representation of the Eval."""
        if self.prompt_sequence:
            prompt_sequence = '[\n'
            indent_value = ' ' * 16
            prompt_sequence += ',\n'.join(
                [indent(str(s), indent_value) for s in self.prompt_sequence],
            )
            prompt_sequence += '\n            ]'
        else:
            prompt_sequence = '[]'
        metadata = '' if not self.metadata else f'            metadata={self.metadata},\n{" " * 12}'  # noqa
        return dedent(f"""
        Eval(
            {metadata}prompt_sequence={prompt_sequence},
        )
        """).strip()

    def clone(self) -> 'Eval':
        """
        Returns a copy of the Candidate with the same state but with a different instance of the
        underlying model (e.g. same parameters but reset history/context).
        """
        # return Eval(**deepcopy(self.to_dict()))
        return Eval(
            prompt_sequence=[p.clone() for p in self.prompt_sequence],
            system_message=self.system_message,
            previous_messages=deepcopy(self.previous_messages),
            metadata=deepcopy(self.metadata),
        )


# class PromptComparison:
#     """
#     The intent of this class is to form an interface for defining/creating multiple PromptTest
#     objects that access different prompts across the same set of checks, which is usedful
#     for prompt engineering.

#     A PromptTest object is returned for each of the prompts provided. The checks and ideal response
#     are shared across all PromptTest objects.

#     `prompt_parameters` can be used to share text across prompts. For example, if you are using
#     few-shot learning, you can use `prompt_parameters` to share the the same few-shot example
#     across all prompts.

#     This class could (should?) have been implemented as a function, but it is implemented as a
#     callable class to provide a consistent interface with the other classes in the llm_eval module
#     when defining evaluations (e.g. PromptTest, MultiEval).
#     """

#     def __init__(self,
#         prompts: list[str | dict],
#         prompt_parameters: dict | None = None,
#         checks: list[Check | dict] | None = None,
#         ideal_response: str | None = None) -> None:
#         """
#         Args:
#             prompts:
#                 A list of prompts that can be either strings or dictionaries. If dictionaries are
#                 used, the dictionary must contain a 'prompt' key. Future versions may support
#                 additional keys.
#             prompt_parameters:
#                 `prompt_parameters` is a dictionary where the keys correspond to placeholders in
#                 the prompts (e.g. key is 'value_a' and the prompt is 'Here is my prompts with
#                 {value_a}') and the values will replace the placeholders (e.g. '{value_a}') in the
#                 prompts.
#             checks:
#                 List of checks to run against the response. All PromptTest objects will share the
#                 same checks.
#             ideal_response:
#                 Optional ideal response for the prompts. Currently not used in llm_eval.
#         """
#         self.prompts = [str(p['prompt']) if isinstance(p, dict) else str(p) for p in prompts]
#         self.prompt_parameters = prompt_parameters or {}
#         self.checks = [Check.from_dict(c) if isinstance(c, dict) else c for c in checks or []]
#         self.ideal_response = str(ideal_response) if ideal_response is not None else None

#     def __call__(self) -> list[PromptTest]:
#         """Creates a list of PromptTest objects from the PromptComparison."""
#         tests = []
#         for prompt in self.prompts:
#             tests.append(PromptTest(
#                 prompt=prompt.format(**self.prompt_parameters) if self.prompt_parameters else prompt,  # noqa
#                 ideal_response=self.ideal_response,
#                 checks=self.checks,
#             ))
#         return tests


# class MultiEval:
#     """
#     A MultiEval object describes and validates the structure required to create one or more evals
#     from either multiple system messages, multiple sets of previous messages, and/or multiple
#     prompts.

#     Unlike an Eval object, A MultiEval object does not generate LLM responses or execute checks.
#     It is simply a mechanism to create multiple Eval objects from a single dictionary. However,
#     MultiEvals can be passed to the TestHarness in the same way as Eval objects, and the
#     TestHarness will automatically create multiple Eval objects from the MultiEval object.

#     For example, if the `system_message` is a list of strings, the intent intent is to create two
#     Eval objects (duplicated for each system_message).
#     """

#     def __init__(self,
#             prompts: list[PromptTest | dict] | PromptComparison | PromptTest | dict,
#             system_message: list[str] | str | None = None,
#             previous_messages: list[dict | tuple | list] | None = None,
#             metadata: dict | None = None,
#         ) -> None:
#         """
#         Args:
#             prompts:
#                 If a single dictionary is passed in, it is assumed to represent a single PromptTest
#                 object (and, therefore, a single Eval object) or a single PromptComparison object
#                 (and, therefore, multiple Eval objects). Please see those classes for the expected
#                 fields (and corresponding dictionary structure).

#                 If a list of PromptTest objects is passed in, it is assumed to represent a single
#                 eval with one or more sequential prompts.

#                 If a list of dictionaries is passed in, it is assumed to represent either a single
#                 PromptTest object or a single PromptComparison object.
#             system_message:
#                 Either a single system message or a list of system messages. If multiple system
#                 messages are passed in, multiple Eval objects will be created (one for each system
#                 message), which will share the same prompts and previous messages.
#             previous_messages:
#                 Previous messages are user/assistant pairs to set the state of the LLM. Each pair
#                 should either be a dictionary or a tuple. If a dictionary is used then it should
#                 contain a 'user' key and an 'assistant' key. If a tuple is used, it should contain
#                 two items: the user message in the first position and the assistant message in the
#                 second position.

#                 Either a list of previous messages or a list of lists of previous messages can be
#                 passed in. If a list of lists is passed in, it is assumed that each outer list
#                 represents a single Eval object and each inner list represents a list of previous
#                 messages for that Eval object. Therefore a list of lists will create multiple Eval
#                 objects, one for each inner list of previous messages. Each Eval object will share
#                 the same prompts and system message.
#             metadata:
#                 Metadata shared across all Eval objects.
#         """
#         if system_message is None or isinstance(system_message, list):
#             self.system_message = system_message
#         else:
#             self.system_message = [system_message]
#         assert previous_messages is None or isinstance(previous_messages, list), \
#             "previous_messages must be a list"
#         if previous_messages and isinstance(previous_messages[0], (dict, tuple)):
#             self.previous_messages = [previous_messages]
#         else:
#             self.previous_messages = previous_messages
#         self.metadata = metadata or {}

#         if isinstance(prompts, dict):
#             # A single dictionary can represent a single PromptTest or a single PromptComparison.
#             # A dictionary representing a PromptTest will contain a `prompt` key; a dictionary
#             # representing a PromptComparison will contain a `prompts` key.
#             if 'prompt' in prompts:
#                 prompts = [PromptTest(**prompts)]
#             else:
#                 assert 'prompts' in prompts, "Invalid dictionary; expected 'prompt' or 'prompts'"
#                 # type(prompts['checks'][0])
#                 # PromptComparison(prompts=prompts['prompts'], checks=prompts.get('checks'))
#                 prompts = PromptComparison(**prompts)()
#         elif isinstance(prompts, PromptComparison):
#             prompts = prompts()
#         elif isinstance(prompts, PromptTest):
#             prompts = [prompts]
#         else:
#             # either a list of PromptTest objects or a list of dictionaries representing a
#             # PromptTest object; either way, there are multiple objects but only for a single eval
#             # so we need to wrap it in another list
#             assert isinstance(prompts, list)
#             assert all(isinstance(p, (PromptTest, dict)) for p in prompts)
#             prompts = [
#                 PromptTest(**p) if isinstance(p, dict) else p for p in prompts
#             ]
#             prompts = [prompts]

#         assert isinstance(prompts, list), "prompts must be a list"
#         self.prompts = prompts

#     def __call__(self) -> list[Eval]:
#         """Creates a list of Eval objects from the MultiEval object."""
#         evals = []
#         for system_message in self.system_message or [None]:
#             for previous_messages in self.previous_messages or [None]:
#                 for prompt_sequence in self.prompts:
#                     evals.append(Eval(
#                         prompt_sequence=prompt_sequence,
#                         system_message=system_message,
#                         previous_messages=previous_messages,
#                         metadata=self.metadata,
#                     ))
#         return evals

#     @classmethod
#     def from_dict(cls, config: dict) -> list[Eval]:  # noqa: ANN102
        # """
        # Parse a dictionary into a MultiEval object.

        # A dictionary can either have `prompt_comparison` key which represents a single
        # PromptComparison object (which in turn represents multiple PromptTest objects), or a
        # `prompt_sequence` key which represents a list of PromptTest objects.
        # """
        # config = deepcopy(config)
        # prompts = config.get('prompt_comparison') or config.get('prompt_sequence')
        # return cls(
        #     prompts=prompts,
        #     system_message=config.get('system_message'),
        #     previous_messages=config.get('previous_messages'),
        #     metadata=config.get('metadata'),
        # )


class EvalResult(DictionaryEqualsMixin):
    """
    An EvalResult is the result of evaluating a specific Candidate/LLM against a specific
    Eval.
    """

    def __init__(
        self,
        eval_obj: Eval | dict,
        candidate_obj: Candidate | dict,
        response: str | object,
        total_time_seconds: float,
        num_code_blocks: int,
        cost : float | None,
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
            total_time_seconds:
                The total time (in seconds) it took to run the Eval.
            num_code_blocks:
                The total number of code blocks generated across all responses.
            cost:
                The cost associated with the candidate. This is optional and only applicable to
                candidates that have a `cost` property.
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
            raise TypeError("candidate_obj must be either a Candidate or a dictionary")
        self.response = response
        self.total_time_seconds = total_time_seconds
        self.num_code_blocks = num_code_blocks
        self.cost = cost
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

    # @property
    # def response_characters(self) -> int | None:
    #     """Returns the number of characters across all responses."""
    #     if not self.responses or not isinstance(self.responses[0], str):
    #         return None
    #     return sum(len(r) for r in self.responses)

    # @property
    # def characters_per_second(self) -> float | None:
    #     """Returns the number of characters per second across all responses."""
    #     if not self.responses or not isinstance(self.responses[0], str):
    #         return None
    #     # Adding a tiny value to prevent divide-by-zero error
    #     return sum(len(r) for r in self.responses) / (self.total_time_seconds + 1e-6)

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

    @property
    def expects_code_blocks(self) -> bool:
        """Returns a list of CheckResults for code block present checks."""
        return any(
            r for r in self.check_results
            if r.metadata.get('check_type', '') == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
        )

    def get_code_block_tests_result(self) -> CheckResult | None:
        """
        Only applicable for PythonCodeBlockTests (PYTHON_CODE_BLOCK_TESTS) checks.

        Returns the CheckResult object associated with the PythonCodeBlockTests check, if it
        exists, otherwise None.
        """
        results = [
            r for r in self.check_results
            if r.metadata.get('check_type', '') == CheckType.PYTHON_CODE_BLOCK_TESTS.name
        ]
        if results:
            assert len(results) == 1
            return results[0]
        return None

    def get_num_code_blocks_successful(self) -> int | None:
        """
        Only applicable for PythonCodeBlockTests (PYTHON_CODE_BLOCK_TESTS) checks.

        Returns the number of code blocks generated that successfully execute across all responses.
        If there are no code blocks or no PythonCodeBlockTests check, returns None.
        """
        result = self.get_code_block_tests_result()
        if result:
            return result.metadata.get('num_code_blocks_successful', None)
        return None

    def get_num_code_tests_defined(self) -> int | None:
        """
        Only applicable for PythonCodeBlockTests (PYTHON_CODE_BLOCK_TESTS) checks.

        Returns the number of code tests defined (i.e. the number of individual tests for the
        PythonCodeBlockTests check, if it exists). If there are no code blocks or no
        PythonCodeBlockTests check, returns None.
        """
        result = self.get_code_block_tests_result()
        if result:
            return result.metadata.get('num_code_tests', None)
        return None

    def get_num_code_tests_successful(self) -> int | None:
        """
        Only applicable for PythonCodeBlockTests (PYTHON_CODE_BLOCK_TESTS) checks.

        Returns the number of code tests (i.e. the number of individual tests for the
        PythonCodeBlockTests check, if it exists) that successfully pass. If there
        are no code blocks or no PythonCodeBlockTests check, returns None.
        """
        result = self.get_code_block_tests_result()
        if result:
            return result.metadata.get('num_code_tests_successful', None)
        return None

    def __str__(self) -> str:
        cost_str = f'\n{" " * 12}Cost:{" " * 22} ${self.cost:.4f}' if self.cost else ''
        # check if candidate_obj has metadata field
        candidate_name = self.candidate_obj.metadata.get('name', '')
        if candidate_name:
            candidate_name = f"Candidate:{' ' * 18}{candidate_name}\n{' ' * 12}"
        eval_name = self.eval_obj.metadata.get('name', '')
        if eval_name:
            eval_name = f"Eval:{' ' * 24}{eval_name}\n{' ' * 12}"
        # response_characters and characters_per_second are properties that can return None
        if self.response_characters is None:
            response_characters = 'N/A'
            characters_per_second = 'N/A'
        else:
            response_characters = f'{self.response_characters:,}'
            characters_per_second = f'{self.characters_per_second:,.1f}'
        result = f"""
        EvalResult:
            {candidate_name}{eval_name}# of Prompts Tested:         {len(self.eval_obj.prompt_sequence)}{cost_str}
            Total Response Time:         {self.total_time_seconds:0.1f} seconds
            # of Response Characters:    {response_characters}
            Characters per Second:       {characters_per_second}
            # of Checks:                 {self.num_checks}
            # of Successful Checks:      {self.num_successful_checks}
            % of Successful Checks:      {self.perc_successful_checks or 0:.1%}
            # of Code Blocks Generated:  {self.num_code_blocks}
        """  # noqa
        result = result.rstrip()
        if self.get_code_block_tests_result():
            result += f"""
            # of Successful Code Blocks: {self.get_num_code_blocks_successful()}
            # of Code Tests Defined:     {self.get_num_code_tests_defined()}
            # of Successful Code Tests:  {self.get_num_code_tests_successful()}
            """
        return dedent(result).strip()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the EvalResult."""
        return {
            'eval_obj': self.eval_obj.to_dict(),
            'candidate_obj': self.candidate_obj.to_dict(),
            'responses': self.responses,
            'total_time_seconds': self.total_time_seconds,
            'num_code_blocks': self.num_code_blocks,
            'cost': self.cost,
            'timestamp': self.timestamp,
            'results': [[r.to_dict() for r in result] for result in self.results],
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

    Candidates must implement the clone() function if they contain state. This is necessary because
    we need a consistant way of cloning the Candidate for each Eval so that we can run multiple
    Evals against the same Candidate without affecting the state of the Candidate. Stateless
    candidates do not need to implement the clone() function.

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
            # evals: list[Eval | dict | MultiEval] | Eval | MultiEval | dict | None = None,
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
                `harness_exception` property and the checks should have a `False` value for the
                `success` property.
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
        # if isinstance(eval_obj, dict):
        #     # a dict could be in the format of a single Eval or a MultiEval (prompt_sequence or
        #     # prompt_comparison) with single or multiple system_messages and previous_messages
        #     # MutliEval will create the necessary Eval objects
        #     self.evals.extend(MultiEval.from_dict(eval_obj)())
        # elif isinstance(eval_obj, MultiEval):
        #     self.evals.extend(eval_obj())
        if isinstance(eval_obj, Eval):
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
        Candidate and Eval objects are cloned before calling this method. This is especially
        important for the Candidate object to ensure that each Eval is run against a unique
        Candidate in memory (i.e. if the Candidate maintains state/history between prompts, we
        don't want to reuse the same candidate for each eval).
        """
        eval_obj = eval_obj.clone()
        exception = None
        try:
            eval_obj._generate_response(candidate.clone())
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
            await eval_obj._async_generate_response(candidate.clone())
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
                # We can set the responses to empty strings and run the checks to
                # accomplish this.
                # Each eval can have multiple prompts, and there could have been responses
                # that were generated before the exception was raised. We want to keep the
                # responses that were generated before the exception was raised and add
                # empty strings for the remaining responses.
                missing_responses = len(response_eval.prompt_sequence) - len(response_eval._responses)  # noqa: E501
                response_eval._responses.extend(['' for _ in range(missing_responses)])
                eval_result = response_eval._execute_checks()
                eval_result.harness_exception = exception
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
