"""Classes and functions to eval LLMs."""
from textwrap import dedent, indent
import time
from typing import Callable, ForwardRef
from pydantic import BaseModel, Field, field_validator, root_validator
import yaml

from llm_evals.checks import (
    CHECK_REGISTRY,
    Check,
    CheckType,
    CheckResult,
    PassFailResult,
)
from llm_evals.utilities.internal_utilities import extract_valid_parameters

Eval = ForwardRef('Eval')
EvalResult = ForwardRef('EvalResult')
Candidate = ForwardRef('Candidate')


class PromptTest(BaseModel):
    """
    A PromptTest represents a prompt, an optional ideal response, and a list of checks to run
    against the response.

    Checks are optional because even a test without checks still collects performance information
    (e.g. characters per second) as well as responses (which can be visually/subjectively compared
    against either the ideal response or the responses from other LLMs).

    A PromptTest only contains information, it is not directly callable. The reason for this is so
    that all PromptTests can be evaluated sequentially, in order to accurately measure performance.
    """

    prompt: str
    ideal_response: str | None = None
    checks: list[Check | dict] | None = Field(
        default=None,
        description='A list of checks to run against the response. If a dictionary is provided, the checks need to be registered with the CHECK_REGISTRY. The dictionary needs a `check_type` key with the registration value.',  # noqa
    )

    @root_validator(pre=True)
    def process_checks(cls, values):  # noqa
        checks = values.get('checks', None)

        if checks is None:
            return values

        checks_created = []
        for check in checks:
            if isinstance(check, dict):
                check_type = check.pop('check_type')
                assert check_type, "Check dictionary must contain a 'check_type' key"
                check_instance = CHECK_REGISTRY.create_instance(check_type=check_type, params=check)
                checks_created.append(check_instance)
            elif isinstance(check, Check):
                checks_created.append(check)
            else:
                raise TypeError("Checks must be either a Check instance or a dictionary")

        values['checks'] = checks_created
        return values

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
        # model_dump doesn't include `value` on Check objects unless called directly?
        # return self.model_dump(exclude_defaults=True, exclude_none=True)
        value = {'prompt': self.prompt}
        if self.ideal_response:
            value['ideal_response'] = self.ideal_response
        if self.checks:
            value['checks'] = [c.to_dict() for c in self.checks]
        return value


class Candidate(BaseModel):
    """
    A Candidate describes an LLM that may optionally be associated with specific parameters or
    hardware.
    """

    model: Callable[[str], str]
    uuid: str | None = None
    name: str | None = None
    description: str | None = None
    parameters: dict | None = None
    system_info: dict | None = None

    def __call__(self, prompt: str) -> str:
        """Returns a response (string) given a prompt (string)."""
        return self.model(prompt)

    @classmethod
    def from_yaml(cls, path: str) -> Candidate:  # noqa: ANN102
        """
        Creates a Candidate object from a YAML file. This method requires the underlying model
        (callable) be registered with the ModelRegistry before calling this method.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        model_type = config.pop('type')
        # lookup model registry based on type
        config['model'] = lambda x: x
        return cls(**config)

    def __str__(self) -> str:
        """Returns a string representation of the Candidate."""
        parameters = '' if not self.parameters else f'\n            parameters={self.parameters},'
        system_info = '' if not self.system_info else f'\n            system_info={self.system_info},'  # noqa
        return dedent(f"""
        {self.__class__.__name__}(
            uuid={self.uuid},
            name={self.name},
            description={self.description},{parameters}{system_info}
        )
        """).strip()


class Eval(BaseModel):
    """
    An Eval defines a set of one or more prompts and tests that can be used to evaluate an LLM. If
    more than one prompt is provided, the intent is evaluate the the conversation and, therefore,
    it's expected that the underlying model/object will maintain state between prompts.

    The Eval object is evaluated by calling it with a single model_id and a callable (wrapping the
    LLM) that takes a prompt (string) and returns a response (string).

    An Eval corresponds to a set of prompts, while the result of the Eval corresponds to the Eval
    and a specific LLM, and potentially specific to the hardware used to run the LLM.

    The tests are ran after all the prompts have been evaluated. Each test is passed a list of
    responses (strings) and returns a TestResult object.
    """

    test_sequence: list[PromptTest | dict] = Field(
        description='A list of prompts and tests to run against the LLM.',
    )
    metadata: dict | None = Field(
        default=None,
        description='Metadata associated with the Eval.',
    )
    uuid: str | None = Field(
        default=None,
        description='Used to uniquely identify the Eval which is ultimately used to avoid running the same Eval (against the same Candidate/llm) more than once.',  # noqa
        )
    version: str | int | float | None = Field(
        default=None,
        description='Version of the Eval.',
    )
    result: EvalResult | None = Field(
        default=None,
        description='The result of evaluating the Eval.',
    )

    @root_validator(pre=True)
    def process_tests(cls, values: dict) -> dict:  # noqa: N805
        """Converts test_sequence to a list of PromptTest objects."""
        test_sequence = values.get('test_sequence', [])
        tests_created = []
        for test in test_sequence:
            if isinstance(test, dict):
                tests_created.append(PromptTest(**test))
            elif isinstance(test, PromptTest):
                tests_created.append(test)
            else:
                raise TypeError("test_sequence must be either a PromptTest instance or a dictionary")  # noqa
        values['test_sequence'] = tests_created
        return values

    def to_dict(self) -> dict:
        """Return a dictionary representation of the PromptTest."""
        # return self.model_dump(exclude_defaults=True, exclude_none=True) doesn't seem to work
        # recursively, there are a couple of inconsistencies
        value = {'test_sequence': [t.to_dict() for t in self.test_sequence]}
        if self.uuid:
            value['uuid'] = self.uuid
        if self.version:
            value['version'] = self.version
        if self.metadata:
            value['metadata'] = self.metadata
        if self.result:
            value['result'] = self.result.to_dict()
        return value

    @classmethod
    def from_yaml(cls, path: str) -> Eval:  # noqa: ANN102
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

    def __call__(self, candidate: Candidate | Callable) -> EvalResult:
        """Evaluates the model against the prompts and tests."""
        start = time.time()
        responses = [candidate(p.prompt) for p in self.test_sequence]
        end = time.time()
        results = []
        for test, response in zip(self.test_sequence, responses):
            check_results = []
            # this probably won't work with CODE blocks
            # TODO: how do i retain the enviornment of the code block so during the next round of
            # I can execute the subsequent code block in the same environment as the first code
            # blocks?
            # TODO: Extract and report number of code blocks generated
            # If there are code blocks generated, run ...???
            # only need to run if there are CODE_BLOCKS_RUN or CODE_BLOCKS_EVIRONMENT
            # i don't like this dependency
            # hnmm.m.m.mm.m..m..m.mm.m.
            # if there are code blocks. How do i know they are python? I don't. I need to know
            # I can't run them here, this code shouldn't know how to run them because someone
            # needs to register the the check which knows how to run them and which language they
            # are, but i do need to distract them
            code_blocks = []  # TODO: extract_code_blocks()

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

        self.result = EvalResult(
            eval_obj=self,
            candidate_obj=candidate,
            responses=responses,
            results=results,
            total_time_seconds=end - start,
        )
        return self.result

    def __str__(self) -> str:
        """Returns a string representation of the Eval."""
        if self.test_sequence:
            test_sequence = '[\n'
            indent_value = ' ' * 16
            test_sequence += ',\n'.join([indent(str(s), indent_value) for s in self.test_sequence])
            test_sequence += '\n            ]'
        else:
            test_sequence = '[]'
        metadata = '' if not self.metadata else f'\n            metadata={self.metadata},'
        return dedent(f"""
        Eval(
            uuid={self.uuid},{metadata}
            test_sequence={test_sequence},
        )
        """).strip()


class EvalResult(BaseModel):
    """
    An EvalResult is the result of evaluating a specific LLM against a specific Eval, potentially
    using specific hardware. The hardware is not applicable for services like OpenAI's API, but
    would be applicable for running locally or against specific/configurable hardware like Hugging
    Face Endpoints or a custom server. The quality of responses might not change between hardware,
    but the speed of responses could.
    """

    eval_obj: 'Eval'
    candidate_obj: 'Candidate'
    responses: list[str]
    total_time_seconds: float
    results: list[list['CheckResult']]

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

    @field_validator('total_time_seconds')
    def validate_total_time_seconds(cls, v: float) -> float:  # noqa: N805
        """Validates the total_time_seconds field."""
        if v <= 0:
            raise ValueError("Total time must be greater than zero")
        return v

    @property
    def num_pass_fail_checks(self) -> int:
        """Returns the number of pass/fail checks."""
        pass_fail_results = [
            result for prompt_test in self.results for result in prompt_test
            if isinstance(result, PassFailResult)
        ]
        return len(pass_fail_results)

    @property
    def num_passing_checks(self) -> int | None:
        """Returns the number of passing checks. If there are no pass/fail checks, returns None."""
        pass_fail_results = [
            result for prompt_test in self.results for result in prompt_test
            if isinstance(result, PassFailResult)
        ]
        if not pass_fail_results:
            return None
        return sum(r.success for r in pass_fail_results)

    @property
    def perc_passed_checks(self) -> float | None:
        """
        Returns the percentage of passing checks.
        If there are no pass/fail checks, returns None.
        """
        num_passing = self.num_passing_checks
        if num_passing is None:
            return None
        return num_passing / self.num_pass_fail_checks

    def all_results(self) -> list['CheckResult']:
        """Returns a (flattened) list of all CheckResults."""
        return [r for result in self.results for r in result]

    def __str__(self) -> str:
        return dedent(f"""
        EvalResult:
            # of Response Characters={self.response_characters}
            Total Time (seconds)={self.total_time_seconds}
            Characters per Second={self.characters_per_second}
            # of Checks={self.num_checks}
            # of Pass/Fail Checks={self.num_pass_fail_checks}
            # of Passing Checks={self.num_passing_checks}
            % of Passing Checks={self.perc_passed_checks}
        """).strip()



# class EvalResult:
#     """
#     An EvalResult is the result of evaluating a specific LLM against a specific Eval, potentially
#     using specific hardware. The hardware is not applicable for services like OpenAI's API, but
#     would be applicable for running locally or against specific/configurable hardware like Hugging
#     Face Endpoints or a custom server. The quality of responses might not change between hardware,
#     but the speed of responses could.
#     """

#     def __init__(
#             self,
#             eval_obj: 'Eval',
#             candidate_obj: Candidate,
#             responses: list[str],
#             total_time_seconds: float,
#             results: list[list[CheckResult]],
#             ) -> None:
#         self.eval_obj = eval_obj
#         self.candidate_obj = candidate_obj
#         self.responses = responses
#         self.total_time_seconds = total_time_seconds
#         self.response_characters = sum([len(r) for r in responses])
#         total_time_seconds += 1e-6  # prevent divide by zero
#         self.characters_per_second = sum([len(r) for r in responses]) / total_time_seconds
#         self.results = results
#         self.num_checks = sum([len(r) for r in results])
#         # Each item of the outer list is a list of CheckResults. Each item of the inner list is a
#         # CheckResult.
#         pass_fail_results = [
#             result for prompt_test in results for result in prompt_test
#             if isinstance(result, PassFailResult)
#         ]
#         self.num_pass_fail_checks = len(pass_fail_results)
#         if self.num_pass_fail_checks == 0:
#             self.num_passing_checks = None
#             self.perc_passed_checks = None
#         else:
#             self.num_passing_checks = sum(r.success for r in pass_fail_results)
#             self.perc_passed_checks = self.num_passing_checks / self.num_pass_fail_checks

#     def all_results(self) -> list[CheckResult]:
#         """Returns a list of all CheckResults."""
#         return [r for result in self.results for r in result]

#     def __str__(self) -> str:
#         """Returns a string representation of the EvalResult."""
#         return dedent(f"""
#         EvalResult:
#             # of Response Characters={self.response_characters}
#             Total Time (seconds)={self.total_time_seconds}
#             Characters per Second={self.characters_per_second}
#             # of Checks={self.num_checks}
#             # of Pass/Fail Checks={self.num_pass_fail_checks}
#             # of Passing Checks={self.num_passing_checks}
#             % of Passing Checks={self.perc_passed_checks}
#         """).strip()



Eval.model_rebuild()


def summarizer(result: EvalResult) -> dict:
    """Simple summarizer that returns a dictionary of summary statistics."""
    code_run_checks = [
        r for r in result.all_results()
        if r.metadata.get('check_type', '') == CheckType.PYTHON_CODE_BLOCKS_RUN.name
    ]
    summary = {}
    if result.eval_obj.uuid:
        summary['eval_uuid'] = result.eval_obj.uuid
    if result.candidate_obj.uuid:
        summary['candidate_uuid'] = result.candidate_obj.uuid
    summary['num_prompts'] = len(result.responses)
    summary['response_characters'] = result.response_characters
    summary['total_time_seconds'] = result.total_time_seconds
    summary['characters_per_second'] = result.characters_per_second
    summary['num_checks'] = result.num_checks
    summary['num_pass_fail_checks'] = result.num_pass_fail_checks
    summary['num_passing_checks'] = result.num_passing_checks
    summary['perc_passed_checks'] = result.perc_passed_checks
    summary['num_code_blocks'] = sum(r.metadata['num_code_blocks'] for r in code_run_checks)
    if summary['num_code_blocks'] == 0:
        summary['num_code_blocks_successful'] = None
        summary['perc_code_blocks_successful'] = None
    else:
        summary['num_code_blocks_successful'] = sum(r.metadata['num_code_blocks_successful'] for r in code_run_checks)  # noqa
        summary['perc_code_blocks_successful'] = summary['num_code_blocks_successful'] / summary['num_code_blocks']  # noqa
    return summary
