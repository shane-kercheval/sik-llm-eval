"""TODO document."""
from inspect import signature
from textwrap import dedent, indent
import time
from typing import Callable
from pydantic import BaseModel
import yaml
from llm_evals.checks import CheckRegistery, Check, CheckType, CHECK_REGISTRY, CheckResult, PassFailResult
from llm_evals.utilities.internal_utilities import extract_valid_parameters


class PromptTest(BaseModel):
    """
    TODO document.

    Checks can be optional because even a test without checks still collects responses (that
    can be visually/subjectively compared against either the ideal response or the responses from
    other LLMs).
    """

    prompt: str
    ideal_response: str | None = None
    checks: list[Check] | None = None

    class Config:
        """Needed to allow for arbitrary types (i.e. Check object)."""

        arbitrary_types_allowed = True

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

    @classmethod
    def from_yaml(cls, path: str) -> 'Candidate':  # noqa: ANN102
        """Creates a Candidate object from a YAML file."""
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




class Eval:
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

    def __init__(
            self,
            test_sequence: list[PromptTest],
            metadata: dict | None = None,
            uuid: str | None = None,
            version: str | None = None,
            ):
        """
        Args:
            uuid: optional. Used to uniquely identify the Eval which is ultimately used to avoid
                running the same Eval (against the same Candidate/llm) more than once.
        """
        self.metadata = metadata
        self.test_sequence = test_sequence
        self.uuid = uuid
        self.version = version
        self.result = None

    @classmethod
    def from_dict(
            cls,  # noqa: ANN102
            config: dict,
            results: dict | None = None,
            registry: CheckRegistery | None = None) -> 'Eval':
        """
        Creates an Eval object from a config/dictionary.

        Any custom checks must be registered with the CheckRegistry before calling this
        method.

        For example:

        ```
        from llm_evals.checks import register_check

        @register_check('my_custom_check)
        class MyCheck(EvalCheck):
            ...
        ```
        """
        # prompts = [Scenario(**prompt) for prompt in config['test_sequence']]
        # need to register the different types of tests
        # tests = [EvalTest(**test) for test in config['tests']]
        registry = registry or CHECK_REGISTRY

        def create_checks(checks_data: list[dict]) -> list[Check]:
            checks = []
            for check_data in checks_data:
                check_type = CheckType.to_enum(check_data.pop('type'))
                check = registry.create_instance(check_type=check_type, params=check_data)
                checks.append(check)
            return checks

        test_sequence = []
        for test in config['test_sequence']:
            checks = create_checks(test.pop('checks')) if 'checks' in test else None
            test_sequence.append(PromptTest(**test, checks=checks))

        obj = cls(
            uuid=config['uuid'] if 'uuid' in config else None,
            version=config['version'] if 'version' in config else None,
            metadata=config['metadata'] if 'metadata' in config else {},
            test_sequence=test_sequence,
        )
        if results is not None:
            obj.result = EvalResult(**results)
        return obj

    @classmethod
    def from_yaml(cls, path: str, results: dict | None = None) -> 'Eval':  # noqa: ANN102
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
        return cls.from_dict(config, results)

    def __call__(self, candidate: Candidate) -> 'EvalResult':
        """Evaluates the model against the prompts and tests."""
        start = time.time()
        responses = [candidate.model(p.prompt) for p in self.test_sequence]
        end = time.time()
        self._duration = end - start
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
            total_time_seconds=self._duration,
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


class EvalResult:
    """
    An EvalResult is the result of evaluating a specific LLM against a specific Eval, potentially
    using specific hardware. The hardware is not applicable for services like OpenAI's API, but
    would be applicable for running locally or against specific/configurable hardware like Hugging
    Face Endpoints or a custom server. The quality of responses might not change between hardware,
    but the speed of responses could.
    """

    def __init__(
            self,
            eval_obj: 'Eval',
            candidate_obj: Candidate,
            responses: list[str],
            total_time_seconds: float,
            results: list[list[CheckResult]],
            ) -> None:
        self.eval_obj = eval_obj
        self.candidate_obj = candidate_obj
        self.responses = responses
        self.total_time_seconds = total_time_seconds
        self.response_characters = sum([len(r) for r in responses])
        total_time_seconds += 1e-6  # prevent divide by zero
        self.characters_per_second = sum([len(r) for r in responses]) / total_time_seconds
        self.results = results
        self.num_checks = sum([len(r) for r in results])
        self.num_pass_fail_checks = sum([len(r) for r in results if isinstance(r, PassFailResult)])
        if self.num_pass_fail_checks == 0:
            self.num_passing_checks = None
            self.perc_passed_checks = None
        else:
            self.num_passing_checks = sum(r.success for r in self.results if isinstance(r, PassFailResult))  # noqa
            self.perc_passed_checks = self.num_passing_checks / self.num_pass_fail_checks


    def all_results(self) -> list[CheckResult]:
        """Returns a list of all CheckResults."""
        return [r for result in self.results for r in result]

    def __str__(self) -> str:
        """Returns a string representation of the EvalResult."""
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


def summarizer(result: EvalResult) -> dict:
    """Simple summarizer that returns a dictionary of summary statistics."""
    code_run_checks = [
        r for r in result.all_results()
        if r.metadata['type'] == CheckType.PYTHON_CODE_BLOCKS_RUN.name
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
