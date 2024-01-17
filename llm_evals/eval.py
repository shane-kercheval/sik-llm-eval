"""TODO document."""
from textwrap import dedent, indent
import time
from typing import Callable
from pydantic import BaseModel
from llm_evals.checks import CheckRegistery, Check, CheckType, CHECK_REGISTRY


class Scenario(BaseModel):
    """
    TODO document.

    Checks can be optional because even a scenario without checks still collects responses (that
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
        """Returns a string representation of the Scenario."""
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
        Scenario(
            prompt='{self.prompt}',{ideal_response}
            checks={checks},
        )
        """).strip()



class Candidate(BaseModel):
    """
    A Candidate describes an LLM that may optionally be associated with specific parameters or
    hardware.
    """

    name: str
    model: Callable[[str], str]
    description: str | None = None
    uuid: str | None = None
    parameters: dict | None = None
    hardware: dict | None = None


class EvalResult(BaseModel):
    """
    An EvalResult is the result of evaluating a specific LLM against a specific Eval, potentially
    using specific hardware. The hardware is not applicable for services like OpenAI's API, but
    would be applicable for running locally or against specific/configurable hardware like Hugging
    Face Endpoints or a custom server. The quality of responses might not change between hardware,
    but the speed of responses could.
    """

    eval_obj: 'Eval'
    candidate_obj: Candidate
    responses: list[str]
    total_time: float
    response_characters: int
    characters_per_second: float
    num_code_blocks: int
    check_results: list[object]


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
            metadata: dict,
            scenarios: list[Scenario],
            uuid: str | None = None,
            ):
        """
        Args:
            uuid: optional. Used to uniquely identify the Eval which is ultimately used to avoid
                running the same Eval (against the same Candidate/llm) more than once.
        """
        self.metadata = metadata
        self.scenarios = scenarios
        self.uuid = uuid
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
        # prompts = [Scenario(**prompt) for prompt in config['scenarios']]
        # need to register the different types of tests
        # tests = [EvalTest(**test) for test in config['tests']]
        registry = registry or CHECK_REGISTRY

        def create_checks(checks_data: list[dict]) -> list[Check]:
            checks = []
            for check_data in checks_data:
                check_type = CheckType.to_enum(check_data.pop('type'))
                check = registry.create_check(check_type=check_type, params=check_data)
                checks.append(check)
            return checks

        scenarios = []
        for scenario in config['scenarios']:
            checks = create_checks(scenario.pop('checks')) if 'checks' in scenario else None
            scenarios.append(Scenario(**scenario, checks=checks))

        obj = cls(
            uuid=config['uuid'],
            metadata=config['metadata'] if 'metadata' in config else {},
            scenarios=scenarios,
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


    def __call__(self, candidate: Candidate) -> EvalResult:
        """Evaluates the model against the prompts and tests."""
        start = time.time()
        responses = [candidate.model(p.prompt) for p in self.scenarios]
        end = time.time()
        self._duration = end - start
        # TODO: can't actually do this because, for example, we can't run CODE_BLOCK checks until
        # extract/execute code blocks in responses and we have to run `setup` from eval in same
        # environment, and we need to pass the code blocks, not the
        # responses
        results = [check(responses) for check in self.checks]
        code_block_results = [r for r in results if r.type == 'code_block']
        self.result = EvalResult(
            llm_id=llm_id,
            eval_id=self.uuid,
            system=self.metadata,  # TODO: finalize this; can't just pull from unstructured dict
            responses=responses,
            total_time=self._duration,
            response_characters=sum([len(r) for r in responses]),
            characters_per_second=sum([len(r) for r in responses]) / self._duration,
            # Code blocks are a particular type of check, not sure i like that here
            # num_code_blocks=0,
            # code_blocks_passed=0,
            check_results=results,
        )
        return self.result

    def __str__(self) -> str:
        """Returns a string representation of the Eval."""
        if self.scenarios:
            scenarios = '[\n'
            indent_value = ' ' * 16
            scenarios += ',\n'.join([indent(str(s), indent_value) for s in self.scenarios])
            scenarios += '\n            ]'
        else:
            scenarios = '[]'
        metadata = '' if not self.metadata else f'\n            metadata={self.metadata},'
        return dedent(f"""
        Eval(
            uuid={self.uuid},{metadata}
            scenarios={scenarios},
        )
        """).strip()
