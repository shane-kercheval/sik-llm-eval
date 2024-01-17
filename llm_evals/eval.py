"""TODO document."""
import time
from typing import Callable
from pydantic import BaseModel
from llm_evals.checks import CheckRegistery, EvalCheck, CheckType, CHECK_REGISTRY


class Prompt(BaseModel):
    """TODO document."""

    prompt: str
    ideal_response: str | None = None


class EvalResult(BaseModel):
    """
    An EvalResult is the result of evaluating a specific LLM against a specific Eval, potentially
    using specific hardware. The hardware is not applicable for services like OpenAI's API, but
    would be applicable for running locally or against specific/configurable hardware like Hugging
    Face Endpoints or a custom server. The quality of responses might not change between hardware,
    but the speed of responses could.
    """

    llm_id: str
    eval_id: str
    system: dict
    responses: list[str]
    total_time: float
    response_characters: int
    characters_per_second: float
    # Code blocks are a particular type of check, not sure i like that here
    num_code_blocks: int
    code_blocks_passed: int
    check_results: list[object]



class LLMModel(BaseModel):
    """TODO document."""
    name: str
    model: callable
    description: str | None = None
    metadata: dict | None = None   # ??? need hardwhere, where to specify? 


# need to Register the different types of Tests

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
            uuid: str,
            metadata: dict,
            prompts: list[Prompt],
            checks: list[EvalCheck],
            ):
        self.uuid = uuid
        self.metadata = metadata
        self.prompts = prompts
        self.checks = checks
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
        assert 'uuid' in config, "uuid is a required field when creating an Eval object"
        prompts = [Prompt(**prompt) for prompt in config['prompts']]
        # need to register the different types of tests
        # tests = [EvalTest(**test) for test in config['tests']]
        registry = registry or CHECK_REGISTRY
        checks = []
        for test in config['checks']:
            test['eval_uuid'] = config['uuid']
            checks.append(registry.create_check(
                check_type=CheckType.to_enum(test.pop('type')),
                params=test,
            ))
        obj = cls(
            uuid=config['uuid'],
            metadata=config['metadata'] if 'metadata' in config else {},
            prompts=prompts,
            checks=checks,
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


    def __call__(
            self,
            llm_id: str,
            llm: Callable[[str], str]) -> EvalResult:
        """Evaluates the model against the prompts and tests."""
        start = time.time()
        responses = [llm(p.prompt) for p in self.prompts]
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
        from textwrap import dedent
        prompts = ',\n                '.join([str(p) for p in self.prompts])
        metadata = '' if not self.metadata else f'\n            metadata={self.metadata},'
        return dedent(f"""
        Eval(
            uuid={self.uuid},{metadata}
            prompts=[
                {prompts}
            ],
            checks=[{', '.join([str(type(t)) for t in self.checks])}]
        )
        """).strip()
