"""TODO document."""
import time
from typing import Callable
from pydantic import BaseModel
from llm_evals.tests import EvalTest, TestType, TEST_REGISTRY


class Prompt(BaseModel):
    """TODO document."""

    prompt: str
    ideal_response: str | None = None


class EvalResult(BaseModel):
    """
    An EvalResult is the result of evaluating a specific LLM against a specific Eval, potentially
    using specific hardware. The hardware is not applicable for services like OpenAI's API, but
    would be applicable for running locally or against specific/configurable hardware like Hugging
    Face Endpoints or a custom server.
    """

    llm_id: str
    eval_id: str
    system: dict
    # potential duplication of information, but i think we need it on this object
    responses: list[str]
    total_time: float
    response_characters: int
    characters_per_second: float
    # this depends on a particular type of test, not sure i like that
    num_code_blocks: int
    code_blocks_passed: int
    test_results: list[object]

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
            tests: list[EvalTest],
            ):
        self.uuid = uuid
        self.metadata = metadata
        self.prompts = prompts
        self.tests = tests
        self.results = None

    @classmethod
    def from_dict(cls, config: dict, results: dict | None = None) -> 'Eval':  # noqa: ANN102
        """Creates an Eval object from a config/dictionary."""
        assert 'uuid' in config, "uuid is a required field when creating an Eval object"
        prompts = [Prompt(**prompt) for prompt in config['prompts']]
        # need to register the different types of tests
        # tests = [EvalTest(**test) for test in config['tests']]
        tests = []
        for test in config['tests']:
            test['eval_uuid'] = config['uuid']
            tests.append(TEST_REGISTRY.create_test(
                test_type=TestType.to_enum(test.pop('type')),
                params=test,
            ))
        # tests = [
        #     TEST_REGISTRY.create_test(test_type=TestType.to_enum(t.pop('type')), params=t)
        #     for t in config['tests']
        # ]
        obj = cls(
            uuid=config['uuid'],
            metadata=config['metadata'] if 'metadata' in config else {},
            prompts=prompts,
            tests=tests,
        )
        if results is not None:
            obj.results = EvalResult(**results)
        return obj


    def __call__(self, llm_id: str, llm: Callable[[str], str]) -> dict:
        """Evaluates the model against the prompts and tests."""
        start = time.time()
        responses = [llm(p.prompt) for p in self.prompts]
        end = time.time()
        self._duration = end - start

        # TODO
        results = [test(responses) for test in self.tests]

        self.results = EvalResult(
            llm_id=llm_id,
            eval_id=self.uuid,
            system=self.metadata,
            responses=responses,
            total_time=self._duration,
            response_characters=sum([len(r) for r in responses]),
            characters_per_second=sum([len(r) for r in responses]) / self._duration,
            num_code_blocks=0,
            code_blocks_passed=0,
            test_results=results,
        )

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
            tests=[{', '.join([str(type(t)) for t in self.tests])}]
        )
        """).strip()
