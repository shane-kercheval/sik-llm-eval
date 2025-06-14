"""Configures the pytests."""
from copy import deepcopy
import os
import re
from time import sleep
import pytest
import requests
from sik_llms import Parameter, Tool
import yaml
from faker import Faker
from unittest.mock import MagicMock
from sik_llm_eval.candidates import Candidate, CandidateResponse

from dotenv import load_dotenv

from sik_llm_eval.checks import Check, CheckResult, PassFailResult
# from sik_llm_eval.openai import Function, FunctionParameter
load_dotenv()


OPENAI_DEFAULT_MODEL = 'gpt-4o-mini'


@pytest.fixture
def openai_model() -> str:
    """Returns the name of the OpenAI model."""
    return 'gpt-4o-mini'


@pytest.fixture
def bedrock_model() -> str:
    """Returns the name of the OpenAI model."""
    return 'anthropic.claude-3-haiku-20240307-v1:0'


@Candidate.register('MockCandidate')
class MockCandidate(Candidate):
    """
    This class needs to be outside of the test function so that we can test multi-processing, which
    requires that the class be picklable, which requires that it be defined at the top level of the
    module.

    This candidate takes a dictionary of prompts (keys) and responses (values) and returns the
    response for the given prompt.
    """

    def __init__(
            self,
            responses: dict,
            metadata: dict | None = None,
            parameters: dict | None = None):
        super().__init__(metadata=metadata, parameters=parameters)
        self.responses = responses.copy()

    def __call__(self, input: str) -> str:  # noqa: A002
        """Returns the response for the given prompt."""
        response = self.responses[input[0]['content']]
        if isinstance(response, Exception):
            raise response
        return CandidateResponse(response=response)

    def to_dict(self) -> dict:
        """Need to add `responses` to enable proper to_dict values."""
        value = super().to_dict()
        value['responses'] = self.responses
        return value


@Candidate.register('AsyncMockCandidate')
class AsyncMockCandidate(Candidate):
    """
    This class needs to be outside of the test function so that we can test multi-processing, which
    requires that the class be picklable, which requires that it be defined at the top level of the
    module.

    This candidate takes a dictionary of prompts (keys) and responses (values) and returns the
    response for the given prompt.
    """

    def __init__(
            self,
            responses: dict,
            metadata: dict | None = None,
            parameters: dict | None = None):
        super().__init__(metadata=metadata, parameters=parameters)
        self.responses = responses.copy()

    async def _get_response(self, input) -> str:  # noqa
        """Returns the response for the given prompt."""
        response = self.responses[input[0]['content']]
        if isinstance(response, Exception):
            raise response
        return CandidateResponse(response=response)

    async def __call__(self, input) -> str:  # noqa
        """Returns the response for the given prompt."""
        return await self._get_response(input)


    def to_dict(self) -> dict:
        """Need to add `responses` to enable proper to_dict values."""
        value = super().to_dict()
        value['responses'] = self.responses
        return value


class MockCandidateCausesError(Candidate):  # noqa: D101
    def __call__(self, input: object) -> CandidateResponse:  # noqa
        raise ValueError("This candidate always fails.")


class AsyncMockCandidateCausesError(Candidate):  # noqa: D101
    async def __call__(self, input: object) -> CandidateResponse:  # noqa
        raise ValueError("This candidate always fails.")


@Candidate.register('MockCandidateCannedResponse')
class MockCandidateCannedResponse(Candidate):  # noqa
    def __init__(
            self,
            metadata: dict | None = None,
            parameters: dict | None = None):
        super().__init__(metadata=metadata, parameters=parameters)

    def __call__(self, _: str) -> str:
        """Returns the response for the given prompt."""
        sleep(0.01)
        return 'Response'

    def set_message_history(self, messages: list[dict] | list[tuple]) -> None:  # noqa
        return

    def set_system_message(self, system_message: str) -> None:  # noqa
        return

    def to_dict(self) -> dict:
        """Need to add `responses` to enable proper to_dict values."""
        return super().to_dict()

    def clone(self) -> 'Candidate':
        """
        Returns a copy of the Candidate with the same state but with a different instance of the
        underlying model (e.g. same parameters but reset history/context).

        Reques
        """
        return Candidate.from_dict(deepcopy(self.to_dict()))


class MockRetryTestCandidate(Candidate):
    """Mock candidate that fails first `fail_until` attempts, then succeeds."""

    def __init__(
            self,
            fail_until_attempt: int=2,
            error_message: str = "Simulated failure",
            response: str = "success",
            metadata: dict | None=None,
            parameters: dict | None=None,
        ):
        super().__init__(metadata=metadata, parameters=parameters)
        self.fail_until = fail_until_attempt
        self.error_message = error_message
        self.response = response
        self.metadata['attempts'] = 0

    async def __call__(self, input_):  # noqa: ANN001, ARG002
        self.metadata['attempts'] += 1
        # <DO WORK HERE>
        if self.metadata['attempts'] <= self.fail_until:  # Fail first two attempts
            raise ValueError(self.error_message)
        return CandidateResponse(
            response=self.response,
            metadata=self.metadata,
        )


class MockCheckCausesError(Check):  # noqa: D101
    def __call__(self, response: object) -> CheckResult:  # noqa
        raise RuntimeError("This check always fails.")

class MockRetryTestCheck(Check):
    """Mock check that fails first `fail_until_attempt` attempts, then succeeds."""

    fail_until_attempt: int=2
    error_message: str = "Simulated failure"
    attempts: int = 0

    def __call__(self, response: object) -> CheckResult:  # noqa: ARG002
        self.attempts += 1
        if self.attempts <= self.fail_until_attempt:
            raise ValueError(self.error_message)
        return PassFailResult(value=True, metadata={'attempts': self.attempts})


class FakeRetryHandler:
    """A fake retry handler used for unit tests."""

    def __call__(self, f, *args, **kwargs):  # noqa
        return f(*args, **kwargs)


@pytest.fixture
def fake_retry_handler():
    return FakeRetryHandler()


@pytest.fixture
def fake_hugging_face_response_json():
    fake = Faker()
    num_words = fake.random_int(min=8, max=10)
    if num_words == 8:
        return [{
            'generated_text': "",
        }]

    return [{
        'generated_text': " ".join(fake.words(nb=num_words)),
    }]


@pytest.fixture
def fake_hugging_face_response(fake_hugging_face_response_json):  # noqa
    response = MagicMock()
    response.json.return_value = fake_hugging_face_response_json
    return response


def is_endpoint_available(url: str) -> bool:
    """Returns True if the endpoint is available."""
    available = False
    try:
        response = requests.head(url, timeout=5)  # You can use GET or HEAD method
        # checking if we get a response code of 2xx or 3xx
        available = 200 <= response.status_code <= 401
    except requests.RequestException:
        pass

    if not available:
        print('Endpoint Not available.')

    return available


@pytest.fixture
def hugging_face_endpoint() -> str:
    """Returns the endpoint for the hugging face API."""
    return os.getenv('HUGGING_FACE_ENDPOINT_UNIT_TESTS')


def pattern_found(value: str, pattern: str) -> bool:
    """Returns True if the pattern is found in the value."""
    pattern = re.compile(pattern)
    return bool(pattern.match(value))


@pytest.fixture
def conversation_mask_email() -> dict:
    """Returns a mock llm  conversation for masking emails."""
    with open('tests/fake_data/fake_conversation__mask_email_function.yml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def conversation_sum() -> dict:
    """Returns a mock llm conversation for summing numbers."""
    with open('tests/fake_data/fake_conversation__sum_function.yml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_eval_8f9fbf37() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_eval_8F9FBF37.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_eval_subtract_two_numbers() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_eval_subtract_two_numbers.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_eval_sum_two_numbers() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_eval_sum_two_numbers.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_eval_sum_two_numbers_code_blocks_run() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_eval_sum_two_numbers_code_blocks_run.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_eval_no_code_blocks() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_eval_no_code_blocks.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_eval_with_previous_messages() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_eval_with_previous_messages.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_eval_non_string_values() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_eval_non_string_values.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_multi_eval() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_multi_eval.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_multi_eval_non_string_values() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_multi_eval_non_string_values.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def fake_multi_eval_with_prompt_sequence() -> dict:
    """Returns a fake eval."""
    with open('tests/fake_data/fake_multi_eval_with_prompt_sequence.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def openai_candidate_template() -> dict:
    """Returns the yaml template for an OpenAI."""
    with open('examples/candidates/openai_4o-mini.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def openai_tools_candidate_template() -> dict:
    """Returns the yaml template for an OpenAI Tools."""
    with open('examples/candidates/openai_tools_4o-mini.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def hugging_face_candidate_template() -> dict:
    """Returns the yaml template for a Hugging Face Endpoint candidate."""
    with open('examples/candidates/additional_examples/hugging_face_endpoint_mistral_a10g.yaml') as f:  # noqa
        config = yaml.safe_load(f)
    config['parameters']['endpoint_url'] = os.getenv('HUGGING_FACE_ENDPOINT_UNIT_TESTS')
    return config


@pytest.fixture
def weather_tool() -> Tool:
    """Returns a dictionary defining a weather function."""
    return Tool(
        name="get_current_weather",
        description="Get the current weather in a given location",
        parameters=[
            Parameter(
                name="location",
                param_type=str,
                description="The city and state, e.g. San Francisco, CA",
                required=True,
            ),
            Parameter(
                name="unit",
                param_type=str,
                valid_values=["celsius", "fahrenheit"],  # Using enum to constrain possible values
                required=True,
                # description="The unit of temperature",
            ),
        ],
    )


@pytest.fixture
def stocks_tool() -> Tool:
    """Returns a dictionary defining a stock function."""
    return Tool(
        name="get_current_stocks",
        description="Get the current stock price of a given company",
        parameters=[
            Parameter(
                name="company",
                param_type=str,
                description="The name of the company, e.g. Apple",
                required=True,
            ),
        ],
    )


class UnregisteredCheckResult(CheckResult):  # noqa
    pass

class UnregisteredCheck(Check):  # noqa
    def __call__(self, value: str) -> UnregisteredCheckResult:
        return UnregisteredCheckResult(
            success=value is not None,
            value=value,
            metadata={},
        )

    def clone(self) -> Check:
        return UnregisteredCheck()

class UnregisteredCandidate(Candidate):  # noqa
    def __init__(
            self,
            response: object,
            metadata: dict | None = None,
            parameters: dict | None = None,
        ) -> None:
        super().__init__(metadata=metadata, parameters=parameters)
        self.response = response

    def __call__(self, prompt: dict) -> dict:
        # returns dictionary instead of string
        return CandidateResponse(response={'prompt': prompt, 'response': self.response})
