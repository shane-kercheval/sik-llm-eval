"""Configures the pytests."""
from copy import deepcopy
import os
import re
from time import sleep
from pydantic import BaseModel
import pytest
import requests
import yaml
from faker import Faker
from unittest.mock import MagicMock
from llm_eval.candidates import Candidate, CandidateResponse

from dotenv import load_dotenv

from llm_eval.openai import Function, FunctionParameter
load_dotenv()


@pytest.fixture
def openai_model() -> str:
    """Returns the name of the OpenAI model."""
    return "gpt-4o-mini"


@Candidate.register('MockCandidate')
class MockCandidate(Candidate):
    """
    This class needs to be outside of the test function so that we can test multi-processing, which
    requires that the class be picklable, which requires that it be defined at the top level of the
    module.

    This candidate takes a dictionary of prompts (keys) and responses (values) and returns the
    response for the given prompt.
    """  # noqa: D404

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
    """  # noqa: D404

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


# @pytest.fixture
# def fake_docs_abcd() -> list[Document]:
#     """Meant to be used MockABCDEmbeddings model."""
#     return [
#         Document(content="Doc A", metadata={'id': 0}),
#         Document(content="Doc B", metadata={'id': 1}),
#         Document(content="Doc C", metadata={'id': 3}),
#         Document(content="Doc D", metadata={'id': 4}),
#     ]


# class MockABCDEmbeddings(EmbeddingModel):
#     """
#     Used for unit tests to mock the behavior of an LLM.

#     Used in conjunction with a specific document list `fake_docs_abcd`.
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.cost_per_token = 7
#         self._next_lookup_index = None
#         self.lookup = {
#             0: [0.5, 0.5, 0.5, 0.5, 0.5],
#             1: [1, 1, 1, 1, 1],
#             3: [3, 3, 3, 3, 3],
#             4: [4, 4, 4, 4, 4],
#         }

#     def _run(self, docs: list[Document]) -> tuple[list[Document], EmbeddingRecord]:
#         if self._next_lookup_index:
#             embeddings = [self.lookup[self._next_lookup_index]]
#         else:
#             embeddings = [self.lookup[x.metadata['id']] for x in docs]
#         total_tokens = sum(len(x.content) for x in docs)
#         cost = total_tokens * self.cost_per_token
#         return embeddings, EmbeddingRecord(
#             total_tokens=total_tokens,
#             cost=cost,
#             metadata={'content': [x.content for x in docs]},
#         )


class FakeRetryHandler:
    """A fake retry handler used for unit tests."""

    def __call__(self, f, *args, **kwargs):  # noqa
        return f(*args, **kwargs)


@pytest.fixture
def fake_retry_handler():  # noqa
    return FakeRetryHandler()


@pytest.fixture
def fake_hugging_face_response_json():  # noqa
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
def function_weather() -> Function:
    """Returns a dictionary defining a weather function."""
    return Function(
        name="get_current_weather",
        description="Get the current weather in a given location",
        parameters=[
            FunctionParameter(
                name="location",
                type="string",
                description="The city and state, e.g. San Francisco, CA",
                required=True,
            ),
            FunctionParameter(
                name="unit",
                type="string",
                valid_values=["celsius", "fahrenheit"],  # Using enum to constrain possible values
                # description="The unit of temperature",
            ),
        ],
    )


@pytest.fixture
def function_stocks() -> Function:
    """Returns a dictionary defining a stock function."""
    return Function(
        name="get_current_stocks",
        description="Get the current stock price of a given company",
        parameters=[
            FunctionParameter(
                name="company",
                type="string",
                description="The name of the company, e.g. Apple",
                required=True,
            ),
        ],
    )



###################################################################################################
# Mock OpenAI API Client
# Designed to mimic the response structure of the OpenAI API
###################################################################################################
class MockOpenAIDelta(BaseModel):  # noqa: D101
    content: str

class MockOpenAIChoiceChunk(BaseModel):  # noqa: D101
    delta: MockOpenAIDelta
    finish_reason: str

class MockOpenAIChoiceMessage(BaseModel):  # noqa: D101
    content: str
    role: str | None = None

class MockOpenAIChoice(BaseModel):  # noqa: D101
    message: MockOpenAIChoiceMessage
    finish_reason: str
    logprobs: list[float] | None = None

class MockOpenAIUsage(BaseModel):  # noqa: D101
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class MockOpenAIChatCompletionChunkResponse(BaseModel):  # noqa: D101
    object: str
    model: str
    created: int
    choices: list[MockOpenAIChoiceChunk]

class MockOpenAIChatCompletionResponse(BaseModel):  # noqa: D101
    object: str
    model: str
    created: int
    choices: list[MockOpenAIChoice]
    usage: MockOpenAIUsage

class LegacyChoiceDelta(BaseModel):  # noqa: D101
    content: str
    function_call: str | None = None
    role: str | None = None
    tool_calls: str | None = None

class LegacyChoice(BaseModel):  # noqa: D101
    delta: LegacyChoiceDelta | None = None
    message: MockOpenAIChoiceMessage | None = None
    finish_reason: str | None = None
    index: int = 0
    logprobs: list[float] | None = None

class LegacyCompletionUsage(BaseModel):  # noqa: D101
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class LegacyChatCompletionChunkResponse(BaseModel):  # noqa: D101
    choices: list[LegacyChoice]
    created: int
    model: str
    object: str

class LegacyChatCompletionResponse(BaseModel):  # noqa: D101
    choices: list[LegacyChoice]
    created: int
    model: str
    object: str
    usage: LegacyCompletionUsage

class AsyncMockOpenAI:
    """
    Mock OpenAI API client for testing. Returns a fixed response with the same structure as the
    OpenAI API response.
    """

    def __init__(self, fake_responses: list[str], legacy: bool = False):
        self.fake_responses = fake_responses
        self.response_index = 0
        self.legacy = legacy
        self.chat = self.Chat(fake_responses=fake_responses, legacy=legacy)

    class Chat:  # noqa: D106
        def __init__(self, fake_responses: list[str], legacy: bool):
            self.completions = AsyncMockOpenAI.Chat.Completions(
                fake_responses=fake_responses,
                legacy=legacy,
            )
            self.fake_responses = fake_responses
            self.response_index = 0
            self.legacy = legacy

        class Completions:  # noqa: D106
            def __init__(self, fake_responses: list[str], legacy: bool):
                self.fake_responses = fake_responses
                self.response_index = 0
                self.legacy = legacy

            class MockAsyncIterator:  # noqa
                def __init__(self, chunks):  # noqa: ANN001
                    self.chunks = chunks
                    self.index = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.index < len(self.chunks):
                        chunk = self.chunks[self.index]
                        self.index += 1
                        return chunk
                    raise StopAsyncIteration

            async def create(self, *args, stream=False, **kwargs):  # noqa
                messages = kwargs.get('messages', [])
                assert isinstance(messages, list), "messages must be provided as a list"
                for message in messages:
                    assert isinstance(message, dict), "messages must be provided as a list of dictionaries"  # noqa: E501
                    assert 'role' in message, "role must be provided for each message"
                    assert 'content' in message, "content must be provided for each message"
                assert len(messages) > 0, "messages must be provided to the OpenAI API"

                response = self.fake_responses[self.response_index]
                self.response_index += 1
                model = kwargs.get('model')

                if stream:
                    if self.legacy:
                        chunks = [
                            LegacyChatCompletionChunkResponse(
                                choices=[
                                    LegacyChoice(
                                        finish_reason='length',
                                        delta=LegacyChoiceDelta(content=response[i:i + 4]),
                                    ),
                                ],
                                created=1234567890,
                                model='/repository',
                                object='text_completion',
                            )
                            for i in range(0, len(response), 4)
                        ]
                    else:
                        chunks = [
                            MockOpenAIChatCompletionChunkResponse(
                                object="chat.completion.chunk",
                                model=model,
                                created=1234567890,
                                choices=[
                                    MockOpenAIChoiceChunk(
                                        finish_reason='length',
                                        delta=MockOpenAIDelta(content=response[i:i + 4]),
                                    ),
                                ],
                            )
                            for i in range(0, len(response), 4)
                        ]
                    return self.MockAsyncIterator(chunks)

                if self.legacy:
                    return LegacyChatCompletionResponse(
                        choices=[
                            LegacyChoice(
                                finish_reason='length',
                                message=MockOpenAIChoiceMessage(
                                    content=response,
                                    role="assistant",
                                ),
                            )],
                        created=1234567890,
                        model='/repository',
                        object='text_completion',
                        usage=LegacyCompletionUsage(
                            completion_tokens=20,
                            prompt_tokens=10,
                            total_tokens=30,
                        ),
                    )

                return MockOpenAIChatCompletionResponse(
                    object="chat.completion",
                    model=model,
                    created=1234567890,
                    choices=[
                        MockOpenAIChoice(
                            finish_reason='length',
                            message=MockOpenAIChoiceMessage(content=response, role="assistant"),
                        )],
                    usage=MockOpenAIUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                )
