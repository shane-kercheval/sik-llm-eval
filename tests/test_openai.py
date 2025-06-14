"""Tests OpenAI classes."""
import pytest
from pytest_mock import MockerFixture
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from sik_llm_eval.openai import (
    AsyncOpenAICompletion,
    OpenAICompletion,
    OpenAICompletionResponse,
    OpenAIResponse,
    Function,
    OpenAITools,
    OpenAIToolsResponse,
    num_tokens,
    system_message,
    user_message,
)
from tests.conftest import OPENAI_DEFAULT_MODEL


@pytest.fixture
def mock_openai_client(mocker: MockerFixture):
    # Create base mock client
    mock_client = mocker.Mock()
    # Create the nested structure for chat.completions.create
    mock_client.chat = mocker.Mock()
    mock_client.chat.completions = mocker.Mock()

    def create_side_effect(**kwargs):  # noqa: ANN003, ANN202
        mock_response = mocker.Mock()
        mock_response.model = "mock"
        mock_response.created = 1234567890
        mock_response.object = "chat.completion"
        mock_response.usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        # Store all input parameters in content
        mock_choice = mocker.Mock()
        mock_choice.message.content = str(kwargs)
        mock_choice.message.role = "assistant"
        mock_choice.message.tool_calls = None
        mock_choice.logprobs = None
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.completions.create = mocker.Mock(side_effect=create_side_effect)
    return mock_client


def test__OpenAIResponse__dict_metadata_model_dump():
    response = OpenAIResponse(
        object_name='test name',
        model='test model',
        created=123456,
        metadata={'test_1': 'test', 'test_2': 1},
        finish_reason='length',
    )
    assert response.model_dump() == {
        'object_name': 'test name',
        'model': 'test model',
        'created': 123456,
        'metadata': {'test_1': 'test', 'test_2': 1},
        'finish_reason': 'length',
    }

def test__OpenAIResponse__BaseModel_metadata_model_dump():
    class Metadata(BaseModel):
        test_1: str
        test_2: int

    response = OpenAIResponse(
        object_name='test name',
        model='test model',
        created=123456,
        metadata=Metadata(test_1='test', test_2=1),
    )
    assert response.model_dump() == {
        'object_name': 'test name',
        'model': 'test model',
        'created': 123456,
        'metadata': {'test_1': 'test', 'test_2': 1},
        'finish_reason': None,
    }

def test__OpenAICompletionWrapper() -> None:
    client = OpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]

    model = OpenAICompletion(
        client=client,
        model=OPENAI_DEFAULT_MODEL,
        temperature=0.1,
    )
    response = model(messages=messages)
    assert expected_response in response.content.lower()
    assert response.model is not None
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert 'logprobs' not in response
    assert 'logprobs_tokens' not in response

    # max tokens == 1
    response = model(messages=messages, max_tokens=1)
    assert len(response.content.split()) == 1
    assert response.model is not None
    assert response.role == 'assistant'
    assert response.finish_reason == 'length'  # stopped due to max_tokens
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] == 1
    assert response.usage['total_tokens'] > 0
    assert 'logprobs' not in response
    assert 'logprobs_tokens' not in response

    # logprobs
    response = model(messages=messages, logprobs=True)
    assert expected_response in response.content.lower()
    assert response.model is not None
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.logprobs is not None
    assert len(response.logprobs) > 0
    assert len(response.logprobs) == len(response.logprobs_tokens)

def test__OpenAICompletionWrapper__streaming() -> None:
    # test valid parameters for streaming
    callback_chunks = []
    def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = OpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    model = OpenAICompletion(
        client=client,
        model=OPENAI_DEFAULT_MODEL,
        stream_callback=streaming_callback,
    )
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]
    response = model(messages=messages)
    assert expected_response in response.content.lower()
    assert len(callback_chunks) >= 3  # 2 chunks + 1 empty chunk w/ finish_reason
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'stop'
    assert callback_chunks[-1].finish_reason == 'stop'

    ####
    # logprobs is still tested in case it is set even though logprobs is not returned in streaming
    # response by openai
    ####
    callback_chunks = []
    def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    response = model(messages=messages, logprobs=True)
    assert expected_response in response.content.lower()
    assert len(callback_chunks) >= 3  # 2 chunks + 1 empty chunk w/ finish_reason
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'stop'
    assert callback_chunks[-1].finish_reason == 'stop'

    # max_tokens == 1; only 1 chunk should be returned
    callback_chunks = []
    def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    response = model(messages=messages, max_tokens=1)
    assert len(response.content.split()) == 1
    assert len(callback_chunks) >= 2  # 1 chunk + 1 empty chunk w/ finish_reason
    assert callback_chunks[-1].finish_reason is not None
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'length'
    assert callback_chunks[-1].finish_reason == 'length'

def test__OpenAICompletionWrapper__model_kwargs(mock_openai_client) -> None:  # noqa: ANN001
    """Ensures that the correct model parameters are passed to the OpenAI API."""
    # test passing in model_kwargs to the OpenAICompletion object
    completion = OpenAICompletion(client=mock_openai_client, model='mock', temperature=0.5, max_tokens=10)  # noqa: E501
    response = completion(messages=[{'role': 'user', 'content': 'test'}])
    input_received = eval(response.content)
    assert input_received['messages'] == [{'role': 'user', 'content': 'test'}]
    assert input_received['temperature'] == 0.5
    assert input_received['max_tokens'] == 10

    # test passing in model_kwargs to the __call__ method
    completion = OpenAICompletion(client=mock_openai_client, model='mock')
    response = completion(messages=[{'role': 'user', 'content': 'test'}], temperature=0.5, max_tokens=10)  # noqa: E501
    input_received = eval(response.content)
    assert input_received['messages'] == [{'role': 'user', 'content': 'test'}]
    assert input_received['temperature'] == 0.5
    assert input_received['max_tokens'] == 10

    # test merging parameters by passing in model_kwargs to both the OpenAICompletion object and
    # the __call__ method
    completion = OpenAICompletion(client=mock_openai_client, model='mock', temperature=0.5)
    response = completion(messages=[{'role': 'user', 'content': 'test'}], max_tokens=10)
    input_received = eval(response.content)
    assert input_received['messages'] == [{'role': 'user', 'content': 'test'}]
    assert input_received['temperature'] == 0.5
    assert input_received['max_tokens'] == 10

    # test overriding parameters by passing in model_kwargs to both the OpenAICompletion object and
    # the __call__ method (__call__ method should take precedence)
    completion = OpenAICompletion(client=mock_openai_client, model='mock', temperature=0.5)
    response = completion(messages=[{'role': 'user', 'content': 'test'}], max_tokens=10, temperature=0.1)  # noqa: E501
    input_received = eval(response.content)
    assert input_received['messages'] == [{'role': 'user', 'content': 'test'}]
    assert input_received['temperature'] == 0.1
    assert input_received['max_tokens'] == 10

@pytest.mark.asyncio
async def test__AsyncOpenAICompletionWrapper() -> None:
    client = AsyncOpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]

    model = AsyncOpenAICompletion(
        client=client,
        model=OPENAI_DEFAULT_MODEL,
        temperature=0.1,
    )
    response = await model(messages=messages)
    assert expected_response in response.content.lower()
    assert response.model is not None
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert 'logprobs' not in response
    assert 'logprobs_tokens' not in response

    # max tokens == 1
    response = await model(messages=messages, max_tokens=1)
    assert len(response.content.split()) == 1
    assert response.model is not None
    assert response.role == 'assistant'
    assert response.finish_reason == 'length'  # stopped due to max_tokens
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] == 1
    assert response.usage['total_tokens'] > 0
    assert 'logprobs' not in response
    assert 'logprobs_tokens' not in response

    # logprobs
    response = await model(messages=messages, logprobs=True)
    assert expected_response in response.content.lower()
    assert response.model is not None
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.logprobs is not None
    assert len(response.logprobs) > 0
    assert len(response.logprobs) == len(response.logprobs_tokens)

@pytest.mark.asyncio
async def test__AsyncOpenAICompletionWrapper__streaming() -> None:
    # test valid parameters for streaming
    callback_chunks = []
    async def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = AsyncOpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    model = AsyncOpenAICompletion(
        client=client,
        model=OPENAI_DEFAULT_MODEL,
        stream_callback=streaming_callback,
    )
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]
    response = await model(messages=messages)
    assert expected_response in response.content.lower()
    assert len(callback_chunks) >= 3  # 2 chunks + 1 empty chunk w/ finish_reason
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'stop'
    assert callback_chunks[-1].finish_reason == 'stop'

    ####
    # logprobs is still tested in case it is set even though logprobs is not returned in streaming
    # response by openai
    ####
    callback_chunks = []
    async def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    response = await model(messages=messages, logprobs=True)
    assert expected_response in response.content.lower()
    assert len(callback_chunks) >= 3  # 2 chunks + 1 empty chunk w/ finish_reason
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'stop'
    assert callback_chunks[-1].finish_reason == 'stop'

    # max_tokens == 1; only 1 chunk should be returned
    callback_chunks = []
    async def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    response = await model(messages=messages, max_tokens=1)
    assert len(response.content.split()) == 1
    assert len(callback_chunks) >= 2  # 1 chunk + 1 empty chunk w/ finish_reason
    assert callback_chunks[-1].finish_reason is not None
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'length'
    assert callback_chunks[-1].finish_reason == 'length'

def test__num_tokens():
    assert num_tokens(model='gpt-3.5-turbo-0613', value="This should be six tokens.") == 6

def test__Function__get_current_weather__to_dict(weather_tool: Function):
    assert weather_tool.to_dict() == {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }

def test__Function__get_current_stocks__to_dict(stocks_tool: Function):
    assert stocks_tool.to_dict() == {
        "type": "function",
        "function": {
            "name": "get_current_stocks",
            "description": "Get the current stock price of a given company",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "The name of the company, e.g. Apple",
                    },
                },
                "required": ["company"],
            },
        },
    }

def test__OpenAITools(weather_tool: Function, stocks_tool: Function):
    tools = [
        weather_tool.to_dict(),
        stocks_tool.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model=OPENAI_DEFAULT_MODEL)
    ####
    # first interaction
    ####
    messages = [{"role": "user", "content": "What's the weather like in Boston today in degrees F?"}]  # noqa: E501
    response = model(
        messages=messages,
        tools=tools,
    )
    assert isinstance(response, OpenAIToolsResponse)
    assert OPENAI_DEFAULT_MODEL in response.model
    assert response.role == 'assistant'
    assert response.finish_reason == 'tool_calls'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert len(response.tools) == 1
    assert response.tools[0]['type'] == 'function'
    assert response.tools[0]['name'] == 'get_current_weather'
    assert response.tools[0]['arguments'] == {'location': 'Boston, MA', 'unit': 'fahrenheit'}

def test__OpenAITools__unrelated_prompt__auto(weather_tool: Function, stocks_tool: Function):
    """
    When the prompt is unrelated to any tool and the tool_choice is 'auto', then we will get
    a OpenAICompletionResponse object rather than a OpenAIToolResponse object.
    """
    tools = [
        weather_tool.to_dict(),
        stocks_tool.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model=OPENAI_DEFAULT_MODEL)
    response = model(
        messages= [{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="auto",
    )
    assert isinstance(response, OpenAICompletionResponse)
    assert OPENAI_DEFAULT_MODEL in response.model
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.content

def test__OpenAITools__unrelated_prompt__required(weather_tool: Function, stocks_tool: Function):
    """
    When the prompt is unrelated to any tool and the tool_choice is 'required', then we will get
    a OpenAIToolResponse object, because `required` means that a tool must be returned.
    """
    tools = [
        weather_tool.to_dict(),
        stocks_tool.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model=OPENAI_DEFAULT_MODEL)
    response = model(
        messages= [{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="required",
    )
    assert isinstance(response, OpenAIToolsResponse)

def test__OpenAITools__unrelated_prompt__none(weather_tool: Function, stocks_tool: Function):
    tools = [
        weather_tool.to_dict(),
        stocks_tool.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model=OPENAI_DEFAULT_MODEL)
    response = model(
        messages= [{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="none",
    )
    assert isinstance(response, OpenAICompletionResponse)
    assert OPENAI_DEFAULT_MODEL in response.model
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.content

def test__OpenAITools__structured_response():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    messages=[
        system_message("Extract the event information."),
        user_message("Alice and Bob went to a science fair on 1984-01-30."),
    ]
    model = OpenAICompletion(
        client= OpenAI(),
        model=OPENAI_DEFAULT_MODEL,
        response_format=CalendarEvent,
        temperature=0.1,
    )
    response = model(messages=messages)
    assert isinstance(response.parsed, CalendarEvent)
    assert response.parsed.name.lower() == "science fair"
    assert response.parsed.date == "1984-01-30"
    assert set(response.parsed.participants) == {"Alice", "Bob"}
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['input_tokens'] > 0
    assert response.usage['output_tokens'] > 0
    assert response.usage['total_tokens'] > 0
