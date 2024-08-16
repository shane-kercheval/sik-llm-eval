"""Tests OpenAI classes."""
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from llm_eval.openai import (
    AsyncOpenAICompletion,
    OpenAICompletion,
    OpenAICompletionResponse,
    OpenAIResponse,
    Function,
    FunctionParameter,
    OpenAITools,
    OpenAIToolsResponse,
    num_tokens,
)
from tests.conftest import AsyncMockOpenAI


def test__OpenAIResponse__dict_metadata_model_dump():  # noqa
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

def test__OpenAIResponse__BaseModel_metadata_model_dump():  # noqa
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

def test__OpenAICompletionWrapper() -> None:  # noqa
    client = OpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]

    model = OpenAICompletion(
        client=client,
        model='gpt-4o-mini',
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
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
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
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] == 1
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
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.logprobs is not None
    assert len(response.logprobs) == 2
    assert response.logprobs_tokens is not None
    assert len(response.logprobs_tokens) == 2

def test__OpenAICompletionWrapper__streaming() -> None:  # noqa
    # test valid parameters for streaming
    callback_chunks = []
    def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = OpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    model = OpenAICompletion(
        client=client,
        model='gpt-4o-mini',
        stream_callback=streaming_callback,
    )
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]
    response = model(messages=messages)
    assert expected_response in response.content.lower()
    assert len(callback_chunks) == 3  # 2 chunks + 1 empty chunk w/ finish_reason
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
    assert len(callback_chunks) == 3  # 2 chunks + 1 empty chunk w/ finish_reason
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
    assert len(callback_chunks) == 2  # 1 chunk + 1 empty chunk w/ finish_reason
    assert callback_chunks[-1].finish_reason is not None
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'length'
    assert callback_chunks[-1].finish_reason == 'length'

@pytest.mark.asyncio()
async def test__AsyncOpenAICompletionWrapper() -> None:  # noqa
    client = AsyncOpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]

    model = AsyncOpenAICompletion(
        client=client,
        model='gpt-4o-mini',
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
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
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
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] == 1
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
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.logprobs is not None
    assert len(response.logprobs) == 2
    assert response.logprobs_tokens is not None
    assert len(response.logprobs_tokens) == 2

@pytest.mark.asyncio()
async def test__AsyncOpenAICompletionWrapper__streaming() -> None:  # noqa
    # test valid parameters for streaming
    callback_chunks = []
    async def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = AsyncOpenAI()
    assert client.api_key is not None  # via load_dotenv in conftest.py
    model = AsyncOpenAICompletion(
        client=client,
        model='gpt-4o-mini',
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
    assert len(callback_chunks) == 2  # 1 chunk + 1 empty chunk w/ finish_reason
    assert callback_chunks[-1].finish_reason is not None
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert response.role == 'assistant'
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == 'length'
    assert callback_chunks[-1].finish_reason == 'length'

@pytest.mark.asyncio()
async def test__async_MockOpenAI_object() -> None:  # noqa
    # ensure our mock object is working as expected before we use it in actual tests
    expected_response = "Hello, world!"
    expected_model = 'my-model'
    messages = [{'role': 'user', 'content': expected_response}]
    client = AsyncMockOpenAI(fake_responses=[expected_response])
    wrapper = AsyncOpenAICompletion(
        client,
        model=expected_model,
        logprobs=True,
        temperature=0.8,
    )
    ####
    # Non-streaming example
    ####
    response = await wrapper(messages=messages)
    assert expected_response == response.content
    assert response.model == expected_model
    assert response.role == 'assistant'
    assert response.created is not None
    assert response.usage is not None
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0

    # test valid parameters for streaming
    callback_chunks = []
    async def stream_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = AsyncMockOpenAI(fake_responses=[expected_response])
    wrapper = AsyncOpenAICompletion(client, model=expected_model)
    streaming_response = await wrapper(
        messages=messages,
        stream_callback=stream_callback,
    )
    print(streaming_response)
    assert expected_response == response.content
    # +1 to account for the empty chunk at the end with finished=True
    assert len(callback_chunks) == len(range(0, len(expected_response), 4)) + 1
    assert callback_chunks[-1].finish_reason is not None
    assert streaming_response.content == ''.join(x.content for x in callback_chunks)
    assert streaming_response.model == expected_model
    assert streaming_response.created is not None

@pytest.mark.asyncio()
async def test__async_MockOpenAI_object_streaming() -> None:  # noqa
    callback_chunks = []
    async def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    expected_response = "Hello, world!"
    expected_model = 'my-model'
    messages = [{'role': 'user', 'content': expected_response}]
    client = AsyncMockOpenAI(fake_responses=[expected_response])
    wrapper = AsyncOpenAICompletion(
        client,
        model=expected_model,
        logprobs=True,
        temperature=0.8,
        stream_callback=streaming_callback,
    )
    ####
    # Non-streaming example
    ####
    response = await wrapper(messages=messages)
    assert expected_response == response.content
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert callback_chunks[-1].finish_reason is not None

@pytest.mark.asyncio()
async def test__async_MockOpenAI_object__legacy_structure() -> None:  # noqa
    expected_response = "Hello, world!"
    expected_model = 'my-model'
    messages = [{'role': 'user', 'content': expected_response}]
    client = AsyncMockOpenAI(fake_responses=[expected_response], legacy=True)
    wrapper = AsyncOpenAICompletion(
        client,
        model=expected_model,
        logprobs=True,
        temperature=0.8,
    )
    ####
    # Non-streaming example
    ####
    response = await wrapper(messages=messages)
    assert expected_response == response.content
    assert response.model
    assert response.role == 'assistant'
    assert response.created is not None
    assert response.usage is not None
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0

    # test valid parameters for streaming
    callback_chunks = []
    async def stream_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = AsyncMockOpenAI(fake_responses=[expected_response])
    wrapper = AsyncOpenAICompletion(client, model=expected_model)
    streaming_response = await wrapper(
        messages=messages,
        stream_callback=stream_callback,
    )
    print(streaming_response)
    assert expected_response == response.content
    # +1 to account for the empty chunk at the end with finished=True
    assert len(callback_chunks) == len(range(0, len(expected_response), 4)) + 1
    assert callback_chunks[-1].finish_reason is not None
    assert streaming_response.content == ''.join(x.content for x in callback_chunks)
    assert streaming_response.model == expected_model
    assert streaming_response.created is not None

@pytest.mark.asyncio()
async def test__async_MockOpenAI_object_streaming__legacy_structure() -> None:  # noqa
    callback_chunks = []
    async def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    expected_response = "Hello, world!"
    expected_model = 'my-model'
    messages = [{'role': 'user', 'content': expected_response}]
    client = AsyncMockOpenAI(fake_responses=[expected_response], legacy=True)
    wrapper = AsyncOpenAICompletion(
        client,
        model=expected_model,
        logprobs=True,
        temperature=0.8,
        stream_callback=streaming_callback,
    )
    ####
    # Non-streaming example
    ####
    response = await wrapper(messages=messages)
    assert expected_response == response.content
    assert response.content == ''.join(x.content for x in callback_chunks)
    assert callback_chunks[-1].finish_reason is not None

def test_num_tokens():  # noqa
    assert num_tokens(model_name='gpt-3.5-turbo-0613', value="This should be six tokens.") == 6


def test_Function__get_current_weather__to_dict(function_weather: Function):  # noqa
    assert function_weather.to_dict() == {
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

def test_Function__get_current_stocks__to_dict(function_stocks: Function):  # noqa
    assert function_stocks.to_dict() == {
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

def test_OpenAITools(function_weather: Function, function_stocks: Function):  # noqa
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model='gpt-4o-mini')
    ####
    # first interaction
    ####
    messages = [{"role": "user", "content": "What's the weather like in Boston today in degrees F?"}]  # noqa: E501
    response = model(
        messages=messages,
        tools=tools,
    )
    assert isinstance(response, OpenAIToolsResponse)
    assert 'gpt-4o-mini' in response.model
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert len(response.tools) == 1
    assert response.tools[0]['type'] == 'function'
    assert response.tools[0]['name'] == 'get_current_weather'
    assert response.tools[0]['arguments'] == {'location': 'Boston, MA', 'unit': 'fahrenheit'}

def test_OpenAITools__unrelated_prompt__auto(function_weather: Function, function_stocks: Function):  # noqa
    """
    When the prompt is unrelated to any tool and the tool_choice is 'auto', then we will get
    a OpenAICompletionResponse object rather than a OpenAIToolResponse object.
    """
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model='gpt-4o-mini')
    response = model(
        messages= [{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="auto",
    )
    assert isinstance(response, OpenAICompletionResponse)
    assert 'gpt-4o-mini' in response.model
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.content


def test_OpenAITools__unrelated_prompt__required(function_weather: Function, function_stocks: Function):  # noqa
    """
    When the prompt is unrelated to any tool and the tool_choice is 'required', then we will get
    a OpenAIToolResponse object, because `required` means that a tool must be returned.
    """
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model='gpt-4o-mini')
    response = model(
        messages= [{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="required",
    )
    assert isinstance(response, OpenAIToolsResponse)

def test_OpenAITools__unrelated_prompt__none(function_weather: Function, function_stocks: Function):  # noqa
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = OpenAITools(client=OpenAI(), model='gpt-4o-mini')
    response = model(
        messages= [{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="none",
    )
    assert isinstance(response, OpenAICompletionResponse)
    assert 'gpt-4o-mini' in response.model
    assert response.role == 'assistant'
    assert response.finish_reason == 'stop'
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0
    assert response.content
