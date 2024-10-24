"""Tests Mistral classes."""

import os
import pytest
from mistralai import Mistral
from pydantic import BaseModel
from llm_eval.mistralai import (
    MistralAICompletion,
    MistralAICompletionResponse,
    MistralAIResponse,
    MistralAIToolsResponse,
    MistralAITools,
)
from llm_eval.openai import Function


def test__MistralAIResponse__dict_metadata_model_dump():
    response = MistralAIResponse(
        object_name="test name",
        model="test model",
        created=123456,
        metadata={"test_1": "test", "test_2": 1},
        finish_reason="length",
    )
    assert response.model_dump() == {
        "object_name": "test name",
        "model": "test model",
        "created": 123456,
        "metadata": {"test_1": "test", "test_2": 1},
        "finish_reason": "length",
    }


def test__MistralAIResponse__BaseModel_metadata_model_dump():
    class Metadata(BaseModel):
        test_1: str
        test_2: int

    response = MistralAIResponse(
        object_name="test name",
        model="test model",
        created=123456,
        metadata=Metadata(test_1="test", test_2=1),
    )
    assert response.model_dump() == {
        "object_name": "test name",
        "model": "test model",
        "created": 123456,
        "metadata": {"test_1": "test", "test_2": 1},
        "finish_reason": None,
    }

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__MistralAICompletionWrapper() -> None:
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]

    model = MistralAICompletion(
        client=client,
        model="ministral-8b-latest",
        temperature=0.1,
    )
    response = model(messages=messages)
    assert expected_response in response.content.lower()
    assert response.model is not None
    assert response.role == "assistant"
    assert response.finish_reason == "stop"
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0
    assert response.usage["total_tokens"] > 0

    # max tokens == 1
    response = model(messages=messages, max_tokens=1)
    assert len(response.content.split()) == 1
    assert response.model is not None
    assert response.role == "assistant"
    assert response.finish_reason == "length"  # stopped due to max_tokens
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] == 1
    assert response.usage["total_tokens"] > 0

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__MistralAICompletionWrapper__streaming() -> None:
    # test valid parameters for streaming
    callback_chunks = []

    def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    model = MistralAICompletion(
        client=client,
        model="ministral-8b-latest",
        stream_callback=streaming_callback,
    )
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]
    response = model(messages=messages)
    assert expected_response in response.content.lower()
    assert len(callback_chunks) >= 3  # 2 chunks + 1 empty chunk w/ finish_reason
    assert response.content == "".join(x.content for x in callback_chunks)
    assert response.role == "assistant"
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == "stop"
    assert callback_chunks[-1].finish_reason == "stop"

    # max_tokens == 1; only 1 chunk should be returned
    callback_chunks = []

    def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    response = model(messages=messages, max_tokens=1)
    assert len(response.content.split()) == 1
    assert len(callback_chunks) >= 2  # 1 chunk + 1 empty chunk w/ finish_reason
    assert callback_chunks[-1].finish_reason is not None
    assert response.content == "".join(x.content for x in callback_chunks)
    assert response.role == "assistant"
    assert response.model is not None
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.finish_reason == "length"
    assert callback_chunks[-1].finish_reason == "length"

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__Function__get_current_weather__to_dict(function_weather: Function):
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

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__Function__get_current_stocks__to_dict(function_stocks: Function):
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

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__MistralAITools(function_weather: Function, function_stocks: Function):
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = MistralAITools(
        client=Mistral(api_key=os.getenv("MISTRAL_API_KEY")),
        model="ministral-8b-latest",
    )
    ####
    # first interaction
    ####
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in degrees F?",
        },
    ]
    response = model(
        messages=messages,
        tools=tools,
    )
    assert isinstance(response, MistralAIToolsResponse)
    assert "ministral-8b-latest" in response.model
    assert response.role == "assistant"
    assert response.finish_reason == "tool_calls"
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0
    assert response.usage["total_tokens"] > 0
    assert len(response.tools) == 1
    assert response.tools[0]["type"] == "function"
    assert response.tools[0]["name"] == "get_current_weather"
    assert "Boston" in response.tools[0]["arguments"]["location"]
    assert "fahrenheit" in response.tools[0]["arguments"]["unit"]

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__MistralAITools__unrelated_prompt__auto(
    function_weather: Function,
    function_stocks: Function,
) -> None:
    """
    When the prompt is unrelated to any tool and the tool_choice is 'auto', then we will get
    a MistralAICompletionResponse object rather than a MistralAIToolResponse object.
    """
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = MistralAITools(
        client=Mistral(api_key=os.getenv("MISTRAL_API_KEY")),
        model="ministral-8b-latest",
    )
    response = model(
        messages=[{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="auto",
    )
    assert isinstance(response, MistralAICompletionResponse)
    assert "ministral-8b-latest" in response.model
    assert response.role == "assistant"
    assert response.finish_reason == "stop"
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0
    assert response.usage["total_tokens"] > 0
    assert response.content

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__MistralAITools__unrelated_prompt__required(
    function_weather: Function,
    function_stocks: Function,
) -> None:
    """
    When the prompt is unrelated to any tool and the tool_choice is 'required', then we will get
    a MistralAIToolResponse object, because `required` means that a tool must be returned.
    """
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = MistralAITools(
        client=Mistral(api_key=os.getenv("MISTRAL_API_KEY")),
        model="ministral-8b-latest",
    )
    response = model(
        messages=[{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="required",
    )
    assert isinstance(response, MistralAIToolsResponse)

@pytest.mark.skipif('MISTRAL_API_KEY' not in os.environ, reason="MISTRAL_API_KEY not set in environment variables")  # noqa
def test__MistralAITools__unrelated_prompt__none(
    function_weather: Function,
    function_stocks: Function,
):
    tools = [
        function_weather.to_dict(),
        function_stocks.to_dict(),
    ]
    model = MistralAITools(
        client=Mistral(api_key=os.getenv("MISTRAL_API_KEY")),
        model="ministral-8b-latest",
    )
    response = model(
        messages=[{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice="none",
    )
    assert isinstance(response, MistralAICompletionResponse)
    assert "ministral-8b-latest" in response.model
    assert response.role == "assistant"
    assert response.finish_reason == "stop"
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0
    assert response.usage["total_tokens"] > 0
    assert response.content
