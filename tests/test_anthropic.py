"""Tests Anthropic classes."""

import os
import pytest
from anthropic import Anthropic
from pydantic import BaseModel
from sik_llm_eval.anthropic import (
    AnthropicCompletion,
    AnthropicCompletionResponse,
    AnthropicResponse,
    AnthropicToolsResponse,
    AnthropicTools,
)
from sik_llm_eval.openai import Function


def test__AnthropicResponse__dict_metadata_model_dump():
    response = AnthropicResponse(
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


def test__AnthropicResponse__BaseModel_metadata_model_dump():
    class Metadata(BaseModel):
        test_1: str
        test_2: int

    response = AnthropicResponse(
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

@pytest.mark.skipif('ANTHROPIC_API_KEY' not in os.environ, reason="ANTHROPIC_API_KEY not set in environment variables")  # noqa
def test__AnthropicCompletionWrapper() -> None:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat: '{expected_response}'"}]

    model = AnthropicCompletion(
        client=client,
        model="claude-3-haiku-20240307",
        temperature=0.1,
    )
    response = model(messages=messages)
    assert expected_response in response.content.lower()
    assert response.model is not None
    assert response.role == "assistant"
    assert response.finish_reason == "end_turn"
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["input_tokens"] > 0
    assert response.usage["output_tokens"] > 0
    assert response.usage["total_tokens"] > 0

    # max tokens == 1
    response = model(messages=messages, max_tokens=1)
    assert len(response.content.split()) == 1
    assert response.model is not None
    assert response.role == "assistant"
    assert response.finish_reason == "max_tokens"  # stopped due to max_tokens
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["input_tokens"] > 0
    assert response.usage["output_tokens"] == 1
    assert response.usage["total_tokens"] > 0

@pytest.mark.skipif('ANTHROPIC_API_KEY' not in os.environ, reason="ANTHROPIC_API_KEY not set in environment variables")  # noqa
def test__AnthropicCompletionWrapper__streaming() -> None:
    # test valid parameters for streaming
    callback_chunks = []

    def streaming_callback(record) -> None:  # noqa: ANN001
        nonlocal callback_chunks
        callback_chunks.append(record)

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = AnthropicCompletion(
        client=client,
        model="claude-3-haiku-20240307",
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
    assert response.finish_reason == "end_turn"
    assert callback_chunks[-1].finish_reason == "end_turn"

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
    assert response.finish_reason == "max_tokens"
    assert callback_chunks[-1].finish_reason == "max_tokens"

@pytest.mark.skipif('ANTHROPIC_API_KEY' not in os.environ, reason="ANTHROPIC_API_KEY not set in environment variables")  # noqa
def test__Function__get_weather__to_dict(weather_tool: Function):
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

@pytest.mark.skipif('ANTHROPIC_API_KEY' not in os.environ, reason="ANTHROPIC_API_KEY not set in environment variables")  # noqa
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

@pytest.mark.skipif('ANTHROPIC_API_KEY' not in os.environ, reason="ANTHROPIC_API_KEY not set in environment variables")  # noqa
def test__AnthropicTools():
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "object",
                        "description": "The units in Fahrenheit or Celsius",
                        "enum": ["fahrenheit", "celsius"],
                    },
                },
                "required": ["location"],
            },
        },
    ]
    model = AnthropicTools(
        client=Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        model="claude-3-haiku-20240307",
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
        tool_choice={"type": "auto"},
    )
    assert isinstance(response, AnthropicToolsResponse)
    assert "claude-3-haiku-20240307" in response.model
    assert response.role == "assistant"
    assert response.finish_reason == "tool_use"
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["input_tokens"] > 0
    assert response.usage["output_tokens"] > 0
    assert response.usage["total_tokens"] > 0
    assert len(response.tools) == 1
    assert response.tools[0]["type"] == "tool_use"
    assert response.tools[0]["name"] == "get_weather"
    assert "Boston" in response.tools[0]["arguments"]["location"]
    assert "fahrenheit" in response.tools[0]["arguments"]["unit"]

@pytest.mark.skipif('ANTHROPIC_API_KEY' not in os.environ, reason="ANTHROPIC_API_KEY not set in environment variables")  # noqa
def test__AnthropicTools__unrelated_prompt__auto() -> None:
    """
    When the prompt is unrelated to any tool and the tool_choice is 'auto', then we will get
    a AnthropicCompletionResponse object rather than a AnthropicToolResponse object.
    """
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        },
    ]
    model = AnthropicTools(
        client=Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        model="claude-3-haiku-20240307",
    )
    response = model(
        messages=[{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice={"type": "auto"},
    )
    assert isinstance(response, AnthropicCompletionResponse)
    assert "claude-3-haiku-20240307" in response.model
    assert response.role == "assistant"
    assert response.finish_reason == "end_turn"
    assert response.created is not None
    assert response.duration_seconds > 0
    assert response.usage is not None
    assert response.usage["input_tokens"] > 0
    assert response.usage["output_tokens"] > 0
    assert response.usage["total_tokens"] > 0
    assert response.content

@pytest.mark.skipif('ANTHROPIC_API_KEY' not in os.environ, reason="ANTHROPIC_API_KEY not set in environment variables")  # noqa
def test__AnthropicTools__unrelated_prompt__required() -> None:
    """
    When the prompt is unrelated to any tool and the tool_choice is 'required', then we will get
    a AnthropicToolResponse object, because `required` means that a tool must be returned.
    """
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        },
    ]
    model = AnthropicTools(
        client=Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        model="claude-3-haiku-20240307",
    )
    response = model(
        messages=[{"role": "user", "content": "How's it going?"}],
        tools=tools,
        tool_choice={"type": "tool", "name": "get_weather"},
    )
    assert isinstance(response, AnthropicToolsResponse)
